import json
import warnings
from copy import copy

import numpy as np
import pandas as pd
import polars as pl
import yaml
import logging
import platform
import pickle
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.stats import sem, ttest_ind, ttest_1samp
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed

from .aggregate_ephys_funcs import get_responses_by_pip_and_condition, run_decoding, parse_args, plot_aggr_cm
from .population_analysis_funcs import PopPCA

from ..behaviour import get_all_cond_filts

from ..io_utils import extract_date
from ..paths import posix_from_win

from ..plotting import plot_shaded_error_ts, format_axis, get_sorted_psth_matrix, plot_sorted_psth_matrix
from ..stats import save_stats_to_tex
from .spike_time_utils import zscore_by_trial
from .unit_analysis import UnitAnalysis

def rolling_mean_convolve(arr, window):
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def padded_rolling_mean(arr, window):
    _arr = arr.copy()
    _arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 1:
        mean_vals = rolling_mean_convolve(arr.flatten(), window)
        pad = np.full(window - 1, np.nan)
        return np.concatenate((pad, mean_vals))
    else:
        mean_vals = [rolling_mean_convolve(e, window) for e in _arr]
        _pad = [np.full(window - 1, np.nan) for e in _arr]
        _padded = [np.concatenate((e, ee)) for e, ee in zip(_pad, mean_vals)]
        return np.concatenate(_padded, axis=0).reshape(arr.shape)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

    # Load config
    config_path = Path(args.config_file)
    if config_path.is_file():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded config from {config_path}")
    else:
        logging.warning(f"Config {config_path} not found. Continuing without.")

    ceph_dir = config['ceph_dir_' + platform.system().lower()]

    # Load plot config if provided
    plot_config = {}
    if args.plot_config_path:
        plot_path = Path(args.plot_config_path)
        if plot_path.is_file():
            with open(plot_path, 'r') as f:
                plot_config = yaml.safe_load(f)
            logging.info(f"Loaded plot config from {plot_path}")
        else:
            logging.warning(f"Plot config {plot_path} not found. Continuing without.")

    pkl_dir = Path(args.event_responses_pkl).parent

    # Set figure directories
    psth_figdir = ceph_dir / posix_from_win(plot_config.get('psth_figdir', r'X:\Dammy\figures\psth_analysis'))
    decoding_figdir = ceph_dir / posix_from_win(plot_config.get('decoding_figdir', r'X:\Dammy\figures\rare_freq_decoding'))
    pca_fidir = ceph_dir / posix_from_win(plot_config.get('pca_figdir', r'X:\Dammy\figures\pca_plots'))

    for fig_dir in [decoding_figdir, pca_fidir, psth_figdir]:
        if not fig_dir.is_dir():
            fig_dir.mkdir(parents=False)

    mpl.style.use('figure_stylesheet.mplstyle')

    all_stim_resps = AggregateSession(pkl_dir, args, plot_config, ['A-0', 'X', 'base'])

    all_stim_resps.aggregate_sess_decoding('stim_decoding', df_save_path=decoding_figdir / 'stim_decoding_df.h5')
    all_stim_resps.plot_decoder_boxplot(decoding_figdir)

    for pips2decode in plot_config['stim_decoding']['pips2decode']:
        all_stim_resps.decoding_ttest('_vs_'.join(pips2decode), 'data', 'shuffled')

    for ttest_name, ttest_res in all_stim_resps.ttest_res.items():
        save_stats_to_tex(ttest_res, decoding_figdir / f'{ttest_name}.tex')

    logging.info("All done.")


class ConcatResponses:
    def __init__(self, batch_event_responses: dict, event_features: dict, pips, plot_config: dict, zscore_flag=False, **psth_kwargs):
        self.concatenated_event_sessnames = None
        self.peak_ts_by_pips = None
        self.event_mats_4_sorting = None
        self.concatenated_event_responses_sem = None
        self.concatenated_event_responses = None
        self.smoothed_event_responses = None
        self.smoothed_event_responses_sem = None

        self.event_features = event_features

        self.get_concatenated_event_responses(batch_event_responses, zscore_flag=zscore_flag)
        self.get_smoothed_responses(batch_event_responses, psth_kwargs.get('smoothing_window', 25))

    def get_concatenated_event_responses(self, batch_event_responses: dict, zscore_flag=False):
        concatenated_event_responses = {}
        concatenated_event_responses_sem = {}
        concatenated_event_sessnames = {}

        if zscore_flag:
            batch_event_responses = zscore_by_trial(batch_event_responses)

        for pip in list(batch_event_responses.values())[0]:
            concatenated_event_sessnames[pip] = np.concatenate(
                [[e] * batch_event_responses[e][pip].shape[1] for e in batch_event_responses], axis=0
            )
            concatenated_event_responses[pip] = np.concatenate(
                [batch_event_responses[e][pip].mean(axis=0) for e in batch_event_responses], axis=0
            )
            concatenated_event_responses_sem[pip] = np.concatenate(
                [sem(batch_event_responses[e][pip]) for e in batch_event_responses], axis=0
            )

        self.concatenated_event_responses = concatenated_event_responses
        self.concatenated_event_responses_sem = concatenated_event_responses_sem
        self.concatenated_event_sessnames = concatenated_event_sessnames

    def get_sorted_resp_mat(self, batch_event_responses: dict, pips_2_plot: list, window: tuple, **psth_plot_kwargs):
        resp_mats_by_pips = {}
        peak_ts_by_pips = {}

        for sessname in list(batch_event_responses.keys()):
            if not all([e in list(batch_event_responses[sessname].keys()) for e in pips_2_plot]):
                logging.warning(f'Skipping {sessname} due to missing data')
                batch_event_responses.pop(sessname)
                continue

        for sessname in list(batch_event_responses.keys()):
            if any([len(batch_event_responses[sessname][pip]) < 4 for pip in pips_2_plot]):
                logging.warning(f'Skipping {sessname} due to insufficient data')
                batch_event_responses.pop(sessname)
                continue

        if len(batch_event_responses) == 0:
            return None

        for pip in pips_2_plot:
            kwargs = dict(window=window, sessname_filter=None, **psth_plot_kwargs)
            _, batch_resp_mat, peak_ts, _ = get_sorted_psth_matrix(batch_event_responses, pip, pip, **kwargs)
            resp_mats_by_pips[pip] = batch_resp_mat
            peak_ts_by_pips[pip] = peak_ts

        self.event_mats_4_sorting = resp_mats_by_pips
        self.peak_ts_by_pips = peak_ts_by_pips

    def get_smoothed_responses(self, batch_event_responses: dict, smoothing_window=25):
        concatenated_event_responses = {}
        concatenated_event_responses_sem = {}

        smoothed_responses = {}
        for sessname in list(batch_event_responses.keys()):
            smoothed_responses[sessname] = {}
            for pip in list(batch_event_responses[sessname].keys()):
                smoothed_responses[sessname][pip] = padded_rolling_mean(
                    batch_event_responses[sessname][pip], window=smoothing_window
                )

        for pip in list(smoothed_responses.values())[0]:
            concatenated_event_responses[pip] = np.concatenate(
                [smoothed_responses[e][pip].mean(axis=0) for e in smoothed_responses], axis=0
            )
            concatenated_event_responses_sem[pip] = np.concatenate(
                [sem(smoothed_responses[e][pip]) for e in smoothed_responses], axis=0
            )

        self.smoothed_event_responses = concatenated_event_responses
        self.smoothed_event_responses_sem = concatenated_event_responses_sem


class AggregateSession:
    def __init__(self, dataset, all_td_df_h5_path, plot_config, pips_2_plot=None):
        self.dataset = dataset
        self.plot_config = plot_config

        self.pca = {}
        self.ttest_res = {}
        self.plots = {}
        self.decoder_name = None
        self.aggregate_decoding_df = None
        self.cms = None

        self.pips_2_plot = pips_2_plot if pips_2_plot is not None else plot_config['pips_2_plot']
        self.window = plot_config['window']
        dt = plot_config.get('dt', 0.01)
        self.x_ser = np.round(np.arange(self.window[0], self.window[1] + dt, dt), 2)

        self.concatenated_event_responses = None
        self.concatenated_event_responses_sem = None
        self.smoothed_concatenated_responses = None
        self.smoothed_concatenated_responses_sem = None
        self.concatenated_sessnames = None
        self.event_features = {}

        self.all_td_df: pd.DataFrame = pd.read_hdf(all_td_df_h5_path)

    def fetch_event_features_polars(self, sessions: list[str], events: list[str]) -> pl.DataFrame:
        """Fast multi-session event feature loading using Polars lazy scanning."""
        events_lazy = pl.scan_parquet(self.dataset.root / "events" / "**" / "*.parquet")
        return (
            events_lazy.filter(
                (pl.col("session_id").is_in(sessions)) &
                (pl.col("event_name").is_in(events))
            ).collect()
        )

    @staticmethod
    def _process_single_session_worker(sess, raw_events, dataset, events_df_slice, aggr_sess_kwargs):
        """Worker function executed in parallel for a single session."""
        session_resps = {sess: {}}
        for event in raw_events:
            pop_tensors = dataset.get_population_tensor([sess], event_name=event)
            if sess in pop_tensors:
                session_resps[sess][event] = pop_tensors[sess]

        if not session_resps[sess]:
            return None

        aggr_sess_instance = aggr_sess_kwargs['aggr_sess_instance']
        conds = aggr_sess_kwargs.get('conds')
        cond_filts = aggr_sess_kwargs.get('cond_filts')
        pips_2_plot = aggr_sess_kwargs.get('pips_2_plot')

        subsetted = aggr_sess_instance.subset_responses(
            session_responses=session_resps,
            events_df=events_df_slice,
            conds=conds,
            cond_filts=cond_filts,
            **aggr_sess_kwargs.get('kwargs', {})
        )

        if not subsetted or sess not in subsetted:
            return None

        sess_resps = subsetted[sess]

        if not all(pip in sess_resps for pip in pips_2_plot):
            return None
        if any(len(sess_resps[pip]) < 4 for pip in pips_2_plot):
            return None

        return sess, sess_resps

    def aggregate_mean_sess_responses_parallel(self, tag=None, conds=None, concat_savename=None, n_jobs=-1, **kwargs):
        """Parallelized response aggregation exploiting per-session Zarr & Parquet storage architecture."""
        if tag is None:
            tag = 'all'

        cond_filts = get_all_cond_filts()
        pips_2_plot = copy(self.pips_2_plot)

        if conds is not None:
            pips_2_plot = [f'{pip}_{cond}' for pip in pips_2_plot for cond in conds]

        if concat_savename is not None and kwargs.get('reload_save', True):
            if self._load_aggr_means(concat_savename):
                return None

        target_sessions = kwargs.get('sessions', self.plot_config.get('sessions', []))
        if not target_sessions and not self.dataset.registry.empty:
            target_sessions = self.dataset.registry['session_id'].astype(str).tolist()

        excluded = set(self.plot_config.get('excluded_sessions', []))
        sessions = [s for s in target_sessions if s not in excluded]

        raw_events = list(set([p.split('_')[0] for p in pips_2_plot]))

        events_df = self.fetch_event_features_polars(sessions, raw_events)

        worker_kwargs = {
            'aggr_sess_instance': self,
            'conds': conds,
            'cond_filts': cond_filts,
            'pips_2_plot': pips_2_plot,
            'kwargs': kwargs
        }

        logging.info(f"Extracting and subsetting {len(sessions)} sessions using n_jobs={n_jobs}...")

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(self._process_single_session_worker)(
                sess,
                raw_events,
                self.dataset,
                events_df.filter(pl.col("session_id") == sess) if events_df is not None else None,
                worker_kwargs
            )
            for sess in sessions
        )

        batch_responses = {sess: resps for res in results if res is not None for sess, resps in [res]}

        if len(batch_responses) == 0:
            logging.info(f"No valid responses remaining after parallel processing for conditions: {conds}")
            return None

        resp_obj = ConcatResponses(
            batch_event_responses=batch_responses,
            event_features=self.event_features,
            pips=pips_2_plot,
            plot_config=self.plot_config,
            zscore_flag=kwargs.get('zscore_flag', False),
            **self.plot_config.get('psth_plot_kwargs', {})
        )

        if resp_obj is None or resp_obj.concatenated_event_responses is None:
            logging.warning("Failed to aggregate responses in ConcatResponses.")
            return None

        self.peak_ts_by_pips = resp_obj.peak_ts_by_pips
        self.event_mats_4_sorting = resp_obj.event_mats_4_sorting
        self.concatenated_event_responses = resp_obj.concatenated_event_responses
        self.concatenated_event_responses_sem = resp_obj.concatenated_event_responses_sem
        self.smoothed_concatenated_responses = resp_obj.smoothed_event_responses
        self.smoothed_concatenated_responses_sem = resp_obj.smoothed_event_responses_sem
        self.concatenated_sessnames = resp_obj.concatenated_event_sessnames[pips_2_plot[0]]

        logging.info(f"{pips_2_plot} shape: {[e.shape for e in self.concatenated_event_responses.values()]}")

        if concat_savename:
            self._save_aggr_means(concat_savename)

    @staticmethod
    def _process_decoding_session_worker(sess, raw_events, dataset, sess_events_df, worker_kwargs):
        """Parallel worker to process, splice, and run decoding on a single session's data."""
        aggr_inst = worker_kwargs['aggr_sess_instance']
        conds = worker_kwargs['conds']
        cond_filts = worker_kwargs['cond_filts']
        dec_tag = worker_kwargs['dec_tag']
        pips2decode = worker_kwargs['pips2decode']
        decoding_window = worker_kwargs['decoding_window']
        shuffled_cm_flag = worker_kwargs['shuffled_cm_flag']
        splice_responses = worker_kwargs['splice_responses']
        splice_windows = worker_kwargs['splice_windows']
        new_names = worker_kwargs['new_names']
        old_x_ser = worker_kwargs['old_x_ser']
        new_x_ser = worker_kwargs['new_x_ser']
        kwargs = worker_kwargs['kwargs']

        batch_responses = aggr_inst.extract_and_subset_responses(
            sess=sess,
            raw_events=raw_events,
            dataset=dataset,
            sess_events_df=sess_events_df,
            conds=conds,
            cond_filts=cond_filts,
            **kwargs
        )

        if not batch_responses or len(batch_responses) == 0:
            logging.info(f"No responses found for conditions {conds} in session {sess}")
            return None

        if splice_responses:
            batch_responses = aggr_inst.splice_responses(
                batch_responses, splice_windows, new_names, old_x_ser
            )
            current_x_ser = new_x_ser
        else:
            current_x_ser = aggr_inst.x_ser

        dec_results = run_decoding(
            batch_responses,
            current_x_ser,
            decoding_window,
            pips2decode,
            overwrite=True,
            **aggr_inst.plot_config[dec_tag].get('decoding_kwargs', {})
        )

        if shuffled_cm_flag:
            decode_dfs, cms, cms_shuffle = dec_results
        else:
            decode_dfs, cms = dec_results
            cms_shuffle = None

        return decode_dfs, cms, cms_shuffle

    def aggregate_sess_decoding_parallel(self, dec_tag, conds=None, n_jobs=-1, **kwargs):
        """Parallelized decoding aggregation exploiting per-session storage."""
        cond_filts = get_all_cond_filts()
        pips2decode = kwargs.get('pips2decode', self.plot_config[dec_tag]['pips2decode'])
        decoding_window = self.plot_config[dec_tag]['decoding_window']

        df_loaded, cms_loaded = False, False

        if kwargs.get('df_save_path'):
            df_save_path = Path(kwargs.get('df_save_path'))
            cm_save_path = df_save_path.with_name(df_save_path.name.replace('df.h5', 'cm.npy'))

            if kwargs.get('reload_save', True) and df_save_path.is_file():
                self.aggregate_decoding_df = pd.read_hdf(df_save_path)
                df_loaded = True
            if kwargs.get('reload_save', True) and cm_save_path.is_file():
                self.cms = np.load(cm_save_path)
                cms_loaded = True

        if all([df_loaded, cms_loaded]):
            self.decoder_name = dec_tag
            logging.info(f"Decoding {dec_tag} reloaded from disk.")
            return

        assert decoding_window is not None, f"Decoding window must be specified in {self.plot_config} or in {kwargs}"

        shuffled_cm_flag = self.plot_config[dec_tag].get('decoding_kwargs', {}).get('return_shuffled_cms', False)
        splice_responses = kwargs.get('splice_responses', False)

        if splice_responses:
            old_x_ser = self.x_ser
            splice_windows = kwargs.get('splice_windows')
            new_names = kwargs.get('new_names')
            new_x_ser = kwargs.get('new_x_ser')
            assert all([e is not None for e in [splice_windows, new_names, new_x_ser]])
        else:
            old_x_ser, splice_windows, new_names, new_x_ser = None, None, None, None

        target_sessions = kwargs.get('sessions', self.plot_config.get('sessions', []))
        if not target_sessions and not self.dataset.registry.empty:
            target_sessions = self.dataset.registry['session_id'].astype(str).tolist()

        excluded = set(self.plot_config.get('excluded_sessions', []))
        sessions = [s for s in target_sessions if s not in excluded]

        raw_events = list(set([p.split('_')[0] for p in pips2decode]))

        events_df = self.fetch_event_features_polars(sessions, raw_events)

        worker_kwargs = {
            'aggr_sess_instance': self,
            'conds': conds,
            'cond_filts': cond_filts,
            'dec_tag': dec_tag,
            'pips2decode': pips2decode,
            'decoding_window': decoding_window,
            'shuffled_cm_flag': shuffled_cm_flag,
            'splice_responses': splice_responses,
            'splice_windows': splice_windows,
            'new_names': new_names,
            'old_x_ser': old_x_ser,
            'new_x_ser': new_x_ser,
            'kwargs': kwargs
        }

        logging.info(f"Decoding across {len(sessions)} sessions using n_jobs={n_jobs}...")

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._process_decoding_session_worker)(
                sess,
                raw_events,
                self.dataset,
                events_df.filter(pl.col("session_id") == sess) if events_df is not None else None,
                worker_kwargs
            )
            for sess in sessions
        )

        results = [res for res in results if res is not None]

        if not results:
            logging.warning(f"No decoding results generated across any session for conditions: {conds}")
            return None

        batch_decoding_dfs = [res[0] for res in results]
        batch_cms = [res[1] for res in results if res[1] is not None and res[1].ndim == 3]
        batch_cms_shuffled = [res[2] for res in results if res[2] is not None]

        self.aggregate_decoding_df = pd.concat(batch_decoding_dfs, axis=0)
        self.cms = np.concatenate(batch_cms, axis=0)
        if shuffled_cm_flag and batch_cms_shuffled:
            self.cms_shuffled = np.concatenate(batch_cms_shuffled, axis=0)

        self.decoder_name = dec_tag

        if kwargs.get('df_save_path'):
            df_save_path = Path(kwargs.get('df_save_path'))
            cm_save_path = df_save_path.with_name(df_save_path.name.replace('df.h5', 'cm.npy'))

            self.aggregate_decoding_df.to_hdf(df_save_path, 'df')
            np.save(cm_save_path, self.cms)
            if shuffled_cm_flag:
                np.save(cm_save_path.with_stem(f'{cm_save_path.stem}_shuffle'), self.cms_shuffled)

    @staticmethod
    def splice_responses(batch_responses: dict, windows: list, new_names: list, x_ser: np.ndarray) -> dict:
        """Splices response time series arrays across specified time windows."""
        windows_idxs = [[np.where(x_ser == t)[0][0] for t in _window] for _window in windows]
        spliced_batch_responses = {}

        for sess, resps_dict in batch_responses.items():
            spliced_responses = []
            for resp in resps_dict.values():
                _spliced = [resp[:, :, w_start:w_end] for w_start, w_end in windows_idxs]
                spliced_responses.extend(_spliced)

            assert len(spliced_responses) == len(new_names), (
                f"Spliced responses length ({len(spliced_responses)}) "
                f"does not match new_names length ({len(new_names)})"
            )

            spliced_batch_responses[sess] = dict(zip(new_names, spliced_responses))

        return spliced_batch_responses

    def subset_responses(self, session_responses, events_df=None, conds=None, cond_filts=None, sessname_filts=None, **kwargs):
        """Subsets neural response matrices (3D: trials x units x time) per session and event."""
        if sessname_filts is not None:
            session_responses = self._filter_by_session_names(session_responses, sessname_filts)
            if not session_responses:
                return session_responses

        if conds is not None:
            assert cond_filts is not None, "cond_filts dictionary must be provided when conds is set."
            filtered_responses = {}
            for sess, sess_resps in session_responses.items():
                filtered_resps = self._filter_session_by_conditions(
                    sess, sess_resps, conds, cond_filts, events_df
                )
                if filtered_resps:
                    filtered_responses[sess] = filtered_resps
            session_responses = filtered_responses

        session_responses = {
            sess: resps for sess, resps in session_responses.items()
            if resps and all(r.shape[0] > 0 for r in resps.values())
        }

        if not session_responses:
            logging.info("No responses remaining after condition filtering.")
            return session_responses

        if kwargs.get('filt_by_trial_num', False):
            session_responses = self._filter_by_trial_numbers(
                session_responses, events_df, cond_filts=cond_filts, **kwargs
            )

        return session_responses

    def _filter_by_session_names(self, session_responses, sessname_filts):
        """Filters session dictionary keys against allowed session strings/patterns."""
        if isinstance(sessname_filts, (str, set)):
            allowed = {sessname_filts} if isinstance(sessname_filts, str) else sessname_filts
            return {s: res for s, res in session_responses.items() if s in allowed}
        elif callable(sessname_filts):
            return {s: res for s, res in session_responses.items() if sessname_filts(s)}
        return session_responses

    def _filter_session_by_conditions(self, sess, sess_resps, conds, cond_filts, events_df=None):
        """Applies condition filters to session event tensors using Polars or pandas metadata."""
        df_source = events_df if events_df is not None else getattr(self, 'all_td_df', None)
        if df_source is None:
            logging.warning(f"No metadata dataframe available for filtering session {sess}.")
            return {}

        if isinstance(self.all_td_df.index, pd.MultiIndex):
            sess_level = "sess_id" if "sess_id" in self.all_td_df.index.names else "session_id"
            trial_level = "trial_num" if "trial_num" in self.all_td_df.index.names else "trial"
            try:
                sess_td_df = self.all_td_df.xs(sess, level=sess_level, drop_level=False)
            except KeyError:
                return {}
        else:
            sess_col = "session_id" if "session_id" in self.all_td_df.columns else "sess_id"
            trial_level = "trial_num" if "trial_num" in self.all_td_df.columns else "trial"
            sess_td_df = self.all_td_df[self.all_td_df[sess_col] == sess]

        if sess_td_df.empty:
            return {}

        filtered_resps = {}

        if isinstance(df_source, pl.DataFrame):
            sess_events_df = df_source.filter(pl.col("session_id") == sess)
        else:
            sess_events_df = df_source[df_source["session_id"] == sess]

        for raw_event, tensor in sess_resps.items():
            if isinstance(sess_events_df, pl.DataFrame):
                event_col = "event_name" if "event_name" in sess_events_df.columns else "event_type"
                event_df = (
                    sess_events_df.filter(pl.col(event_col) == raw_event)
                    if event_col in sess_events_df.columns
                    else sess_events_df
                )
            else:
                event_col = "event_name" if "event_name" in sess_events_df.columns else "event_type"
                event_df = (
                    sess_events_df[sess_events_df[event_col] == raw_event]
                    if event_col in sess_events_df.columns
                    else sess_events_df
                )

            for cond in conds:
                pip_label = f"{raw_event}_{cond}"

                if cond not in cond_filts:
                    continue

                filt = cond_filts[cond]

                try:
                    matching_td = sess_td_df.query(filt)
                    if isinstance(matching_td.index, pd.MultiIndex):
                        filt_trial_nums = matching_td.index.get_level_values(trial_level).values
                    else:
                        filt_trial_nums = matching_td[trial_level].values
                except Exception as e:
                    logging.debug(f"Query '{filt}' failed for session {sess}: {e}")
                    continue

                if len(filt_trial_nums) == 0:
                    continue

                filt_trial_nums = np.array(filt_trial_nums, dtype=np.int64)

                if isinstance(event_df, pl.DataFrame):
                    t_col = "trial_num" if "trial_num" in event_df.columns else "trial"
                    valid_indices = (
                        event_df.with_row_index("row_idx")
                        .filter(pl.col(t_col).cast(pl.Int64).is_in(filt_trial_nums))["row_idx"]
                        .to_list()
                    )
                else:
                    t_col = "trial_num" if "trial_num" in event_df.columns else "trial"
                    valid_indices = event_df.index[event_df[t_col].isin(filt_trial_nums)].tolist()

                if valid_indices:
                    filtered_resps[pip_label] = tensor[valid_indices, :, :]

        return filtered_resps

    def _filter_by_trial_numbers(self, session_responses, events_df=None, cond_filts=None, **kwargs):
        """Subsets session matrices to exact trial counts across conditions for balanced sampling."""
        min_trials = kwargs.get('min_trial_num', None)

        for sess, pips_dict in list(session_responses.items()):
            if not pips_dict:
                continue

            counts = [mat.shape[0] for mat in pips_dict.values()]
            if not counts:
                continue

            target_count = min(counts) if min_trials is None else min_trials

            if min_trials is not None and target_count < min_trials:
                session_responses.pop(sess, None)
                continue

            for pip, mat in pips_dict.items():
                pips_dict[pip] = mat[:target_count, :, :]

        return session_responses

    def _save_aggr_means(self, concat_savename: Path):
        savename_stem = concat_savename.stem
        joblib.dump(self.concatenated_event_responses, concat_savename)
        joblib.dump(self.concatenated_event_responses_sem, concat_savename.with_stem(f'{savename_stem}_sem'))
        joblib.dump(self.concatenated_sessnames, concat_savename.with_stem(f'{savename_stem}_sessnames'))

    def _load_aggr_means(self, concat_savename: Path) -> bool:
        savename_stem = concat_savename.stem
        if not concat_savename.exists():
            warnings.warn(f'Concatenation {concat_savename} does not exist')
            return False
        self.concatenated_event_responses = joblib.load(concat_savename)
        self.concatenated_event_responses_sem = joblib.load(concat_savename.with_stem(f'{savename_stem}_sem'))
        self.concatenated_sessnames = joblib.load(concat_savename.with_stem(f'{savename_stem}_sessnames'))
        return True


class AggregateVisualizer:
    """Handles plotting, PCA, and statistical visualisations for already concatenated neural data."""

    def __init__(
        self,
        concatenated_event_responses: dict,
        x_ser: np.ndarray,
        window: tuple,
        concatenated_sessnames: np.ndarray = None,
        smoothed_concatenated_responses: dict = None,
        event_mats_4_sorting: dict = None,
        peak_ts_by_pips: dict = None,
        aggregate_decoding_df: pd.DataFrame = None,
        cms: np.ndarray = None,
        decoder_name: str = None
    ):
        self.concatenated_event_responses = concatenated_event_responses
        self.smoothed_concatenated_responses = (
            smoothed_concatenated_responses if smoothed_concatenated_responses is not None
            else concatenated_event_responses
        )
        self.x_ser = x_ser
        self.window = window
        self.concatenated_sessnames = concatenated_sessnames
        self.event_mats_4_sorting = event_mats_4_sorting
        self.peak_ts_by_pips = peak_ts_by_pips

        self.aggregate_decoding_df = aggregate_decoding_df
        self.cms = cms
        self.decoder_name = decoder_name

        self.pca = {}
        self.ttest_res = {}
        self.plots = {}

    def _resolve_pips(self, pips) -> list:
        return pips if pips is not None else list(self.concatenated_event_responses.keys())

    def _save_and_display(self, fig, save_path: Path = None):
        if save_path:
            fig.savefig(save_path)

    def plot_mean_ts(self, figdir: Path, pips=None, **kwargs):
        pips = self._resolve_pips(pips)

        fig = Figure()
        ax = fig.add_subplot(111)

        if kwargs.get('ts_plot_window') is not None:
            plot_t_idxs = [np.where(self.x_ser == t)[0][0] for t in kwargs.get('ts_plot_window')]
            plot_x_ser = self.x_ser[plot_t_idxs[0]:plot_t_idxs[1]]
        else:
            plot_t_idxs = [0, self.x_ser.shape[0]]
            plot_x_ser = self.x_ser

        event_mean_dict = self.smoothed_concatenated_responses if kwargs.get('plot_smoothed_ts', True) else self.concatenated_event_responses

        colours = kwargs.get('plot_cols', [f'C{i}' for i in range(len(pips))])
        name_date_df = pd.DataFrame(
            [(sessname, sessname.split('_')[0], extract_date(sessname)) for sessname in self.concatenated_sessnames],
            columns=['sess', 'name', 'date']
        )

        for pip, col in zip(pips, colours):
            event_dict_df = pd.DataFrame(event_mean_dict[pip], index=pd.MultiIndex.from_frame(name_date_df))
            mean_resp = event_dict_df.groupby('name').mean().mean(axis=0)[plot_t_idxs[0]:plot_t_idxs[1]]
            sem_resp = event_dict_df.groupby('name').mean().sem(axis=0)[plot_t_idxs[0]:plot_t_idxs[1]]

            ax.plot(plot_x_ser, mean_resp, label=pip, c=col, lw=1)
            plot_shaded_error_ts(ax, plot_x_ser, mean_resp, sem_resp, fc=col, alpha=0.1)

        if any(['A' in pip for pip in pips]):
            format_axis(ax, vlines=[0], vspan=[[t, t + 0.15] for t in np.arange(0, min([1, plot_x_ser.max()]), 0.25)])
        else:
            format_axis(ax, vlines=[0])

        ax.legend()
        fig.set_size_inches(kwargs.get('figsize', (2, 1.5)))
        fig.set_layout_engine('tight')

        self._save_and_display(fig, figdir / f'{"_".join(pips)}_mean_ts.pdf')
        self.plots[f'{"_".join(pips)}_mean_ts'] = (fig, ax)

        plot_info = {
            'n_units': [e.shape[0] for e in event_mean_dict.values()],
            'sessnames': np.unique(self.concatenated_sessnames).tolist(),
            'n_sessions': len(np.unique(self.concatenated_sessnames)),
            'names': np.unique([e.split('_')[0] for e in self.concatenated_sessnames]).tolist(),
            'n_names': np.unique([e.split('_')[0] for e in self.concatenated_sessnames]).shape[0]
        }

        with open(figdir / f'{"_".join(pips)}_mean_ts_info.json', 'w') as f:
            json.dump(plot_info, f)

    def plot_sorted_psth_mat(self, pips=None, plot_kwargs=None):
        pips = self._resolve_pips(pips)
        plot_kwargs = plot_kwargs or {}

        for pip in pips:
            sorted_resp_mat = self.event_mats_4_sorting[pip][self.peak_ts_by_pips[pip].argsort()]
            psth_plot = plot_sorted_psth_matrix(sorted_resp_mat, self.x_ser, pip, **plot_kwargs)

            fig, axes = psth_plot[0], psth_plot[1]
            t_max = min(1, plot_kwargs.get('plot_window', self.window)[1])

            format_axis(
                axes[1],
                vlines=([0] if 'A' not in pip else np.arange(0, t_max, 0.25).tolist()),
                ylabel='',
                xlabel=''
            )
            axes[1].set_xticks([])
            format_axis(axes[0], vlines=[0])

            try:
                format_axis(psth_plot[2].ax)
            except Exception:
                pass

            self.plots[f'{pip}_sorted_psth'] = psth_plot

    def scatter_unit_means(self, pips, diff_window, figdir: Path, **kwargs):
        assert len(pips) == 2

        plot_t_idxs = [np.where(self.x_ser == t)[0][0] for t in diff_window]
        unit_resps = [
            self.concatenated_event_responses[pip][:, plot_t_idxs[0]:plot_t_idxs[1]].max(axis=1) for pip in pips
        ]

        fig1 = Figure()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(*unit_resps, alpha=0.02, fc='#1f76b2ff', ec='#1f76b2ff', lw=0.01)
        ax1.set_xlim(*np.percentile(unit_resps, [1, 99]))
        ax1.set_ylim(*np.percentile(unit_resps, [1, 99]))
        format_axis(ax1)

        ax1.locator_params(axis='both', nbins=6)
        ax1.set_xlabel('frequent mean response')
        ax1.set_ylabel('rare mean response')
        ax1.plot(ax1.get_xlim(), ax1.get_ylim(), ls='--', c='k')

        fig1.set_size_inches(2, 2)
        self._save_and_display(fig1, figdir / f'unit_resps_scatter{"_".join(pips)}.pdf')
        fig1.savefig(figdir / f'unit_resps_scatter{"_".join(pips)}.svg')

        fig2 = Figure()
        ax2 = fig2.add_subplot(111)
        cond_diffs_by_unit = unit_resps[0] - unit_resps[1]
        bins2use = np.histogram(cond_diffs_by_unit, bins='fd', density=False)
        ax2.hist(cond_diffs_by_unit, bins=bins2use[1], density=False, alpha=0.9, fc='#a8cfe2ff', ec='k', lw=0.05)
        ax2.set_xlim(*np.percentile(cond_diffs_by_unit, [1, 99]))

        format_axis(ax2, vlines=[0], lw=0.5, ls='--')
        ax2.set_ylabel('Frequency')
        ax2.set_xlabel('\u0394firing rate (rare - frequent)')

        fig2.set_size_inches(2, 2)
        fig2.set_layout_engine('tight')
        self._save_and_display(fig2, figdir / f'unit_resps_hist_by_unit{"_".join(pips)}.pdf')

    def decoding_ttest(self, decoder_name, key1, key2, **ttest_kwargs):
        data_acc = self.aggregate_decoding_df[f'{decoder_name}_{key1}_accuracy'].dropna().values
        if isinstance(key2, float):
            ttest_res = ttest_1samp(data_acc, key2, **ttest_kwargs)
        else:
            shuff_acc = self.aggregate_decoding_df[f'{decoder_name}_{key2}_accuracy'].dropna().values
            ttest_res = ttest_ind(data_acc, shuff_acc, **ttest_kwargs)

        self.ttest_res[f'{decoder_name}_{"_vs_".join([key1, key2])}'] = ttest_res

    def plot_decoder_boxplot(self, decoding_figdir: Path, decs2plot=None):
        boxplot_kwargs = dict(
            widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(lw=1),
            meanprops=dict(mfc='k'), boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5)
        )

        fig = Figure()
        ax = fig.add_subplot(111)
        labels = []

        if decs2plot is None:
            all_dec_cols = self.aggregate_decoding_df.columns.tolist()
            all_data_dec_cols = [col for col in all_dec_cols if 'data_accuracy' in col]
        else:
            if isinstance(decs2plot, str):
                decs2plot = [decs2plot]
            all_data_dec_cols = [f'{dec}_data_accuracy' for dec in decs2plot]

        all_shuffle_dec_cols = [col.replace('data', 'shuffled') for col in all_data_dec_cols]

        for dec_i, (data_name, shuff_name) in enumerate(zip(all_data_dec_cols, all_shuffle_dec_cols)):
            data_acc = self.aggregate_decoding_df[data_name].dropna().values
            shuff_acc = self.aggregate_decoding_df[shuff_name].dropna().values
            lbls = [
                data_name.replace('_data_accuracy', f'\ndata: n {len(data_acc)}'),
                shuff_name.replace('_shuffled_accuracy', f'\nshuffle: n {len(data_acc)}')
            ]

            box = ax.boxplot(
                [data_acc, shuff_acc],
                labels=lbls,
                positions=np.array([-0.3, 0.3]) + dec_i * len(lbls),
                **boxplot_kwargs
            )
            for patch in box['boxes']:
                patch.set_facecolor('white')
            labels.extend(lbls)

        ax.set_ylabel('Decoding accuracy')
        format_axis(ax, hlines=[0.5])
        ax.set_xticks([])

        fig.set_size_inches(0.9 * len(all_data_dec_cols) + 0.15, 1.75)
        fig.set_layout_engine('tight')
        self._save_and_display(fig, decoding_figdir / f'{self.decoder_name}_decoding_accuracy.pdf')

    def plot_confusion_matrix(self, decoding_figdir: Path, cm_config):
        fig, ax = plot_aggr_cm(self.cms, **cm_config)
        self._save_and_display(fig, decoding_figdir / f'{self.decoder_name}_cm.pdf')

    def pca_pseudo_pop(self, pca_name: str, pips=None, standardise=True, by_animal=False, animal=None):
        pips = self._resolve_pips(pips)

        if by_animal:
            names = np.unique([sess.split('_')[0] for sess in self.concatenated_sessnames])
            sess_by_name_mask = {name: [name in sess for sess in self.concatenated_sessnames] for name in names}
            dict_for_pca = {
                'by_class': {
                    f'{pip}': self.concatenated_event_responses[pip][sess_by_name_mask[animal]] for pip in pips
                }
            }
            pca_name = f'{pca_name}_{animal}'
        else:
            dict_for_pca = {'by_class': {pip: self.concatenated_event_responses[pip] for pip in pips}}

        pca = PopPCA(dict_for_pca)
        pca.get_trial_averaged_pca(standardise=standardise)
        pca.get_projected_pca_ts(standardise=standardise)
        self.pca[pca_name] = pca

    def plot_3d_pca(self, pca_name, pca_comps_2plot, figdir, pca_kwargs):
        pca = self.pca[pca_name]
        pca.plot_3d_pca_ts(
            'by_class', self.window, x_ser=self.x_ser, pca_comps_2plot=pca_comps_2plot, **pca_kwargs['plot_kwargs']
        )

        fig, ax = pca.proj_3d_plot
        ax.get_legend().remove()
        if pca_kwargs['fig_kwargs'].get('figsize') is not None:
            fig.set_size_inches(*pca_kwargs['fig_kwargs']['figsize'])

        fig.savefig(figdir / f'{pca_name}_pca_{"_".join(list(map(str, pca_comps_2plot)))}.pdf')

    def plot_1d_pca(self, pca_name, pca_comp, figdir, pca_kwargs):
        pca = self.pca[pca_name]
        pca.plot_1d_pca_ts('by_class', self.window, x_ser=self.x_ser, pca_comp=pca_comp, **pca_kwargs['plot_kwargs'])

        fig, ax = pca.pca_ts_plot
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        format_axis(ax)
        if pca_kwargs['fig_kwargs'].get('figsize') is not None:
            fig.set_size_inches(*pca_kwargs['fig_kwargs']['figsize'])

        fig.savefig(figdir / f'{pca_name}_1d_pca_PC{int(pca_comp)}.pdf')

    def plot_2d_pca(self, pca_name, pca_comps_2plot, figdir, pca_kwargs):
        pca = self.pca[pca_name]
        pca.plot_2d_pca_ts(
            'by_class', self.window, x_ser=self.x_ser, pca_comps_2plot=pca_comps_2plot, **pca_kwargs['plot_kwargs']
        )
        fig, ax = pca.proj_2d_plot
        format_axis(ax)
        if pca_kwargs['fig_kwargs'].get('figsize') is not None:
            fig.set_size_inches(*pca_kwargs['fig_kwargs']['figsize'])

        fig.savefig(figdir / f'{pca_name}_2d_pca_{"_".join(list(map(str, pca_comps_2plot)))}.pdf')

    def scatter_pca(self, pca_name, t_s, pca_comps_2plot, figdir, pca_kwargs):
        pca = self.pca[pca_name]
        pca.scatter_2d_pca(
            'by_class', t_s, x_ser=self.x_ser, pca_comps_2plot=pca_comps_2plot,
            title=f'Time {t_s[0]}s to {t_s[1]}s', **pca_kwargs['plot_kwargs']
        )

        fig, ax = pca.scatter_plot
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        self._save_and_display(fig, figdir / f'{pca_name}_scatter_{"_".join(list(map(str, pca_comps_2plot)))}.pdf')

    def plot_pca_euclidean_distance(
        self, pca_name: str, times, figdir: Path, metric='cosine', n_pcs: int = None,
        align_trajs: bool = False, align_method: str = "orthogonal", align_pip_grouping=None,
        global_align_window: tuple = None, reference: str = None, ref_id=0,
        reduce_when_pairwise: str = "per_event_mean", labels_map: dict = None,
        title: str = None, legend: bool = True, figsize: tuple = (7, 4), lw: float = 2.0,
        alpha: float = 0.95, save_pdf: bool = True, save_csv: bool = True
    ):
        pca = self.pca[pca_name]
        res = pca.pcspace_distances(
            prop='by_class', event_window=self.window, times=times, n_pcs=n_pcs, x_ser=self.x_ser,
            align_trajs=align_trajs, align_method=align_method, align_pip_grouping=align_pip_grouping,
            global_align_window=global_align_window, reference=reference, return_squareform=True,
            metric=metric, ref_id=ref_id
        )

        fig, ax = pca.plot_pcspace_distances(
            prop='by_class', event_window=self.window, times=times, n_pcs=n_pcs, x_ser=self.x_ser,
            align_trajs=align_trajs, align_method=align_method, align_pip_grouping=align_pip_grouping,
            global_align_window=global_align_window, reference=reference, reduce_when_pairwise=reduce_when_pairwise,
            labels_map=labels_map, figsize=figsize, ax=None, legend=legend, title=title, lw=lw, alpha=alpha
        )

        tparts = []
        for t in (times if isinstance(times, (list, tuple, np.ndarray)) else [times]):
            if isinstance(t, (list, tuple)) and len(t) == 2:
                tparts.append(f"{t[0]:.3f}_{t[1]:.3f}")
            else:
                tparts.append(f"{float(t):.3f}")
        tlabel = "-".join(tparts)

        ref_tag = (reference or "pairwise")
        red_tag = (reduce_when_pairwise if reference is None else "to_ref")
        pcs_tag = f"pc{n_pcs}" if n_pcs is not None else "pcALL"
        alg_tag = f"align_{align_method}" if align_trajs else "noalign"

        if save_pdf:
            fig.savefig(
                figdir / f"{pca_name}_pcspace_dist_{ref_tag}_{red_tag}_{alg_tag}_{pcs_tag}_{tlabel}.pdf",
                dpi=300
            )
        if save_csv:
            events, specs, tmean, D = res["events"], res["specs"], res["times"], res["distances"]
            tidy = []
            if reference is not None:
                others = [e for e in events if e != reference]
                for k, (t0, t1) in enumerate(specs):
                    for j, ev in enumerate(others):
                        tidy.append({
                            "pca_name": pca_name, "reference": reference, "event": ev, "t0": t0, "t1": t1,
                            "t_mean": tmean[k], "distance": float(D[k, j]), "n_pcs": n_pcs,
                            "align_trajs": align_trajs, "align_method": align_method
                        })
            else:
                K, N, _ = D.shape
                for k, (t0, t1) in enumerate(specs):
                    for i in range(N):
                        for j in range(N):
                            if i == j:
                                continue
                            tidy.append({
                                "pca_name": pca_name, "event_i": events[i], "event_j": events[j], "t0": t0,
                                "t1": t1, "t_mean": tmean[k], "distance": float(D[k, i, j]),
                                "n_pcs": n_pcs, "align_trajs": align_trajs, "align_method": align_method
                            })
            pd.DataFrame(tidy).to_csv(
                figdir / f"{pca_name}_pcspace_dist_{ref_tag}_{red_tag}_{alg_tag}_{pcs_tag}_{tlabel}.csv",
                index=False
            )

        return res, (fig, ax)


if __name__ == '__main__':
    main()