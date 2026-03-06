import pickle
from copy import deepcopy as copy
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from .ephys.generate_synthetic_spikes import gen_patterned_unit_rates, gen_patterned_time_offsets
from .io_utils import load_sound_bin, format_sound_writes, read_lick_times
from .ephys.spike_time_utils import SessionSpikes, get_times_in_window

# Component imports
from .components.events import SoundEvent
from .components.licks import SessionLicks
from .components.pupil import SessionPupil
from .components.utils import zscore_by_unit


class Session:
    def __init__(self, sessname, ceph_dir, pkl_dir, ):
        self.pupil_obj = None
        self.lick_obj = None
        self.iti_zscore = None
        self.pip_desc =  None
        self.td_df = pd.DataFrame()
        self.sessname = sessname
        self.spike_obj = None
        self.sound_event_dict = {}
        self.beh_event_dict = {}
        self.decoders = {}
        self.ceph_dir = ceph_dir
        self.sound_writes_df = pd.DataFrame

    def init_spike_obj(self, spike_times_path, spike_cluster_path, start_time, parent_dir,**kwargs ):
        self.spike_obj = SessionSpikes(spike_times_path, spike_cluster_path, start_time, parent_dir,**kwargs)

    def init_sound_event_dict(self, sound_write_path, **format_kwargs):
        patterns = format_kwargs.get('patterns', None)
        normal_patterns = format_kwargs.get('normal_patterns', None)
        if normal_patterns is None:
            raise ValueError('No normal patterns provided')
        if patterns is None:
            raise ValueError('No patterns provided')

        patterns_df = pd.DataFrame(patterns, columns=list('ABCD'))
        ABCD_patts = [pattern for pattern in patterns if np.all(np.diff(pattern)>0)]
        non_ABCD_patts = [pattern for pattern in patterns if not np.all(np.diff(pattern)>0)]
        patt_rules = [0 if patt in ABCD_patts else 1 for patt in patterns]
        patt_group = [np.where(patterns_df['A']==patt[0])[0][0] for patt in patterns]
        ptypes = {'_'.join([str(e) for e in patt]): int(f'{group}{rule}')
                  for patt, group, rule in zip(patterns, patt_group, patt_rules)}

        if not sound_write_path:
            assert self.sound_writes_df is not None
            sound_writes = self.sound_writes_df
        else:
            sound_writes = load_sound_bin(sound_write_path)
            sound_writes = format_sound_writes(sound_writes, ptypes=ptypes, **format_kwargs if format_kwargs else {})
            self.sound_writes_df = sound_writes

        base_pip_idx = [e for e in sound_writes['Payload'].unique() if e not in sum(patterns, []) and e >= 8]
        if len(base_pip_idx) > 1:
            if 3 not in sound_writes['Payload'].values:
                base_pip_idx = min(base_pip_idx)
            else:
                base_pip_idx = sound_writes.query('Payload in @base_pip_idx')['Payload'].mode().iloc[0]
        else:
            base_pip_idx = base_pip_idx[0]
        non_base_pip_idx = sorted([e for e in sound_writes['Payload'].unique() if e not in [base_pip_idx,3]])

        event_sound_times = {}
        event_sound_idx = {}
        if 3 in sound_writes['Payload'].values:
            for idx, p in patterns_df.iterrows():
                for pi, pip in enumerate(p):
                    ptype= ptypes['_'.join([str(e) for e in p])]
                    timesss = sound_writes.query(f'Payload == {pip} and pip_counter == {pi + 1} '
                                                 f'and ptype == @ptype')['Timestamp']
                    if not timesss.empty:
                        event_sound_times[f'{p.index[pi]}-{idx}'] = timesss
                        event_sound_idx[f'{p.index[pi]}-{idx}'] = pip
        else:
            for pip_i, pip in enumerate(non_base_pip_idx):
                lbl = chr(ord('A') + pip_i)
                event_sound_times[f'{lbl}-0'] = \
                    sound_writes.query(f'Payload == {pip}')['Timestamp']
                event_sound_idx[f'{lbl}-0'] = pip

        long_dt = np.squeeze(np.argwhere(sound_writes['Time_diff'].values > 1))
        long_dt_df = sound_writes.iloc[long_dt]
        trial_start_times = long_dt_df['Timestamp']

        sound_events = [3, base_pip_idx, -1]
        sound_event_labels = ['X', 'base', 'trial_start']

        sound_writes['idx_diff'] = sound_writes['Payload'].diff()
        writes_to_use = np.all([sound_writes['Payload'] == base_pip_idx, sound_writes['idx_diff'] == 0,
                                sound_writes['Trial_Number'] > 10, sound_writes['Time_diff'] < 0.3], axis=0)
        base_pips_times = base_pip_times_same_prev = sound_writes[sorted(writes_to_use)]['Timestamp']
        if base_pips_times.empty:
            base_pips_times = sound_writes[sound_writes['Payload'] == base_pip_idx]['Timestamp']

        if sound_writes[sound_writes['Payload'] == 3].empty:
            sound_events.remove(3), sound_event_labels.remove('X')
        for i, lbl in zip(sound_events, sound_event_labels):
            if i in [base_pip_idx, -1]:
                continue
            event_sound_times[lbl] = sound_writes.query('Payload == @i').drop_duplicates(subset='Trial_Number')[
                'Timestamp']
            event_sound_idx[lbl] = i

        if not base_pips_times.empty:
            idxss = np.random.choice(base_pips_times.index, min(base_pips_times.shape[0],
                                                                np.max([len(e) for e in event_sound_times.values()])),
                                     replace=False)
            event_sound_times['base'] = sound_writes.loc[idxss, 'Timestamp']
            event_sound_idx['base'] = base_pip_idx
        else:
            sound_event_labels.remove('base'), sound_events.remove(base_pip_idx)
        event_sound_times['trial_start'] = trial_start_times
        event_sound_idx['trial_start'] = -1

        assert len(event_sound_times) == len(event_sound_idx), Warning('sound event and labels must be equal length')
        for event_lbl in event_sound_times:
            self.sound_event_dict[event_lbl] = SoundEvent(event_sound_idx[event_lbl], event_sound_times[event_lbl],
                                                          event_lbl)

            self.sound_event_dict[event_lbl].trial_nums = np.squeeze(
                sound_writes.loc[event_sound_times[event_lbl].index, ['Trial_Number']].values)

    def get_event_free_zscore(self,event_free_time_window=2):
        spike_dict = self.spike_obj.cluster_spike_times_dict

        event_times = self.sound_writes_df['Timestamp'].values
        zscored_comp_plot = plt.subplots(figsize=(20, 10))
        mean_unit_mean = []
        mean_unit_std = []

        time_bin = 0.1
        event_free_time_bins = [
            get_times_in_window(event_times, [t - event_free_time_window, t + event_free_time_window]).size == 0
            for t in np.arange(self.spike_obj.start_time, self.spike_obj.start_time
                                                   + self.spike_obj.duration-time_bin, time_bin)]
        event_free_time_bins = np.array(event_free_time_bins)
        print('old means')
        print(self.spike_obj.unit_means)
        self.spike_obj.unit_means = self.spike_obj.get_unit_mean_std(event_free_time_bins)
        print('new means')
        print(self.spike_obj.unit_means)

    def get_grouped_rates_by_property(self, pip_desc, pip_prop, group_noise):
        groups = np.unique([e[pip_prop] for e in pip_desc.values()])
        print(groups)
        assert group_noise
        for group in groups:
            group_pips = [k for k, v in pip_desc.items() if v[pip_prop] == group]
            unit_rates = gen_patterned_unit_rates(len(self.spike_obj.cluster_spike_times_dict), len(group_pips), group_noise)
            unit_times_offsets = gen_patterned_time_offsets(len(self.spike_obj.cluster_spike_times_dict), len(group_pips),0.1)
            for k, v,t in zip(group_pips, unit_rates, unit_times_offsets):
                self.sound_event_dict[k].synth_params = {'unit_rates': v, 'unit_time_offsets': t}

    def get_sound_psth(self, psth_window, baseline_dur=0.25, zscore_flag=False, redo_psth=False, to_exclude=(None,),
                       use_iti_zscore=True, **psth_kwargs):
        if zscore_flag:
            baseline_dur = 0
        self.get_iti_baseline()

        for e in tqdm(self.sound_event_dict, desc='get sound psth', total=len(self.sound_event_dict)):
            if e not in to_exclude:
                self.sound_event_dict[e].get_psth(self.spike_obj, psth_window, title=f'session {self.sessname}',
                                                  baseline_dur=baseline_dur, zscore_flag=zscore_flag,
                                                  redo_psth=redo_psth,
                                                  iti_zscore=(self.iti_zscore if use_iti_zscore else None),
                                                  **psth_kwargs)

    def reorder_psth(self, ref_name, reorder_list):
        if ref_name in list(self.sound_event_dict.keys()):
            for e in reorder_list:
                reord = self.sound_event_dict[e].psth[1].loc[self.sound_event_dict[ref_name].psth[2]]
                self.sound_event_dict[e].psth = (
                    self.sound_event_dict[e].psth[0], reord, self.sound_event_dict[e].psth[2])

    def get_iti_baseline(self):
        assert 'trial_start' in list(self.sound_event_dict.keys())
        trial_starts = self.sound_event_dict['trial_start'].times.values
        iti_means = []
        iti_stds = []
        all_spikes = self.spike_obj.cluster_spike_times_dict

        for ui, unit in tqdm(enumerate(all_spikes), total=len(all_spikes), desc='get_iti_baseline'):
            unit_spikes = [(all_spikes[unit][np.logical_and(all_spikes[unit] >= t-2,all_spikes[unit] <=t)].shape[0])
                           for t in trial_starts]
            unit_mean_rate = np.array([np.nanmean(e)/2 for e in unit_spikes])
            iti_means.append(unit_mean_rate)
            iti_stds.append(unit_mean_rate.std())
        iti_means = np.vstack(iti_means).T  
        iti_stds = np.vstack(iti_stds).T

        self.iti_zscore = iti_means.mean(axis=0), np.squeeze(iti_stds)

    def save_psth(self, figdir: Path, keys='all', **kwargs):
        if keys == 'all':
            keys2save = list(self.sound_event_dict.keys())
        else:
            keys2save = keys
        for key in keys2save:
            self.sound_event_dict[key].save_plot_as_svg(suffix=self.sessname, figdir=figdir)

    def pickle_obj(self, pkldir:Path):
        if not pkldir.is_dir():
            pkldir.mkdir()
        with open(pkldir / f'{self.sessname}.pkl', 'wb') as pklfile:
            to_save = copy(self)
            to_save.spike_obj.event_spike_matrices = None
            to_save.spike_obj.event_cluster_spike_times = None
            pickle.dump(to_save, pklfile)

    def init_decoder(self, decoder_name: str, predictors, features, model_name='logistic'):
        raise NotImplementedError('Decoder as part of Session class deprecated, use standalone Decoder class instead')
        # self.decoders[decoder_name] = Decoder(predictors, features, model_name)

    def run_decoder(self, decoder_name, labels, plot_flag=False, decode_shuffle_flag=False, dec_kwargs=None, **kwargs):
        raise NotImplementedError('Decoder as part of Session class deprecated, use standalone Decoder class instead')
        # self.decoders[decoder_name].decode(dec_kwargs=dec_kwargs, **kwargs)
        # if plot_flag:
        #     from .plotting import plot_decoder_accuracy
        #     plot_decoder_accuracy(self.decoders[decoder_name].accuracy, labels=labels, )

    def map_preds2sound_ts(self, sound_event, decoders, psth_window, window, map_kwargs={}):
        raise NotImplementedError('Decoder as part of Session class deprecated, use standalone Decoder class instead')
        # sound_event_ts = get_predictor_from_psth(self, sound_event, psth_window, window, mean=None, mean_axis=0)
        # prediction_ts_list = [np.array([self.decoders[decoder].map_decoding_ts(trial_ts, **map_kwargs)
        #                                 for decoder in decoders]) for trial_ts in sound_event_ts]
        # prediction_ts = np.array(prediction_ts_list)
        # return prediction_ts

    def load_trial_data(self, tdfile_path):
        self.td_df = pd.read_csv(tdfile_path)
        self.get_n_since_last()
        self.get_local_rate()
        sessname = self.sessname
        name, date = sessname.split('_')
        if not date.isnumeric():
            date = date[:-1]

        if 'Session_Block' not in self.td_df.columns:
            if 'WarmUp' not in self.td_df.columns:
                self.td_df['WarmUp'] = np.full_like(self.td_df.index, False)
            first_dev_trial = self.td_df.query('Pattern_Type != 0').index[0] if len(self.td_df.query('Pattern_Type != 0'))>0 else self.td_df.shape[0]
            if 'Stage' not in self.td_df.columns:
                if len(self.td_df['Pattern_Type'].unique())>1:
                    self.td_df['Stage'] = 3
                else:
                    self.td_df['Stage'] = 4
            sess_block = [-1 if r['WarmUp'] else 0 if r['Stage'] <= 3 else 2 if r['Stage']==4 and idx<first_dev_trial
                          else 3 if r['Stage']==4 and idx>=first_dev_trial else 0 for idx, r in self.td_df.iterrows()]
            self.td_df['Session_Block'] = sess_block

        self.td_df.index = pd.MultiIndex.from_arrays(
            [[sessname] * len(self.td_df), [name] * len(self.td_df), [date] * len(self.td_df),
             self.td_df.reset_index().index+1],
            names=['sess', 'name', 'date', 'trial_num'])

    def get_n_since_last(self):
        self.td_df['n_since_last'] = np.arange(self.td_df.shape[0])
        since_last = self.td_df[self.td_df['Tone_Position'] == 0].index
        if not since_last.empty:
            for t, tt in zip(since_last, np.pad(since_last, [1, 0])):
                self.td_df.loc[tt + 1:t, 'n_since_last'] = self.td_df.loc[tt + 1:t, 'n_since_last'] - tt
            self.td_df.loc[t + 1:, 'n_since_last'] = self.td_df.loc[t + 1:, 'n_since_last'] - t

    def get_local_rate(self, window=10):
        self.td_df['local_rate'] = self.td_df['Tone_Position'].rolling(window=window).mean()

    def init_lick_obj(self, lick_times_path, sound_events_path, normal):
        licks = read_lick_times(lick_times_path)
        sound_events = pd.read_csv(sound_events_path)
        sound_events = format_sound_writes(sound_events, normal)
        self.lick_obj = SessionLicks(licks, sound_events)

    def get_licks_to_event(self, event_idx, event_name, window=(-3, 3), align_kwargs=None,plot=False):
        self.lick_obj.get_event_spikes(event_idx, event_name, window, self.sessname,
                                       **align_kwargs if align_kwargs else {})
        if plot:
            self.lick_obj.plot_licks(event_name, window)

    def init_pupil_obj(self, pupil_data, sound_events_path, beh_events_path, normal):
        if sound_events_path is None:
            sound_events = None
        else:
            sound_events = pd.read_csv(sound_events_path)
            sound_events = format_sound_writes(sound_events, normal)

        if beh_events_path is None:
            beh_events = None
        else:
            beh_events = pd.read_csv(beh_events_path)
        self.pupil_obj = SessionPupil(pupil_data, sound_events, beh_events)

    def get_pupil_to_event(self, event_idx, event_name, window=(-3, 3), align_kwargs=None,alignmethod='w_soundcard',
                           plot_kwargs=None):
        if alignmethod == 'w_soundcard':
            self.pupil_obj.align2events(event_idx, event_name, window, self.sessname,
                                        **align_kwargs if align_kwargs else {})
        elif alignmethod == 'w_td_df':
            if align_kwargs and 'sound_df_query' in align_kwargs:
                align_kwargs.pop('sound_df_query')
            if event_name == 'X':
                col2use = 'Gap_Time_dt'
                td_query = 'Trial_Outcome in [0,1]'
            elif event_name == 'A':
                col2use = 'ToneTime_dt'
                td_query = 'Tone_Position == 0 & N_TonesPlayed > 0'
            else:
                return
            try:
                self.pupil_obj.align2events_w_td_df(self.td_df,col2use, td_query,
                                                    window, self.sessname,event_idx, event_name,
                                                    **align_kwargs if align_kwargs else {})
            except KeyError:
                print(f'Could not align to event {event_name} in session {self.sessname}')
        else:
            raise NotImplementedError

def get_predictor_from_psth(sess_obj: Session, event_key, psth_window, new_window, mean=np.mean, mean_axis=2,
                            use_iti_zscore=False, use_unit_zscore=True, baseline=0) -> np.ndarray:
    assert not (use_iti_zscore and use_unit_zscore), 'Can only use one zscore option at a time'
    event_arr = sess_obj.sound_event_dict[event_key].psth[0]
    event_arr_tseries = np.linspace(psth_window[0], psth_window[1], event_arr.shape[-1])
    time_window_idx = np.logical_and(event_arr_tseries >= new_window[0], event_arr_tseries <= new_window[1])

    if baseline:
        baseline_window_idx = np.logical_and(event_arr_tseries >= -baseline, event_arr_tseries <=0)
        event_arr = event_arr - np.mean(event_arr[:, :, baseline_window_idx], axis=2, keepdims=True)
    predictor = event_arr[:, :, time_window_idx]
    if use_iti_zscore:
        predictor = zscore_by_unit(predictor, sess_obj.iti_zscore[0], sess_obj.iti_zscore[1])
    if use_unit_zscore:
        predictor = zscore_by_unit(predictor, sess_obj.spike_obj.unit_means[0], sess_obj.spike_obj.unit_means[1])
    if mean:
        predictor = mean(predictor, axis=mean_axis)

    return predictor