from functools import partial
import json
import multiprocessing
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.signal.windows import gaussian, exponential
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from tqdm import tqdm

from .generate_synthetic_spikes import gen_responses
from ..io_utils import load_spikes
from typing import Dict, List, Optional, Union
TrialSelector = Optional[Union[slice, List[int]]]


def _worker_process_event(event_time, window, fs, cluster_dict, get_matrix):
    """
    Top-level worker function to process a single event time.
    Returns a tuple of (event_time, spike_times_dict, spike_matrix_df)
    """
    spikes = get_spike_times_in_window(event_time, cluster_dict, window, fs)
    matrix = gen_spike_matrix(spikes, window, fs) if get_matrix else None
    return event_time, spikes, matrix

def fast_instantaneous_rate(spike_times: np.ndarray, t_start: float, t_stop: float, 
                            sampling_period: float, kernel_type='gaussian', kernel_width=0.04):
    """
    Fast replacement for elephant.statistics.instantaneous_rate.
    """
    # Create time bins
    bins = np.arange(t_start, t_stop + sampling_period, sampling_period)
    
    # Bin spikes (count spikes in each bin)
    counts, _ = np.histogram(spike_times, bins=bins)
    
    # Convert counts to firing rate (Hz)
    rate = counts / sampling_period
    
    # Apply smoothing
    sigma_bins = kernel_width / sampling_period
    
    if kernel_type == 'gaussian':
        # gaussian_filter1d is highly optimized for 1D arrays
        smoothed_rate = gaussian_filter1d(rate.astype(float), sigma=sigma_bins, mode='constant', cval=0.0)
    elif kernel_type == 'exponential':
        # Create a symmetric exponential kernel approximation
        window_size = int(sigma_bins * 10) + 1
        kernel = exponential(window_size, center=(window_size-1)//2, tau=sigma_bins, sym=True)
        kernel /= kernel.sum()
        smoothed_rate = convolve(rate, kernel, mode='same')
    else:
        smoothed_rate = rate
        
    return smoothed_rate


def cluster_spike_times(spike_times: np.ndarray, spike_clusters: np.ndarray) -> dict:
    assert spike_clusters.shape == spike_times.shape, Warning('spike times/ cluster arrays need to be same shape')
    cluster_spike_times_dict = {}
    for i in tqdm(np.unique(spike_clusters), desc='getting session cluster spikes times',
                  total=len(np.unique(spike_clusters)), disable=True):
        cluster_spike_times_dict[i] = spike_times[spike_clusters == i]
    return cluster_spike_times_dict


def get_times_in_window(all_times: np.ndarray,window: (list | np.ndarray)) -> np.ndarray:
    return all_times[(all_times >= window[0]) * (all_times <= window[1])]


def get_spike_rate_in_window(spike_times: np.ndarray, window: (list | np.ndarray), fs):
    bin_firing_rate = fast_instantaneous_rate(spike_times, window[0], window[1], 
                                              sampling_period=0.1, kernel_type='gaussian', kernel_width=0.02)
    return bin_firing_rate


def get_spike_times_in_window(event_time: int, spike_time_dict: dict, window: (list | np.ndarray), fs):
    """
    Get spike times in a specified window for a given event.
    """
    window_spikes_dict = {}

    for cluster_id in tqdm(spike_time_dict, desc='getting spike times for event', total=len(spike_time_dict),
                           disable=True):
        all_spikes = (spike_time_dict[cluster_id] - event_time)  # / fs
        window_spikes_dict[cluster_id] = all_spikes[(all_spikes >= window[0]) * (all_spikes <= window[1])]

    return window_spikes_dict


def gen_spike_matrix(spike_time_dict: dict, window, fs, kernel_width=40):
    fs = 100
    precision = np.ceil(np.log10(fs)).astype(int)
    time_cols = np.round(np.arange(window[0], window[1] + 1 / fs, 1 / fs), precision)

    event_psth = []
    for c_spiketimes in spike_time_dict.values():
        rate = fast_instantaneous_rate(c_spiketimes, window[0], window[-1] + 0.011, 
                                       sampling_period=0.01, kernel_type='exponential', 
                                       kernel_width=kernel_width / 1000.0)
        # Ensure array length aligns exactly with time_cols
        event_psth.append(rate[:len(time_cols)])
        
    spike_matrix = pd.DataFrame(event_psth,columns=time_cols,index=list(spike_time_dict.keys()))
    spike_matrix.columns = pd.to_timedelta(spike_matrix.columns, 's')

    return spike_matrix


def gen_firing_rate_matrix(spike_matrix: pd.DataFrame, bin_dur=0.01, baseline_dur=0.0,
                           zscore_flag=False, gaus_std=0.04) -> pd.DataFrame:
    if gaus_std:
        gaus_window = gaussian(int(gaus_std / bin_dur*2), int(gaus_std / bin_dur*2))
        gaus_window = np.array_split(gaus_window,2)[1]
    spike_matrix.columns = pd.to_timedelta(spike_matrix.columns, 's')
    rate_matrix = spike_matrix.T.resample(f'{bin_dur}S').mean().T / bin_dur
    cols = rate_matrix.columns
    if gaus_std:
        rate_matrix = np.array([convolve(row, gaus_window, mode='same') for row in rate_matrix.values])
    assert not all([baseline_dur, zscore_flag])
    rate_matrix = pd.DataFrame(rate_matrix, columns=cols)
    if baseline_dur:
        rate_matrix = pd.DataFrame(rate_matrix, columns=cols)
        rate_matrix = rate_matrix.sub(np.mean(rate_matrix.loc[:, timedelta(0, -baseline_dur):timedelta(0, 0)],axis=1),
                                      axis=0)
    if zscore_flag:
        rate_matrix = zscore(rate_matrix, axis=1, )
    rate_matrix = rate_matrix.fillna(0)
    return rate_matrix


def zscore_by_trial(resp_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    zscore_by_trial_resps: Dict[str, Dict[str, np.ndarray]] = {}
    for sess, sess_resps in resp_dict.items():
        zscored = {}
        for pip, pip_resps in sess_resps.items():
            mean = pip_resps.mean(axis=-1, keepdims=True)
            std = pip_resps.std(axis=-1, keepdims=True)
            zscored[pip] = (pip_resps - mean) / (std+1e-14)

        is_nan_by_pip = [np.any(np.isnan(pip_resps.mean(axis=0)),axis=-1) for pip_resps in zscored.values()]
        mask = ~np.any(is_nan_by_pip, axis=0)

        zscore_by_trial_resps[sess] = {pip: arr[:,mask] for pip, arr in zscored.items()}
    return zscore_by_trial_resps


def concat_and_clean_responses(resp_dict: Dict[str, Dict[str, np.ndarray]],
                               trial_slices: Optional[List[TrialSelector]] = None) -> Dict[str, np.ndarray]:
    pip_keys = list(resp_dict.values())[0].keys()

    if trial_slices is None:
        trial_slices = [None] * len(pip_keys)

    concat_resps: Dict[str, np.ndarray] = {}
    for e, trial_slice in zip(pip_keys, trial_slices):
        all_sessions: list[np.ndarray] = []
        for sessname in resp_dict:
            data = resp_dict[sessname][e]
            if trial_slice is None:
                vals = np.nanmean(data, axis=0)
            elif isinstance(trial_slice, slice):
                vals = np.nanmean(data[trial_slice], axis=0)
            elif isinstance(trial_slice, list):
                vals = np.nanmean(data[trial_slice], axis=0)
            else:
                raise TypeError(f"Unsupported trial_slice type: {type(trial_slice)}")
            all_sessions.append(vals)
        concat_resps[e] = np.concatenate(all_sessions)

    resps_stacked: np.ndarray = np.concatenate(list(concat_resps.values()), axis=1)
    resps_stacked_no_nans: np.ndarray = resps_stacked[~np.any(np.isnan(resps_stacked), axis=1)]
    resps_unstacked: list[np.ndarray] = np.split(resps_stacked_no_nans, len(concat_resps), axis=1)

    for pip, no_nan_resps in zip(concat_resps, resps_unstacked):
        concat_resps[pip] = no_nan_resps

    return concat_resps


class SessionSpikes:
    def __init__(self, spike_times_path: (Path | str), spike_clusters_path: (Path | str), sess_start_time: float,
                 parent_dir=Path(''), fs=3e4, resample_fs=1e3, beh_write_data_path=None, **kwargs):
        self.spike_times_path = spike_times_path
        self.spike_clusters_path = spike_clusters_path
        self.start_time = sess_start_time
        self.fs = fs
        self.new_fs = resample_fs
        self.spike_times, self.clusters = load_spikes(spike_times_path, spike_clusters_path, parent_dir)
        self.spike_times = self.spike_times / fs

        # Sync
        if beh_write_data_path is not None:
            try:
                print(f'Start: {self.spike_times[0]}, End: {self.spike_times[-1]} pre sync')
                ttl_sink, ttl_source = self.get_sync_events(kwargs.get('rec_dir'),beh_write_data_path)
                self.sync_spike_times(np.pad(ttl_sink, (1, 0)), np.insert(ttl_source,0,sess_start_time))
                print(f'Start: {self.spike_times[0]-sess_start_time}, End: {self.spike_times[-1]-sess_start_time} post sync')
                ttl_sync =True
            except (AssertionError, IndexError, ValueError,FileNotFoundError,NotImplementedError) as e:
                print(e)
                print(f'WARNING:Could not sync spike times for with {Path(beh_write_data_path).name}\n'
                      f'Assuming linear drift')
                self.spike_times = self.sync_by_start_end(kwargs.get('rec_dir'),beh_write_data_path)
                ttl_sync=False

        else:
            self.spike_times = self.spike_times + sess_start_time 
            
        self.duration = self.spike_times[-1] - self.start_time

        self.cluster_spike_times_dict = cluster_spike_times(self.spike_times, self.clusters)
        self.bad_units = set()
        
        self.good_units = None
        if (parent_dir / 'good_units.csv').is_file():
            self.good_units = pd.read_csv(parent_dir / 'good_units.csv').iloc[:, 0].to_list()
        elif (parent_dir.parent / 'good_units.csv').is_file():
            self.good_units = pd.read_csv(parent_dir.parent / 'good_units.csv').iloc[:, 0].to_list()
        else:
            self.good_units = None
            
        if self.good_units is not None and kwargs.get('subset_good_units', False):
            unit_ids = list(self.cluster_spike_times_dict.keys())
            for unit in unit_ids:
                if unit not in self.good_units:
                    self.cluster_spike_times_dict.pop(unit)
            print(f'good units: {self.good_units}')
        else:
            self.curate_units_by_rate()
            
        self.unit_means = self.get_unit_mean_std()
        self.units = list(self.cluster_spike_times_dict.keys())
        
        # self.event_spike_matrices = multiprocessing.Manager().dict()
        # self.event_cluster_spike_times = multiprocessing.Manager().dict()

        self.event_spike_matrices = {}
        self.event_cluster_spike_times = {}

    def get_sync_events(self,recording_dir, beh_write_data_path:Path):

        # get set_dict json
        set_dict = None
        set_dict_path_name = beh_write_data_path.stem.replace('write_data', 'set_times')
        set_dict_path = beh_write_data_path.with_name(set_dict_path_name).with_suffix('.json')
        if set_dict_path.is_file():
            with open(set_dict_path) as f:
                set_dict = json.load(f)
            assert any(sync_pin in list(set_dict.keys()) for sync_pin in ['DO2', 'PWM'])

        # Read write data
        write_data_df = pd.read_csv(beh_write_data_path)
        assert any(sync_pin in write_data_df.columns for sync_pin in ['DO2', 'PWM'])

        # use D02 events else use PWM
        if set_dict is not None:
            sync_pin = 'DO2' if 'DO2' in list(set_dict.keys()) else 'PWM'
            sync_arr = np.array(set_dict[sync_pin]['Timestamp'])
        else:
            sync_pin = 'DO2' if 'DO2' in write_data_df.columns else 'PWM'
            # get rising events
            if sync_pin == 'PWM':
                sync_arr = write_data_df[write_data_df[sync_pin].diff()>0]['Times'].values
            else:
                raise NotImplementedError(f'Sess: {beh_write_data_path.stem}, '
                                          f'syncing to DO2 not implemented with write data df')
        if sync_pin == 'PWM':
            assert len(sync_arr) == 1
            dt = 1
            sync_arr = np.arange(sync_arr[0], write_data_df['Times'].max() + 1, 1)

        print(f'{recording_dir = }')
        sync_event_dir = next(recording_dir.rglob('TTL'))
        sync_message_file = next(recording_dir.rglob('sync_messages.txt'))
        
        # read sync messages
        with open(sync_message_file, 'r') as f:
            sync_messages = f.readlines()
        sample_start_t = int(sync_messages[-1].split(' ')[-1])

        assert sync_event_dir is not None
        ttl_events = ['timestamps','states', 'sample_numbers']
        ttl_df = pd.DataFrame([np.load(sync_event_dir/f'{e}.npy') for e in ttl_events]).T
        ttl_df.columns = ttl_events

        n_samples_offset = ttl_df['sample_numbers'].values[0] - sample_start_t
        t_offset = n_samples_offset/self.fs
        ttl_df['timestamps'] = (ttl_df['sample_numbers'] - ttl_df['sample_numbers'].values[0]) / self.fs
        ttl_df['timestamps'] = ttl_df['timestamps'] + t_offset

        ttl_sink_times = ttl_df[ttl_df['states']==1]['timestamps'].values

        if sync_pin == 'PWM':
            assert np.all(np.round(np.diff(ttl_sink_times))==dt)
        assert (len(sync_arr) * 0.9) <= len(ttl_sink_times) <= len(sync_arr)

        return ttl_sink_times, sync_arr[:len(ttl_sink_times)],

    def sync_by_start_end(self, recording_dir,beh_write_data_path):
        # Read write data
        write_data_df = pd.read_csv(beh_write_data_path)
        assert any(sync_pin in write_data_df.columns for sync_pin in ['DO3'])

        sync_pin = 'DO3'
        # get rising events
        start_ttl = write_data_df[write_data_df[sync_pin]==True]['Times'].values[0]
        end_ttl = write_data_df[write_data_df[sync_pin]==False]['Times'].values[0]

        # get sample times
        sync_dir = next(recording_dir.rglob('Acquisition_Board-100.Rhythm Data'))
        sample_ts = np.load(sync_dir/'timestamps.npy')

        print(f'harp dur = {end_ttl- start_ttl}, oeps dur = {sample_ts[-1] - sample_ts[0]}')
        harp_dur = (end_ttl - start_ttl)
        oeps_dur = (sample_ts[-1] - sample_ts[0])
        corrected_ts = (self.spike_times)*(1 / (oeps_dur / harp_dur))

        return corrected_ts + start_ttl

    def sync_spike_times(self,sync_sink_times,sync_source_times):
        from io_utils import sync_using_latest_ttl
        spike_times = self.spike_times
        spike_times = sync_using_latest_ttl(spike_times, sync_sink_times, sync_source_times)
        self.spike_times = spike_times

    def get_event_spikes(self, event_times: (list | np.ndarray | pd.Series), event_name: str,
                         window: (list | np.ndarray), get_spike_matrix=True):
        
        # 1. Identify only the events we haven't processed yet
        missing_events = [t for t in event_times if f'{event_name}_{t}' not in self.event_cluster_spike_times]
        
        if not missing_events:
            return  # Everything is already cached!

        # 2. Setup partial function for the pool
        worker_func = partial(_worker_process_event, 
                              window=window, 
                              fs=self.new_fs, 
                              cluster_dict=self.cluster_spike_times_dict, 
                              get_matrix=get_spike_matrix)

        # 3. Determine execution strategy
        # If there are very few events, the numpy logic is so fast that 
        # spawning new processes will actually slow things down.
        num_cores = min(multiprocessing.cpu_count(), len(missing_events))
        
        if num_cores <= 1 or len(missing_events) < 5:
            # Run sequentially
            results = [worker_func(t) for t in missing_events]
        else:
            # Run in parallel
            with multiprocessing.Pool(processes=num_cores) as pool:
                # chunksize=1 ensures memory isn't hogged by a single worker
                results = pool.map(worker_func, missing_events, chunksize=1)

        # 4. Update standard dictionaries safely in the main process
        for event_time, spikes, matrix in results:
            key = f'{event_name}_{event_time}'
            self.event_cluster_spike_times[key] = spikes
            if get_spike_matrix:
                self.event_spike_matrices[key] = matrix

    def curate_units(self):
        for unit in self.cluster_spike_times_dict:
            d_spike_times = np.diff(self.cluster_spike_times_dict[unit])
            if np.mean(d_spike_times > 10) > 0.05:
                self.bad_units.add(unit)
        print(f'popped units {self.bad_units}, remaining units: {len(self.cluster_spike_times_dict) - len(self.bad_units)}/{len(self.cluster_spike_times_dict)}')
        for unit in self.bad_units:
            self.cluster_spike_times_dict.pop(unit)

    def curate_units_by_rate(self):
        for unit in self.cluster_spike_times_dict:
            d_spike_times = np.diff(self.cluster_spike_times_dict[unit])
            if np.mean(d_spike_times) > (1 / 0.05):
                self.bad_units.add(unit)
        print(f'popped units {self.bad_units}, remaining units: {len(self.cluster_spike_times_dict) - len(self.bad_units)}/{len(self.cluster_spike_times_dict)}')
        for unit in self.bad_units:
            self.cluster_spike_times_dict.pop(unit)

    def get_unit_mean_std(self,time_bins_idx=None):
        unit_means = np.full(len(self.cluster_spike_times_dict), np.nan)
        unit_stds = np.full(len(self.cluster_spike_times_dict), np.nan)

        for i, unit in sorted(list(enumerate(self.cluster_spike_times_dict))):
            unit_rates = fast_instantaneous_rate(self.cluster_spike_times_dict[unit], 
                                                 self.start_time, self.start_time + self.duration, 
                                                 sampling_period=0.1, kernel_type='gaussian', kernel_width=0.04)
            unit_means[i] = np.mean(unit_rates)
            
            if time_bins_idx is not None:
                assert len(time_bins_idx) == len(unit_rates)
                unit_rates = unit_rates[time_bins_idx]
                unit_means[i] = np.mean(unit_rates)
            unit_stds[i] = np.std(unit_rates) if np.std(unit_rates) > 0.01 else 1

        assert np.isnan(unit_means).sum() == 0 + np.isnan(unit_stds).sum() == 0, 'Nans in mean and std'
        return unit_means, unit_stds


def get_event_psth(sess_spike_obj: SessionSpikes, event_idx, event_times: (pd.Series| np.ndarray| list),
                   window: list[float, float], event_lbl: str, baseline_dur=0.25, zscore_flag=False, iti_zscore=None,
                   gaus_std=0.04, synth_data=None, synth_params=None) -> tuple[np.ndarray, pd.DataFrame]:
    if iti_zscore:
        zscore_flag = False

    assert not (zscore_flag and iti_zscore), 'zscore_flag and iti_zscore cannot both be True'

    sess_spike_obj.get_event_spikes(event_times, f'{event_idx}', window)
    event_keys = [f'{event_idx}_{t}' for t in event_times]
    if synth_data is None:
        all_event_list = [sess_spike_obj.event_spike_matrices[key]for key in event_keys]
    else:
        if synth_params is None:
            unit_rates = np.random.rand(len(sess_spike_obj.units))*40
            unit_time_offsets = np.random.rand(len(sess_spike_obj.units))*0.1
        else:
            unit_rates = synth_params['unit_rates']
            unit_time_offsets = synth_params['unit_time_offsets'] if synth_params else None
        synth_times = gen_responses(unit_rates, len(event_times), np.arange(window[0],window[1],0.002),
                                    unit_time_offsets=unit_time_offsets)
        
        # Replaced synth data spike trains with fast instantaneous rate
        all_event_list = []
        for trial in synth_times:
            trial_rates = [fast_instantaneous_rate(ee, window[0], window[1] + 0.11, 
                                                   sampling_period=0.1, kernel_type='gaussian', 
                                                   kernel_width=0.04) for ee in trial]
            all_event_list.append(np.array(trial_rates))

    ratemat_arr3d = np.array(all_event_list) if isinstance(all_event_list, list) else all_event_list
    x_ser = np.linspace(window[0], window[1], ratemat_arr3d.shape[-1])
    rate_mat = pd.DataFrame(ratemat_arr3d.mean(axis=0),
                            columns=pd.to_timedelta(x_ser,'s'))
    rate_mat.index = sess_spike_obj.units

    return ratemat_arr3d, rate_mat, rate_mat.index