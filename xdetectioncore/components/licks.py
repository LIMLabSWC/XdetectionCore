import numpy as np
import pandas as pd
from ..plotting import plot_2d_array_with_subplots
from ..ephys.spike_time_utils import cluster_spike_times, get_spike_times_in_window, gen_spike_matrix

class SessionLicks:
    def __init__(self, lick_times: np.ndarray, sound_writes: pd.DataFrame, fs=1e3, resample_fs=1e2, ):
        self.event_lick_plots = {}
        self.event_licks = {}
        self.fs = fs
        self.new_fs = resample_fs
        self.spike_times, self.clusters = lick_times, np.zeros_like(lick_times)
        self.cluster_spike_times_dict = cluster_spike_times(self.spike_times, self.clusters)
        self.event_spike_matrices = dict()
        self.event_cluster_spike_times = dict()
        self.sound_writes = sound_writes

    def get_event_spikes_slow(self, event_idx: int, event_name: str, window: [list | np.ndarray], sessname: str,
                         sound_df_query='',**kwargs):
        sound_df_query = f'Payload == {event_idx} & {sound_df_query}' if sound_df_query else f'Payload == {event_idx}'
        event_times = self.sound_writes.query(sound_df_query)['Timestamp'].values
        event_trialnum = self.sound_writes.query(sound_df_query)['Trial_Number'].values
        self.event_spike_matrices[event_name] = dict()
        self.event_cluster_spike_times[event_name] = dict()
        for event_time in event_times:
            if f'{event_name}_{event_time}' not in list(self.event_cluster_spike_times.keys()):
                self.event_cluster_spike_times[event_name][f'{event_name}_{event_time}'] = get_spike_times_in_window(
                    event_time, self.cluster_spike_times_dict, window, self.new_fs)
            if f'{event_name}_{event_time}' not in list(self.event_spike_matrices.keys()):
                self.event_spike_matrices[event_name][f'{event_name}_{event_time}'] = gen_spike_matrix(
                    self.event_cluster_spike_times[event_name][f'{event_name}_{event_time}'],
                    window, self.new_fs, kwargs.get('kernel_width',40))
        self.event_licks[f'{event_name}_licks'] = pd.concat(self.event_spike_matrices[event_name])
        self.event_licks[f'{event_name}_licks'].index = pd.MultiIndex.from_arrays([event_times, event_trialnum,
                                                                                   [sessname] * len(event_trialnum)],
                                                                                  names=['time', 'trial', 'sess'])
    @staticmethod
    def trial_average(event_times, trial_times, window, bin_width):
        event_times = np.asarray(event_times)
        trial_times = np.asarray(trial_times)
        w_start, w_stop = window
        n_bins = int(np.ceil((w_stop - w_start) / bin_width))
        all_counts = []

        for onset in trial_times:
            rel_times = event_times - onset
            mask = (rel_times >= w_start) & (rel_times < w_stop)
            rel_times = rel_times[mask]

            if rel_times.size > 0:
                scaled = np.floor((rel_times - w_start) / bin_width).astype(int)
                counts = np.bincount(scaled, minlength=n_bins)
            else:
                counts = np.zeros(n_bins, dtype=int)
            all_counts.append(counts)

        all_counts = np.vstack(all_counts)
        mean_counts = all_counts.mean(axis=0)
        bin_edges = np.arange(n_bins + 1) * bin_width + w_start
        mean_rates = mean_counts / bin_width
        return bin_edges, mean_rates

    def get_event_spikes(self, event_idx: int, event_name: str, window: [list | np.ndarray], sessname: str,
                         sound_df_query='', **kwargs):
        query_str = f'Payload == {event_idx}'
        if sound_df_query:
            query_str += f' & {sound_df_query}'

        queried_sound_writes = self.sound_writes.query(query_str)
        event_times = queried_sound_writes['Timestamp'].values
        event_trialnum = queried_sound_writes['Trial_Number'].values

        if event_name not in self.event_spike_matrices:
            self.event_spike_matrices[event_name] = {}
        if event_name not in self.event_cluster_spike_times:
            self.event_cluster_spike_times[event_name] = {}

        kernel_width = kwargs.get('kernel_width', 40)
        existing_keys = self.event_cluster_spike_times[event_name].keys()
        new_event_times = [et for et in event_times if f'{event_name}_{et}' not in existing_keys]

        for event_time in new_event_times:
            event_key = f'{event_name}_{event_time}'
            self.event_cluster_spike_times[event_name][event_key] = get_spike_times_in_window(
                event_time, self.cluster_spike_times_dict, window, self.new_fs)
            self.event_spike_matrices[event_name][event_key] = gen_spike_matrix(
                self.event_cluster_spike_times[event_name][event_key], window, self.new_fs, kernel_width)

        self.event_licks[f'{event_name}_licks'] = pd.concat(self.event_spike_matrices[event_name].values(), axis=0)
        multi_index = pd.MultiIndex.from_arrays([event_times, event_trialnum, [sessname] * len(event_trialnum)],
                                                names=['time', 'trial', 'sess'])
        self.event_licks[f'{event_name}_licks'].index = multi_index

    def plot_licks(self, event_name, window=(-3, 3)):
        licks_to_event = self.event_licks[f'{event_name}_licks']
        lick_plot = plot_2d_array_with_subplots(licks_to_event, cmap='binary', extent=[window[0], window[1],
                                                                                       licks_to_event.shape[0], 0],
                                                plot_cbar=False)
        lick_plot[1].axvline(0, c='k', ls='--')
        lick_plot[1].set_ylabel('Trials')
        lick_plot[1].set_xlabel(f'time since {event_name} (s)')
        self.event_lick_plots[f'licks_to_{event_name}'] = lick_plot