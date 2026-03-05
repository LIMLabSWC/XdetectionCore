import numpy as np
import pandas as pd
import warnings
from datetime import timezone

class SessionPupil:
    def __init__(self, pupil_data, sound_events, beh_events):
        self.aligned_pupil = {}
        self.pupil_data = pupil_data
        self.sound_writes = sound_events
        self.beh_events = beh_events

    def align2events(self, event_idx: int, event_name: str, window, sessname: str,
                     sound_df_query='', baseline_dur=0.0, size_col='dlc_radii_a_zscored',event_shifts=None):

        pupil_size = self.pupil_data[size_col]
        pupil_isout = self.pupil_data['isout']
        start, end = pupil_size.index[0], pupil_size.index[-1]
        dt = np.round(np.nanmedian(np.diff(pupil_size.index)),2)
        event_tdeltas = np.round(np.arange(window[0], window[1], dt), 2)

        sound_df_query = f'Payload == {event_idx} & {sound_df_query}' if sound_df_query else f'Payload == {event_idx}'
        event_times = self.sound_writes.query(sound_df_query)['Timestamp'].values
        event_trialnums = self.sound_writes.query(sound_df_query)['Trial_Number'].values
        if event_shifts is not None:
            event_times += event_shifts

        aligned_epochs = []
        for eventtime in event_times:
            try:
                a = pupil_size.loc[eventtime + window[0]: eventtime + window[1]]
            except KeyError:
                print(f'{sessname}: {start<eventtime<end= } {eventtime}  {event_name} not found. '
                      f'Event {np.where(event_times==eventtime)[0][0]+1}/{len(event_times)}')
                continue
            if a.isna().all():
                print(f'{sessname}: {start<eventtime<end= } {eventtime} {event_name} not found. '
                      f'Event {np.where(event_times == eventtime)[0][0] + 1}/{len(event_times)}')
                aligned_epochs.append(pd.Series(np.full_like(event_tdeltas, np.nan)))
            else:
                if pupil_isout.loc[eventtime + window[0]: eventtime + window[1]].mean() > 1.5:
                    warnings.warn(f'{sessname}: {eventtime} {event_name} has many outliers. Not using')
                    aligned_epochs.append(pd.Series(np.full_like(event_tdeltas, np.nan)))
                else:
                    aligned_epochs.append(pupil_size.loc[eventtime + window[0]: eventtime + window[1]])
        
        for aligned_epoch, eventtime in zip(aligned_epochs, event_times):
            aligned_epoch.index = np.round(aligned_epoch.index - eventtime, 2)
        sessname_list = [sessname] * len(event_times)
        aligned_epochs_df = pd.DataFrame(aligned_epochs, columns=event_tdeltas,
                                         index=pd.MultiIndex.from_tuples(list(zip(event_times, event_trialnums,
                                                                                  sessname_list)),
                                                                         names=['time', 'trial', 'sess']))
        aligned_epochs_df = aligned_epochs_df.dropna(axis=0, how='all')
        aligned_epochs_df = aligned_epochs_df.interpolate(limit_direction='both',axis=1 )
        if baseline_dur:
            epoch_baselines = aligned_epochs_df.loc[:, -baseline_dur:0.0]
            aligned_epochs_df = aligned_epochs_df.sub(epoch_baselines.mean(axis=1), axis=0)

        self.aligned_pupil[event_name] = aligned_epochs_df

    def align2events_w_td_df(self,td_df: pd.DataFrame, td_df_column: str, td_df_query: str, window, sessname: str,
                     event_idx: int, event_name: str,
                     baseline_dur=0.0, size_col='dlc_radii_a_zscored'):
        raise NotImplementedError('This function is no longer implemented. Use align2events instead.')
        pupil_size = self.pupil_data[size_col]
        start, end = pupil_size.index[0], pupil_size.index[-1]
        dt = np.round(np.nanmedian(np.diff(pupil_size.index)), 2)
        event_tdeltas = np.round(np.arange(window[0], window[1], dt), 2)

        event_trials = td_df.query(td_df_query)
        if 'Bonsai_Time_dt' in event_trials.columns:
            event_trials_tdeltas = event_trials.eval(f'{td_df_column} - Bonsai_Time_dt',inplace=False)
            event_times = event_trials['Harp_Time'] + event_trials_tdeltas.dt.total_seconds()
        else:
            eventtimes_w_tz = [dt.to_pydatetime().astimezone() for dt in event_trials[td_df_column]]
            eventtimes_as_ts = [dt.replace(tzinfo=timezone.utc).timestamp() for dt in eventtimes_w_tz]
            event_times = eventtimes_as_ts

        event_trialnums = event_trials.index.get_level_values('trial_num')

        aligned_epochs = []
        for eventtime in event_times:
            a = pupil_size.loc[eventtime + window[0]: eventtime + window[1]]
            if a.isna().all():
                print(f'{sessname}: {start<eventtime<end= } {eventtime} not found')
                aligned_epochs.append(pd.Series(np.full_like(event_tdeltas, np.nan)))
            else:
                aligned_epochs.append(pupil_size.loc[eventtime + window[0]: eventtime + window[1]])
        for aligned_epoch, eventtime in zip(aligned_epochs, event_times):
            aligned_epoch.index = np.round(aligned_epoch.index - eventtime, 2)
        sessname_list = [sessname] * len(event_times)
        try:
            aligned_epochs_df = pd.DataFrame(aligned_epochs, columns=event_tdeltas,
                                             index=pd.MultiIndex.from_tuples(list(zip(event_times, event_trialnums,
                                                                                      sessname_list)),
                                                                             names=['time', 'trial', 'sess']))
        except pd.errors.InvalidIndexError:
            print(f'{sessname} error')
            print(event_times, event_trialnums, sessname_list)
            raise pd.errors.InvalidIndexError
        
        aligned_epochs_df = aligned_epochs_df.dropna(axis=0, how='all')
        aligned_epochs_df = aligned_epochs_df.ffill(axis=1, )
        aligned_epochs_df = aligned_epochs_df.bfill(axis=1, )
        if baseline_dur:
            epoch_baselines = aligned_epochs_df.loc[:, -baseline_dur:0.0]
            aligned_epochs_df = aligned_epochs_df.sub(epoch_baselines.mean(axis=1), axis=0)

        self.aligned_pupil[event_name] = aligned_epochs_df

    def align2times(self,event_times,size_col,window, sessname, event_name, baseline_dur=0.0):
        pupil_size = self.pupil_data[size_col]
        pupil_isout = self.pupil_data['isout']
        start, end = pupil_size.index[0], pupil_size.index[-1]
        dt = np.round(np.nanmedian(np.diff(pupil_size.index)), 2)
        event_tdeltas = np.round(np.arange(window[0], window[1], dt), 2)

        event_trialnums = np.full_like(event_times, np.nan)
        aligned_epochs = []
        for eventtime in event_times:
            try:
                a = pupil_size.loc[eventtime + window[0]: eventtime + window[1]]
            except KeyError:
                print(f'{sessname}: {start<eventtime<end= } {eventtime}  {event_name} not found. '
                      f'Event {np.where(event_times == eventtime)[0][0] + 1}/{len(event_times)}')
                continue
            if a.isna().all():
                print(f'{sessname}: {start<eventtime<end= } {eventtime} {event_name} not found. '
                      f'Event {np.where(event_times == eventtime)[0][0] + 1}/{len(event_times)}')
                aligned_epochs.append(pd.Series(np.full_like(event_tdeltas, np.nan)))
            else:
                if pupil_isout.loc[eventtime + window[0]: eventtime + window[1]].mean() > 1.5:
                    warnings.warn(f'{sessname}: {eventtime} {event_name} has many outliers. Not using')
                    aligned_epochs.append(pd.Series(np.full_like(event_tdeltas, np.nan)))
                else:
                    aligned_epochs.append(pupil_size.loc[eventtime + window[0]: eventtime + window[1]])
        for aligned_epoch, eventtime in zip(aligned_epochs, event_times):
            aligned_epoch.index = np.round(aligned_epoch.index - eventtime, 2)
        sessname_list = [sessname] * len(event_times)
        aligned_epochs_df = pd.DataFrame(aligned_epochs, columns=event_tdeltas,
                                         index=pd.MultiIndex.from_tuples(list(zip(event_times, event_trialnums,
                                                                                  sessname_list)),
                                                                         names=['time', 'trial', 'sess']))
        aligned_epochs_df = aligned_epochs_df.dropna(axis=0, how='all')
        aligned_epochs_df = aligned_epochs_df.interpolate(limit_direction='both', axis=1)
        if baseline_dur:
            epoch_baselines = aligned_epochs_df.loc[:, -baseline_dur:0.0]
            aligned_epochs_df = aligned_epochs_df.sub(epoch_baselines.mean(axis=1), axis=0)

        self.aligned_pupil[event_name] = aligned_epochs_df