import argparse
import pickle
import platform
from copy import copy
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_ind
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm


from ..session import get_predictor_from_psth
from ..decoding.decoding_funcs import Decoder
from ..paths import posix_from_win
from ..io_utils import load_sess_pkl
from ..pupil.pupil_analysis_funcs import process_pupil_td_data
from ..ephys.spike_time_utils import zscore_by_trial



def aggregate_event_responses(sessions: dict, events=None, events2exclude=None, window=(0, 0.25),
                              pred_from_psth_kwargs=None,zscore_by_trial_flag=False ):

    events_by_session = [list(sess.sound_event_dict.keys()) for sess in sessions.values()]

    if events is None:
        raise NotImplementedError

    if events:
        sessions2use = [sess for sess, s_events in zip(sessions, events_by_session) if
                        all(e in s_events for e in events)]
    else:
        sessions2use = sessions

    if events:
        for sess in copy(sessions2use):
            if not all([e in list(sessions[sess].sound_event_dict.keys()) for e in events]):
                sessions.pop(sess)
                sessions2use.remove(sess)

    # if events2exclude:
    #     common_events = [e for e in common_events if e not in events2exclude]


    event_responses_across_sessions = {sessname: {e: get_predictor_from_psth(sessions[sessname], e, [-2, 3], window,
                                                                             **pred_from_psth_kwargs if pred_from_psth_kwargs else {})
                                                  for e in events}
                                       for sessname in
                                       tqdm(sessions2use, total=len(sessions2use), desc='Getting event responses')}
    if zscore_by_trial_flag:
        event_responses_across_sessions = zscore_by_trial(event_responses_across_sessions)
    return event_responses_across_sessions


def aggregate_event_features(sessions: dict, events=None, events2exclude=None):
    events_by_session = [list(sess.sound_event_dict.keys()) for sess in sessions.values()]
    # print(f'common events = {common_events}')

    if events is None:
        raise NotImplementedError

    if events:
        sessions2use = [sess for sess, s_events in zip(sessions, events_by_session) if all(e in s_events for e in events)]
    else:
        sessions2use = sessions

    if events:
        for sess in copy(sessions2use):
            if not all([e in list(sessions[sess].sound_event_dict.keys()) for e in events]):
                sessions.pop(sess)
                sessions2use.remove(sess)
    if len(sessions2use) == 0:
        return {}

    event_features_across_sessions = {sessname: {e: {'times': sessions[sessname].sound_event_dict[e].times.values,
                                                     'trial_nums': sessions[sessname].sound_event_dict[e].trial_nums, }
                                                 for e in events}
                                      for sessname in
                                      tqdm(sessions2use, total=len(sessions2use), desc='Getting event features')}
    for sessname in tqdm(event_features_across_sessions, total=len(event_features_across_sessions), desc='Adding td_df'):
        sess_td_df = sessions[sessname].td_df
        sess_td_df.index = sess_td_df.reset_index(drop=True).index+1
        for e in event_features_across_sessions[sessname]:
            trial_nums = event_features_across_sessions[sessname][e]['trial_nums']
            event_features_across_sessions[sessname][e]['td_df'] = sess_td_df.query('index in @trial_nums')
    return event_features_across_sessions


def concatenate_responses_by_td(responses_by_sess: dict, features_by_sess: dict, td_df_query=None):
    grouped_by_td = {}
    common_events = set.intersection(*[set(list(e.keys())) for e in responses_by_sess.values()])
    for event in tqdm(common_events, total=len(common_events), desc='Concatenating responses'):
        event_across_sess = [sess_responses[event][:len(sess_features[event]['td_df'])] for sess_responses,sess_features
                             in zip(responses_by_sess.values(), features_by_sess.values())]
        event_idxbool = [sess[event]['td_df'].eval(td_df_query).values for sess in features_by_sess.values()]
        try:
            grouped_by_td[event] = np.concatenate([np.nanmean(sess_e_responses[sess_idx_bool], axis=0)
                                                   for sess_e_responses, sess_idx_bool in
                                                   tqdm(zip(event_across_sess, event_idxbool), total=len(event_across_sess),
                                                        desc='Concatenating features')], axis=0)
        except IndexError:
            print(f'IndexError for {event}')
            grouped_by_td[event] = None

    return grouped_by_td


def group_ephys_resps(event_name:str,responses_by_sess: dict, features_by_sess: dict, td_df_query=None,
                      trial_nums2use=None,sess_list=None):
    resps_across_sess = {}
    trial_nums = {}
    for sess in sess_list if sess_list else responses_by_sess:
        if td_df_query:
            sess_resp_idxs = features_by_sess[sess][event_name]['td_df'].eval(td_df_query).values
        elif trial_nums2use is not None:
            sess_resp_idxs = np.isin(features_by_sess[sess][event_name]['trial_nums'], trial_nums2use)
        else:
            raise ValueError('must specify td_df_query or trial_nums2use')
        # print(sess_resp_idxs)
        if not np.any(sess_resp_idxs) or sess_resp_idxs.sum()<10:
            print(f'{sess} has not enough responses')
            # resps_across_sess[sess] = None
            continue
        sess_resps = responses_by_sess[sess][event_name]
        # print(f'{sess} has {len(sess_resps)} responses, {len(sess_resp_idxs)} indices')
        resps_across_sess[sess] = sess_resps[:len(sess_resp_idxs)][sess_resp_idxs]
        trial_nums[sess] = features_by_sess[sess][event_name]['trial_nums'][:len(sess_resp_idxs)][sess_resp_idxs]
    return resps_across_sess,trial_nums


def group_responses_by_pip_prop(responses_by_sess: dict, pip_desc:dict, properties=None,concatenate_flag=True):
    pip_groups_by_prop = {prop: {val: [pip for pip in pip_desc[prop].keys() if pip_desc[prop][pip] == val]
                                 for val in np.unique(list(pip_desc[prop].values()))} for prop in pip_desc}
    if not properties:
        properties = list(pip_groups_by_prop.keys())
    if concatenate_flag:
        responses_by_prop = {prop: {val: np.concatenate([responses_by_sess[pip] for pip in pip_groups_by_prop[prop][val]
                                                         if pip in responses_by_sess],
                                                        axis=0)
                                    for val in pip_groups_by_prop[prop]} for prop in
                             tqdm(properties, total=len(pip_groups_by_prop), desc='Concatenating responses')}
    else:
        responses_by_prop = {prop: {val: {pip:responses_by_sess[pip] for pip in pip_groups_by_prop[prop][val]
                                          if pip in responses_by_sess}
                                    for val in pip_groups_by_prop[prop]} for prop in
                             tqdm(properties, total=len(pip_groups_by_prop), desc='Concatenating responses')}

    return responses_by_prop


def get_responses_by_pip_and_condition(pip_names, event_responses, event_features, conds, cond_filters,
                                       zip_pip_conds=False):
    """
    Returns a dictionary with keys '{pip}_{cond}' and values as dicts: {sessname: response array}.
    Args:
        pip_names (list): List of pip event names (e.g. ['A-0', 'B-0']).
        event_responses (dict): Nested dict of session -> event -> response arrays.
        event_features (dict): Nested dict of session -> event -> features (e.g. 'td_df').
        conds (list): List of condition names (must be keys in cond_filters).
        cond_filters (dict): Dictionary mapping condition names to query strings.
    Returns:
        dict: {f'{pip}_{cond}': {sessname: response array}}
    """
    responses_by_pip_cond = {}

    if zip_pip_conds:
        pips2use = [f'{pip}_{cond}' for pip,cond in zip(pip_names, conds)]
    else:
        pips2use = [f'{pip}_{cond}' for pip in pip_names for cond in conds]
    for sessname in event_responses:
        responses_by_pip_cond[sessname] = {}
        for key in pips2use:
            pip = key.split('_')[0]
            responses_by_pip_cond[sessname][key] = np.array([])
            # Check if pip and cond exist for this session
            if pip not in event_responses[sessname]:
                continue
            if pip not in event_features[sessname]:
                continue

            cond = '_'.join(key.split('_')[1:])
            cond_query = cond_filters[cond]

            if sessname not in list(event_features):
                print(f'Session {sessname} not in event features. Skipping...')
                continue

            td_df = event_features[sessname][pip].get('td_df', None)
            if td_df is None:
                continue
            try:
                trial_mask = td_df.eval(cond_query)
            except AssertionError:
                continue
            # Only keep if enough trials
            if trial_mask.sum() < 1:
                continue
            n_events = min(len(event_responses[sessname][pip]), len(trial_mask))
            # responses_by_pip_cond[key][sessname] = event_responses[sessname][pip][:n_events][trial_mask.values[:n_events]]
            responses_by_pip_cond[sessname][key] = event_responses[sessname][pip][:n_events][trial_mask.values[:n_events]]

    # clean up dict
    for sessname in list(responses_by_pip_cond.keys()):
        if len(responses_by_pip_cond[sessname]) == 0:
            responses_by_pip_cond.pop(sessname)
    return responses_by_pip_cond


def decode_responses(predictors, features, model_name='logistic', dec_kwargs=None):
    decoder = {dec_lbl: Decoder(np.vstack(predictors), np.hstack(features), model_name=model_name)
               for dec_lbl in ['data', 'shuffled']}
    dec_kwargs = dec_kwargs or {}
    [decoder[dec_lbl].decode(dec_kwargs=dec_kwargs | {'shuffle': shuffle_flag}, parallel_flag=False,
                             )
     for dec_lbl, shuffle_flag in zip(decoder, [False, True])]

    return decoder


def plot_aggr_cm(pip_cms, im_kwargs=None, **cm_kwargs):
    cm_plot = ConfusionMatrixDisplay(np.mean(pip_cms,axis=0) if pip_cms.ndim==3 else pip_cms,
                                     display_labels=cm_kwargs.get('labels',list(range(pip_cms.shape[-1]))))
    cm_plot.plot(cmap=cm_kwargs.get('cmap','bwr'),
                 include_values=cm_kwargs.get('include_values',False),colorbar=cm_kwargs.get('colorbar',True),
                 im_kw=im_kwargs if im_kwargs is not None else {},
                 ax=cm_kwargs.get('ax',None))

    cm_plot.ax_.invert_yaxis()
    cm_plot.ax_.set_xlabel('')
    cm_plot.ax_.set_ylabel('')
    cm_plot.figure_.set_size_inches(cm_kwargs.get('figsize',(2,2)))
    # cm_plot.figure_.set_layout_engine('constrained')

    return cm_plot.figure_, cm_plot.ax_


def decode_over_sliding_t(resps_dict: dict, window_s, resp_window, pips_as_ints_dict, pips2decode: list,
                          animals_to_use:list=None):
    resp_width = list(resps_dict.values())[0][pips2decode[0]].shape[-1]
    resp_x_ser = np.round(np.linspace(resp_window[0], resp_window[1], resp_width), 2)
    window_size = int(np.round(window_s / (resp_x_ser[1] - resp_x_ser[0])))

    dec_res_dict = {}
    bad_dec_sess = set()
    for sessname in tqdm(list(resps_dict.keys()), total=len(resps_dict),
                         desc='decoding across sessions'):
        # if not any(e in sessname for e in ['DO79', 'DO81']):
        if animals_to_use:
            if not any(e in sessname for e in animals_to_use):
                continue

        for t in tqdm(range(resp_width - window_size), total=resp_width - window_size, desc='decoding across time'):
            if 'A-0' in pips2decode:
                xys = [(
                    resps_dict[sessname][pip][150:200][::3] if pip.split('-')[1] == '0' else
                    resps_dict[sessname][pip],
                    np.full_like(resps_dict[sessname][pip][:, 0, 0], pips_as_ints_dict[pip]))
                    for pip in pips2decode]
            else:
                xys = [(resps_dict[sessname][pip],
                        np.full_like(resps_dict[sessname][pip][:, 0, 0], pips_as_ints_dict[pip]))
                       for pip in pips2decode]
            ys = [np.full(xy[0].shape[0], pips_as_ints_dict[pip]) for xy, pip in zip(xys, pips2decode)]
            xs = np.vstack([xy[0][:, :, t:t + window_size].mean(axis=-1) for xy in xys])
            # xs = [xy[0][:,:,15:].mean(axis=-1) for xy in xys]
            ys = np.hstack(ys)
            # if np.unique(ys).shape[0] < len(patt_is):
            #     continue
            try:
                dec_res_dict[f'{sessname}-{t}:{t + window_size}s'] = decode_responses(xs, ys, n_runs=50,
                                                                                      dec_kwargs={'cv_folds': 10})
            except ValueError:
                print(f'{sessname} failed')

                bad_dec_sess.add(sessname)
                continue
    # [decoders_dict[dec_name]['data'].plot_confusion_matrix([f'{pip}-{pip_i}' for pip in 'D' for pip_i in patt_is])
    #  for dec_name in decoders_dict.keys()]
    sess2use = [dec_name.split('-')[0] for dec_name in dec_res_dict.keys()]
    norm_dev_accs_ts_dict = {
        sessname: {t: np.mean(dec_res_dict[f'{sessname}-{t}:{t + window_size}s']['data'].accuracy)
                   for t in range(resp_width - window_size)}
        for sessname in sess2use if sessname not in bad_dec_sess}

    norm_dev_accs_ts_df = pd.DataFrame(norm_dev_accs_ts_dict).T
    norm_dev_accs_ts_df.columns = np.round(resp_x_ser[window_size:], 2)

    return norm_dev_accs_ts_df, norm_dev_accs_ts_dict


def predict_from_responses(dec_model,responses):
    return dec_model.predict(responses)


def predict_over_sliding_t(dec_model,resps_dict,pips2predict,window_s,resp_window):

    resp_width = list(resps_dict.values())[0][pips2predict[0]].shape[-1]
    resp_x_ser = np.round(np.linspace(resp_window[0], resp_window[1], resp_width), 2)
    window_size = int(np.round(window_s / (resp_x_ser[1] - resp_x_ser[0])))

    responses = []
    for pips2predict in pips2predict:
        pip_resp = resps_dict[pips2predict]
        if pip_resp.ndim == 3:
            pass

    return dec_model.predict(responses[:,:,window_size:])



def run_decoding(event_responses, x_ser, decoding_windows, pips2decode, cache_path=None, overwrite=False, **kwargs):
    """
    Run decoding analysis for specified pip pairs and return a DataFrame with one row per session,
    columns for each decoding's data/shuff accuracy.

    Args:
        event_responses (dict): Nested dict of session -> event -> response arrays.
        animals (list): List of animal names to include.
        x_ser (np.ndarray): Time axis for windowing.
        decoding_windows (list): List of [start, end] windows for decoding.
        pips2decode (list): List of [pip1, pip2] pairs to decode.
        cache_path (Path or str, optional): Path to pickle file for caching results.
        overwrite (bool): If True, recompute even if cache exists.

    Returns:
        pd.DataFrame: Index sessname, columns for each decoding's data/shuff accuracy.
    """


    # if cache_path is not None and Path(cache_path).is_file() and not overwrite:
    #     with open(cache_path, 'rb') as f:
    #         all_results_df = pickle.load(f)
    #     return all_results_df

    records = []
    cms = []
    for sessname in tqdm(event_responses.keys(), desc='decoding sessions', total=len(event_responses)):
        session_events = event_responses[sessname]
        record = {'sess': sessname, 'name': sessname.split('_')[0]}
        for pips, dec_wind in zip(pips2decode, decoding_windows):
            dec_sffx = "_vs_".join(pips)
            if not all(p in session_events for p in pips):
                record[f'{dec_sffx}_data_accuracy'] = np.nan
                record[f'{dec_sffx}_shuffled_accuracy'] = np.nan
                continue

            xs_list = [session_events[pip] for pip in pips]
            idx_4_decoding = [np.where(x_ser == t)[0][0] for t in dec_wind]
            xs = np.vstack([x[:, :, idx_4_decoding[0]:idx_4_decoding[1]].mean(axis=-1) for x in xs_list])
            ys = np.hstack([np.full(x.shape[0], ci) for ci, x in enumerate(xs_list)])

            dec_kwargs = kwargs.get('dec_kwargs', {})
            if kwargs.get('train_split_by_cond'):
                conds = list(set(['_'.join(pip.split('_')[1:]) for pip in pips]))
                n_conds1 = session_events[f'{pips[0]}'].shape[0]
                print(f'Debugging: n_conds1 = {n_conds1}, '
                      f'len all = {[session_events[f"{p}"].shape[0] for p in pips]}')
                dec_kwargs['pre_split'] = n_conds1*(len(pips)//len(conds))
                dec_kwargs['cv_folds'] = 0
                ys = ys % int(len(pips)/len(conds))


            try:
                decode_result = decode_responses(xs, ys, dec_kwargs=dec_kwargs)
                decode_result['data'].plot_confusion_matrix(labels=set(ys))

                cms.append(decode_result['data'].cm)

            except (AssertionError, ValueError) as e:
                print(e)
                print(f'WARNING: Could not decode session {sessname}')
                continue
            record[f'{dec_sffx}_data_accuracy'] = np.nanmean(decode_result['data'].accuracy)
            record[f'{dec_sffx}_shuffled_accuracy'] = np.nanmean(decode_result['shuffled'].accuracy)
        records.append(record)
    df = pd.DataFrame(records)
    # Remove duplicate sessname rows by grouping and keeping the first (should not happen, but just in case)
    df = df.groupby('sess', as_index=False).first().set_index('sess')
    if cache_path is not None:
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)

    return df, np.array(cms)


def ttest_decoding_results(decode_dfs, key, col1='data_accuracy', col2='shuff_accuracy'):
    """
    Perform an independent t-test between two columns in the decoding results DataFrame for a given key.

    Args:
        decode_dfs (dict): Output from run_decoding, mapping dec_sffx to DataFrame.
        key (str): Key for the decoding comparison (e.g., 'A-0_vs_base').
        col1 (str): First column for t-test (default 'data_accuracy').
        col2 (str): Second column for t-test (default 'shuff_accuracy').

    Returns:
        ttest_result: scipy.stats.ttest_ind result object.
    """
    df = decode_dfs[key]
    return ttest_ind(df[col1], df[col2], alternative='greater', equal_var=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('pkldir')
    parser.add_argument('event_responses_pkl')
    parser.add_argument('--plot_config_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--by_animal', action='store_true')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--multiprocess', action='store_true')
    parser.add_argument('--skip_batches_if_exist', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    pass


