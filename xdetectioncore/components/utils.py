import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def zscore_by_unit(rate_arr, unit_means, unit_stds):
    f, a = plt.subplots(ncols=2)
    if unit_means.ndim == 2:
        unit_means[unit_means == 0] = np.nan
        unit_stds[unit_stds == 0] = np.nan
        unit_means, unit_stds = [np.nanmean(e, axis=0) for e in [unit_means, unit_stds]]

    assert rate_arr.shape[-2] == unit_means.shape[0]
    assert rate_arr.ndim in [2, 3]

    if isinstance(rate_arr, pd.DataFrame):
        rate_arr_df = rate_arr
        rate_arr = rate_arr.values
    else:
        rate_arr_df = None
        
    for ui, (u_mean, u_std) in enumerate(zip(unit_means, unit_stds)):
        if rate_arr.ndim == 3:
            rate_arr[:, ui, :] = (rate_arr[:, ui, :] - u_mean) / u_std
        else:
            rate_arr[ui, :] = (rate_arr[ui, :] - u_mean) / u_std
            
    assert np.isnan(rate_arr).sum() == 0
    if isinstance(rate_arr_df, pd.DataFrame):
        rate_arr = pd.DataFrame(rate_arr, columns=rate_arr_df.columns, index=rate_arr_df.index)
    return rate_arr