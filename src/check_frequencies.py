'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from scipy.signal import welch

from compute_activity import *
from visualize_activity import active_inactive_bout_lengths_quants
from helper_fns import *
from data_handling import *

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def define_activity_quantiles(ds, inactive_quant: float = 0.1, active_quant: float = 0.9):
    ''' Define inactive and active agents in the dataset using quantiles on high-order finite difference speed.'''

    # Define speed thresholds
    v_low = np.nanquantile(ds.v_high_ord.values, inactive_quant)
    v_high = np.nanquantile(ds.v_high_ord.values, active_quant)
    print(f"Inactive speed threshold (<= {inactive_quant}): {v_low}"
          f"\nActive speed threshold (>= {active_quant}): {v_high}")

    # 0: inactive, 1: active, nan: ambiguous
    ds['active'] = xr.where(ds.v_high_ord <= v_low, 0, xr.where(ds.v_high_ord >= v_high, 1, np.nan))

    return ds, v_low, v_high

def validate_quantiles(ds, f_min = 5, inactive_quant: float = 0.1, active_quant: float = 0.9, exp_name = None, batch_num = None, plot = False):
    '''Check average bout length and number of bouts for active and inactive agents for certain quantiles to verify that there is a good number for PSD.'''

    # Determine minimum tracklet length for inclusion in PSD
    fs = 5 # 5 Hz sampling frequency
    min_tracklet_length = np.ceil(5 * fs / f_min) # Want at least 5 cycles for evaluating frequency power
    print(f'Minimum tracklet length: {min_tracklet_length}')

    # Assign activity based on quantiles
    ds, _, _ = define_activity_quantiles(ds, inactive_quant, active_quant)

    # Compute lengths of active and inactive bouts
    active_lengths = run_lengths(ds['active'], val = 1)
    inactive_lengths = run_lengths(ds['active'], val = 0)

    print('Active bouts:')
    print(f'Mean length: {np.mean(active_lengths)}; median length: {np.median(active_lengths)}')
    print(f'Standard deviation: {np.std(active_lengths)}')
    print(f'Number of bouts larger than minimum tracklet length for a minimum resolution of {f_min} Hz: {np.sum(active_lengths >= min_tracklet_length)}')
    print('Inactive bouts:')
    print(f'Mean length: {np.mean(inactive_lengths)}; median length: {np.median(inactive_lengths)}')
    print(f'Standard deviation: {np.std(inactive_lengths)}')
    print(f'Number of bouts larger than minimum tracklet length for a minimum resolution of {f_min} Hz: {np.sum(inactive_lengths >= min_tracklet_length)}')

    if plot:
        active_inactive_bout_lengths_quants(active_lengths, inactive_lengths, f_min, [inactive_quant, active_quant], min_tracklet_length, exp_name, batch_num)

def compute_psd_tracklets(da: xr.DataArray, fs: float, f_min: float, max_segments: float = 1e9):
    """
    Compute PSD for a DataArray of shape (id, frame) where frame indices may
    contain gaps. Each (id)'s continuous frame segments are treated as independent
    tracklets. Only tracklets >= min_tracklet_length are included.

    Returns an xr.Dataset with averaged PSD and frequency values.
    """

    min_tracklet_length = int(np.ceil(5 * fs / f_min))   # 5 cycles at f_min
    target_nperseg = int(np.ceil(3 * fs / f_min))  # Resolution = f_min/3
    max_allowed = min(target_nperseg, min_tracklet_length)

    # Use power of 2 <= max_allowed
    nperseg = 2 ** int(np.floor(np.log2(max_allowed)))
    noverlap = nperseg // 2

    # Frequency vector
    freq, _ = welch(np.zeros(nperseg), fs=fs, nperseg=nperseg, noverlap=noverlap)
    n_freq = len(freq)

    # Loop over ids
    all_valid_trajs = []
    for i in range(da.sizes["id"]):
        vals = da.isel(id=i).values
        frames = da["frame"].values

        # Mask out invalid values, e.g. gaps
        mask = ~np.isnan(vals)

        if not np.any(mask):
            continue

        valid_vals = vals[mask]
        valid_frames = frames[mask]

        # Find continuous frame segments
        breaks = np.where(np.diff(valid_frames) != 1)[0] + 1
        segments = np.split(np.arange(len(valid_frames)), breaks)

        # Add to list of all segments if long enough
        for seg in segments:
            if len(seg) >= min_tracklet_length:
                all_valid_trajs.append(valid_vals[seg])

    # If more segments than max_segments, randomly select max_segments from list
    if len(all_valid_trajs) > max_segments:
        all_valid_trajs = np.random.choice(all_valid_trajs, size=int(max_segments), replace=False).tolist()

    # Process each segment as a tracklet
    psd_list = []
    seg_lengths = []
    for traj in all_valid_trajs:
        
        # Compute PSD
        _, Pxx = welch(traj, fs=fs, nperseg=nperseg, noverlap=noverlap)
        psd_list.append(Pxx)
        seg_lengths.append(len(traj))

    # Average or return NaN
    if len(psd_list) == 0:
        psd_avg = np.full(n_freq, np.nan)
        variance = np.full(n_freq, np.nan)
    else:
        weights = np.array(seg_lengths)/np.sum(seg_lengths)
        psd_avg = np.average(psd_list, axis=0, weights=weights) # Weighted average
        variance = np.average((psd_list - psd_avg)**2, axis = 0, weights=weights) # Weighted variance

    return psd_avg, variance, freq

def compute_activity_psd(ds, f_min: float, inactive_quant: float, active_quant: float, exp_name: str, batch_num: int, smooth_list: list = ['high_ord'], fs: float = 5):

    # Assign activity based on quantiles
    ds, _, _ = define_activity_quantiles(ds, inactive_quant, active_quant)

    # Determine number of active and inactive tracklets that are long enough
    min_tracklet_length = int(np.ceil(5 * fs / f_min))   # 5 cycles at f_min
    num_active = np.sum(run_lengths(ds['active'] == 1, val = 1) >= min_tracklet_length)
    num_inactive = np.sum(run_lengths(ds['active'] == 0, val = 0) >= min_tracklet_length)
    max_segments = min(num_active, num_inactive) # Downsample to have equal samples

    h5_path = f"./preprocessed/{exp_name}/batch_{batch_num}/psd.h5"

    # We load the PSR h5 to check it quantities have been pre-computed.
    existing = load_psds_hdf5(h5_path)

    # Compute PSDs
    for smooth_name in smooth_list:
        for activity in range(2):
            for coordinate in ['x', 'y']:
                activity_name = ['inactive', 'active'][activity]
                precomputed = (str(f_min) in existing and activity_name in existing[str(f_min)] and str([inactive_quant, active_quant]) in existing[str(f_min)][activity_name] 
                               and coordinate in existing[str(f_min)][activity_name][str([inactive_quant, active_quant])]
                               and smooth_name in existing[str(f_min)][activity_name][str([inactive_quant, active_quant])][coordinate])

                if not precomputed:
                    mean, var, freq = compute_psd_tracklets(ds[f'{coordinate}_{smooth_name}'].where(ds['active'] == activity), fs=fs, f_min=f_min, max_segments=max_segments)

                    # Save PSD data
                    save_psd_hdf5(h5_path, f_min, activity_name, coordinate, smooth_name, mean, var, freq, str([inactive_quant, active_quant]))

    return

def compute_total_psd(ds, f_min: float, exp_name: str, batch_num: int, smooth_list: list = ['high_ord'], fs: float = 5):
    # Compute the psd on all tracklets in this batch

    # Determing minimum length of tracklets
    min_tracklet_length = int(np.ceil(5*fs/f_min)) # 5 cycles at f_min
    num_eligible = [int(np.sum(run_lengths(ds[f'x_{speed}']) > min_tracklet_length)) for speed in smooth_list]
    print('Number of eligible tracklets (per smoothing method):', num_eligible)
    
    # Compute PSDs
    for smooth_name in smooth_list:
        for coordinate in ['x', 'y']:
            mean, var, freq = compute_psd_tracklets(ds[f'{coordinate}_{smooth_name}'], fs=fs, f_min=f_min)

            # Save PSD data
            save_psd_hdf5(f"./preprocessed/{exp_name}/batch_{batch_num}/psd.h5", f_min, 'all', coordinate, smooth_name, mean, var, freq)
