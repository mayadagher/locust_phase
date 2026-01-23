'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import pynumdiff
from scipy.signal import savgol_filter
import scipy.signal
import scipy
import time
import os

from data_handling import load_trex_data, save_ds

'''_____________________________________________________COMPUTATION FUNCTIONS____________________________________________________________'''

def interpolate_small_gaps(x, y, missing, max_gap=1, max_dist=0.5):
    """
    Conditionally interpolate NaN gaps in x,y trajectories.
    Only interpolate if the gap is shorter than `max_gap`
    and the Euclidean distance across the gap is less than `max_dist`.
    """
    x = np.array(x, copy=True)
    y = np.array(y, copy=True)
    missing = np.array(missing, copy = True)
    isnan = np.isnan(x) | np.isnan(y)

    if not np.any(isnan):
        return x, y, missing  # nothing to do

    # Find start and end indices of NaN runs
    diffs = np.diff(np.concatenate([[0], isnan.astype(int), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    s_idx = 0
    for s, e in zip(starts, ends):
        s_idx += 1
        gap_size = e - s
        # skip if at boundary
        if s == 0 or e >= len(x):
            continue

        dx = x[e] - x[s-1]
        dy = y[e] - y[s-1]
        dist_gap = np.sqrt(dx**2 + dy**2)

        if gap_size <= max_gap and dist_gap <= max_dist:

            # Linear interpolate across this small gap
            x[s:e] = np.linspace(x[s-1], x[e], gap_size + 2)[1:-1]
            y[s:e] = np.linspace(y[s-1], y[e], gap_size + 2)[1:-1]
            missing[s:e] = 2 # Indicates interpolated values

    return x, y, missing

def sg_window(x, window_length=7, polyorder=2, deriv=0, delta=1.0):
    """Savitzky-Golay filter with corrected window length."""
    n = len(x)
    wl = min(window_length, n if (n % 2 == 1) else n - 1)
    if wl < polyorder + 2:
        wl = polyorder + 2
    if wl % 2 == 0:
        wl += 1
    if wl > n:
        wl = n if n % 2 == 1 else n - 1
    x_hat = savgol_filter(x=x, window_length=wl, polyorder=polyorder, deriv=deriv, delta=delta, mode = 'interp')
    
    return x_hat

def butter_zero_phase(x, dt=1, filter_order=2, cutoff_freq=0.5):
    """Zero-phase Butterworth filter for 1D array (frame axis). Filtfilt passes filter over data backwards and forwards so that there is no time shift. Results in effectively lower cutoff frequency."""
    n = len(x)
    b, a = scipy.signal.butter(filter_order, cutoff_freq)

    padlen = n - 1
    x_hat = scipy.signal.filtfilt(b, a, x, axis=0, method="pad", padlen=padlen) # applies forward and backward pass so zero phase

    return x_hat

def spline_scipy(x, degree=3, s=1):
    '''x is strictly contiguous.'''

    t = np.arange(len(x))

    spline = scipy.interpolate.UnivariateSpline(t, x, k=degree, s=s*len(x))
    x_hat = spline(t)

    return x_hat

def smooth_nantolerant(x: xr.DataArray, func, params: dict, min_tracklet_length: int):
    """Apply a smoothing function to a 1D array (frame axis) while tolerating NaNs."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    valid = ~np.isnan(x)

    # Find contiguous segments of not-nan
    edges = np.diff(valid.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if valid[0]:
        starts = np.r_[0, starts]
    if valid[-1]:
        ends = np.r_[ends, len(x)]

    for s, e in zip(starts, ends):
        seg = x[s:e]
        n = len(seg)
        if n < min_tracklet_length:  # Too short for interpretation: exclude tracklet
            out[s:e] = np.nan
            continue

        out[s:e] = func(seg, **params)

    return out

def compute_speed(ds, speed_dict, fs: int = 5):
    '''Compute speed using methods specified by speed_types (a dictionary with speed type as keys and parameter dictionaries as values). fs is sample frequency and is used to compute minimum tracklet length used for smoothing (or a tracklet will be excluded).'''
    speed_types = list(speed_dict.keys())

    if 'raw' in speed_types: # Compute instantaneous speed from raw positions
        ds['v_raw'] = np.hypot(ds['x_raw'].diff(dim = 'frame'), ds['y_raw'].diff(dim = 'frame'))

    if 'high_ord' in speed_types: # Compute high-order derivatives
        high_ord_params = speed_dict['high_ord']
        ds['x_high_ord'], vx_high_ord = xr.apply_ufunc(pynumdiff.finite_difference.finitediff, ds['x_raw'], input_core_dims=[['frame']], output_core_dims=[['frame'],['frame']], vectorize = True,  kwargs=high_ord_params, dask = 'parallelized')
        ds['y_high_ord'], vy_high_ord = xr.apply_ufunc(pynumdiff.finite_difference.finitediff, ds['y_raw'], input_core_dims=[['frame']], output_core_dims=[['frame'],['frame']], vectorize = True,  kwargs=high_ord_params, dask = 'parallelized')
        
        # Get speed magnitude
        ds['v_high_ord'] = np.hypot(vx_high_ord, vy_high_ord)

    if 'moving_avg' in speed_types: # Compute moving average speed
        moving_params = speed_dict['moving_avg']
        ds['x_moving_avg'] = ds['x_raw'].rolling(frame=moving_params['window_length'], center=moving_params['center'], min_periods = 1).mean()
        ds['y_moving_avg'] = ds['y_raw'].rolling(frame=moving_params['window_length'], center=moving_params['center'], min_periods = 1).mean()
        ds['v_moving_avg'] = np.hypot(ds['x_moving_avg'].diff(dim='frame'), ds['y_moving_avg'].diff(dim='frame'))

    if 'moving_med' in speed_types: # Compute moving median speed
        moving_params = speed_dict['moving_med']
        ds['x_moving_med'] = ds['x_raw'].rolling(frame=moving_params['window_length'], center=moving_params['center'], min_periods = 1).median()
        ds['y_moving_med'] = ds['y_raw'].rolling(frame=moving_params['window_length'], center=moving_params['center'], min_periods = 1).median()
        ds['v_moving_med'] = np.hypot(ds['x_moving_med'].diff(dim='frame'), ds['y_moving_med'].diff(dim='frame'))

    if 'sg' in speed_types: # Compute Savitzky-Golay smoothed speed
        sg_params = speed_dict['sg']
        min_tracklet_length = np.ceil(5*sg_params['window_length'])

        # Smooth x and y (vectorized over ids)
        ds['x_sg'] = xr.apply_ufunc(smooth_nantolerant, ds['x_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs= {'func': sg_window, 'params': sg_params, 'min_tracklet_length': min_tracklet_length}, dask = 'parallelized')
        ds['y_sg'] = xr.apply_ufunc(smooth_nantolerant, ds['y_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs= {'func': sg_window, 'params': sg_params, 'min_tracklet_length': min_tracklet_length}, dask = 'parallelized')

        # Speed magnitude
        ds['v_sg'] = np.hypot(ds['x_sg'].diff(dim='frame'), ds['y_sg'].diff(dim='frame'))

    if 'butter' in speed_types: # Compute Butterworth filtered speed
        butter_params = speed_dict['butter']
        min_tracklet_length = np.ceil(10*fs/butter_params['cutoff_freq'])

        ds['x_butter'] = xr.apply_ufunc(smooth_nantolerant, ds['x_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize = True,  kwargs={'func': butter_zero_phase, 'params': butter_params, 'min_tracklet_length': min_tracklet_length}, dask = 'parallelized')
        ds['y_butter'] = xr.apply_ufunc(smooth_nantolerant, ds['y_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize = True,  kwargs={'func': butter_zero_phase, 'params': butter_params, 'min_tracklet_length': min_tracklet_length}, dask = 'parallelized')

        # Speed magnitude
        ds['v_butter'] = np.hypot(ds['x_butter'].diff(dim='frame'), ds['y_butter'].diff(dim='frame'))

    if 'spline' in speed_types: # Compute spline smoothed speed
        spline_params = speed_dict['spline']
        min_tracklet_length = 10*(spline_params['degree'] + 1)

        ds['x_spline'] = xr.apply_ufunc(smooth_nantolerant, ds['x_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs={'func': spline_scipy, 'params': spline_params, 'min_tracklet_length': min_tracklet_length}, dask='parallelized')
        ds['y_spline'] = xr.apply_ufunc(smooth_nantolerant, ds['y_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs={'func': spline_scipy, 'params': spline_params, 'min_tracklet_length': min_tracklet_length}, dask='parallelized')

        # Speed magnitude
        ds['v_spline'] = np.hypot(ds['x_spline'].diff(dim='frame'), ds['y_spline'].diff(dim='frame'))
    return ds

def exclude_borders(ds, radius): # Sets missing = 1 for individuals outside a circular region and nan for all other variables
    ds_copy = ds.copy()
    center = (1920/2, 1920/2)
    mask = (ds_copy['x_raw'] - center[0])**2 + (ds_copy['y_raw'] - center[1])**2 > radius**2 # mask for individuals outside of center

    # Set all data variables to np.nan where mask is True
    ds_copy = xr.where(mask, np.nan, ds_copy)

    # Set missing = 1 where mask is True
    ds_copy['missing'] = xr.where(mask, 1, ds_copy['missing'])

    return ds_copy

def compute_tracklet_lengths_and_ids(missing_1d, fill_gaps):
        """
        Compute per-frame tracklet lengths and segment IDs for a 1D boolean array
        indicating missing values.

        Parameters
        ----------
        missing_1d : array-like of bool or nan
            True = missing, False = valid (or NaN treated as missing)

        Returns
        -------
        lengths : np.ndarray
            Array of same shape as input; contains the total length of each
            contiguous valid segment, NaN where missing.
        segment_ids : np.ndarray
            Array of same shape; each valid segment gets a unique integer ID (1, 2, ...),
            and NaN where missing.
        """

        missing_1d = np.array(missing_1d, copy = True)
        missing_1d[np.isnan(missing_1d)] = 1  # Treat NaNs as missing
        missing_1d[missing_1d == 2] = 1 - int(fill_gaps) # Interpolated values are treated as present if fill_gaps is True, otherwise they're treated as absent
        present = ~missing_1d.astype(bool)

        n = len(present)
        lengths = np.full(n, np.nan)
        segment_ids = np.full(n, np.nan)

        # Detect where new present segments start (False → True transition)
        starts = present & ~np.roll(present, 1)
        starts[0] = present[0]

        # Assign segment IDs (increment when a new present block starts)
        seg_id = np.cumsum(starts)
        seg_id[~present] = 0  # keep missing as 0
        unique_ids, counts = np.unique(seg_id[seg_id > 0], return_counts=True)

        # Map each segment ID to its length
        seg_len_map = np.zeros(seg_id.max() + 1, dtype=float)
        seg_len_map[unique_ids] = counts

        # Fill in lengths and segment IDs where valid
        lengths[present] = seg_len_map[seg_id[present]]
        segment_ids[present] = seg_id[present]

        # Convert missing entries back to NaN
        segment_ids[~present] = np.nan
        lengths[~present] = np.nan

        return lengths, segment_ids

def smooth_circular(ds, smooth_func, speed_dict):
    ''' Apply NaN-tolerant smoothing with arbitrary smoothing functions to circular data (like theta).'''

    def smooth_1d(angle_1d):

        # Create a mask for nans
        mask = np.isnan(angle_1d)
        if np.all(mask):
            return angle_1d  # all NaNs → unchanged

        # Temporarily unwrap angles to remove jumps at ±π
        unwrapped = np.unwrap(angle_1d[~mask])

        # Apply filter on valid data only
        if smooth_func.__name__ == 'finitediff': # finitediff has two outputs
            _, smoothed = smooth_func(unwrapped, **speed_dict)
        else:
            dict_copy = speed_dict.copy()
            if smooth_func.__name__ == 'savgol_filter': # assure parameters are appropriate
                dict_copy['window_length'] = min(dict_copy['window_length'], unwrapped.size - (1 - (unwrapped.size % 2)))
                dict_copy['polyorder'] = min(dict_copy['polyorder'], unwrapped.size - 1)
            smoothed = smooth_func(unwrapped, **dict_copy) # ** unpacks dictionary key-value combos

        # Re-wrap to [-π, π]
        smoothed_wrapped = (smoothed + np.pi) % (2 * np.pi) - np.pi

        # Reinsert NaNs
        result = np.full_like(angle_1d, np.nan, dtype=float)
        result[~mask] = smoothed_wrapped
        return result
    
    # Apply vectorized across all non-frame dims
    smoothed = xr.apply_ufunc(smooth_1d, ds['theta_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, dask='allowed', output_dtypes=[float])

    return smoothed

def compute_theta(ds, speed_dict): # Speeds should be pre-computed
    ''' Compute heading direction and instantaneous change in heading direction.'''
    speed_types = list(speed_dict.keys())

    def circular_deriv(thetas):

        new_thetas = thetas.copy()
        nans = np.isnan(thetas)
        valid_unwrapped = np.unwrap(thetas[~nans]) # Unwrap to remove jumps at ±π
        new_thetas[~nans] = valid_unwrapped 
        v = np.diff(new_thetas) # Differentiate

        # Prepend NaN so the returned derivative has the same length as the input
        return np.concatenate(([np.nan], v))

    # Compute (smoothed) orientations
    for speed in speed_types:
        # arctan2 of diffs produces an array of length (frames - 1); reindex to the full frame index to restore original length
        theta = np.arctan2(ds[f'y_{speed}'].diff(dim='frame'), ds[f'x_{speed}'].diff(dim='frame'))
        theta = theta.reindex({"frame": ds['frame']})
        ds[f'theta_{speed}'] = theta
        ds[f'vtheta_{speed}'] = xr.apply_ufunc(circular_deriv, ds[f'theta_{speed}'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, dask='allowed', output_dtypes=[float])
    
    return ds

def compute_dist_from_center(ds, center=(1920/2, 1920/2)):
    ''' Compute distance from center for each individual at each frame using high_ord smoothed positions.'''
    ds_copy = ds.copy()
    ds_copy['dist_from_center'] = np.hypot(ds_copy['y_high_ord'] - center[1], ds_copy['x_high_ord'] - center[0])
    return ds_copy

def preprocess_raw(ds, speed_dict, fill_gaps = False, interp_dict = None, center_only = False, radius = None):
    ''' Interpolates and computes speed, excludes borders, computes tracklet lengths, and gives tracklets identities. Arguably, repeatedly computing speed/interpolating is wasteful if we want to compare
    effects of excluding borders on tracklet lengths, but I can easily split this function later if needed.
    '''
    ds['missing'] = xr.where(np.isnan(ds['missing']), 1, ds['missing']) # 1: missing, 0: present

    # Interpolate gaps
    t1 = time.time()
    if fill_gaps:
        ds_interpolated = xr.apply_ufunc(interpolate_small_gaps, ds['x_raw'], ds['y_raw'], ds['missing'], input_core_dims=[['frame'], ['frame'], ['frame']], output_core_dims=[['frame'], ['frame'], ['frame']], 
                                         vectorize=True, kwargs=interp_dict, dask='allowed', output_dtypes=[float, float, float])

        ds['x_raw'], ds['y_raw'], ds['missing'] = ds_interpolated # 'missing' value for interpolated points is 2
    t2 = time.time()
    print('Time to interpolate gaps:', round(t2 - t1, 3))

    # Compute speed
    ds = compute_speed(ds, speed_dict)
    t3 = time.time()
    print('Time to compute speed:', round(t3 - t2, 3))

    # Exclude borders
    if center_only:
        ds = exclude_borders(ds, radius)
    t4 = time.time()
    print('Time to exclude borders:', round(t4 - t3, 3))

    # Compute tracklet lengths and assign them IDs
    tracklet_lengths, tracklet_ids = xr.apply_ufunc(compute_tracklet_lengths_and_ids, ds['missing'], input_core_dims=[['frame']], output_core_dims=[['frame'], ['frame']], kwargs={'fill_gaps': fill_gaps}, vectorize=True, dask='allowed', output_dtypes=[float, float])

    ds['tracklet_length'] = tracklet_lengths
    ds['tracklet_id'] = tracklet_ids

    t5 = time.time()
    print('Time to compute tracklet lengths and IDs:', round(t5 - t4, 3))

    # Compute orientations and angular speed
    ds = compute_theta(ds, speed_dict)
    t6 = time.time()
    print('Time to compute orientations and angular speed:', round(t6 - t5, 3))

    # Compute distance from center
    ds = compute_dist_from_center(ds)
    t7 = time.time()
    print('Time to compute distance from center:', round(t7 - t6, 3))
    return ds

def load_and_preprocess(batch_num:int, exp_name:str, speed_dict:dict, fill_gaps:bool, interp_dict:dict, center_only:bool, radius:int, load_num_ids:int = None):
    ''' Load raw data for a batch number and preprocess it.'''
    num_ids, ds_raw = load_trex_data(batch_num, exp_name, load_num_ids)
    print('TRex data loaded. Number of IDs:', num_ids)

    ds = preprocess_raw(ds_raw, speed_dict, fill_gaps=fill_gaps, interp_dict=interp_dict, center_only=center_only, radius=radius)
    print('Data pre-processed.')

    return ds, num_ids

def preprocess_save_all_batches(exp_name:str, num_batches:int, speed_dict:dict, fill_gaps:bool, interp_dict:dict, center_only:bool, radius:int, reprocess:bool = False):
    ''' Preprocess all batches and save them.'''

    for batch_i in range(num_batches):
        # Define folder path
        folder_path = f'/output/preprocessed/{exp_name}/batch_{batch_i}'
        
        # Check if folder exists, create if not
        os.makedirs(folder_path, exist_ok=True)

        # Check if preprocessed data file exists
        data_file_path = f'{folder_path}/traj_data.h5'
        exists = os.path.exists(data_file_path)

        if not exists or reprocess:
            print(f'Preprocessing data for batch {batch_i}.')

            ds, _ = load_and_preprocess(batch_i, exp_name, speed_dict, fill_gaps, interp_dict, center_only, radius)

            # SAVE DATA
            params = {'speed_dict': speed_dict, 'fill_gaps': fill_gaps, 'interp_dict': interp_dict, 'center_only': center_only, 'radius': radius}
            save_ds(ds, data_file_path, params)
            print(f'Data for batch {batch_i} saved.\n')

            del ds

# Try to isolate discontinuities by looking at angular speed (to make sure tracklets are actually just one individual)