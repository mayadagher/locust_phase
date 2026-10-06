'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import scipy
import time

from data_handling import save_ds, load_preprocessed_data

'''_____________________________________________________COMPUTATION FUNCTIONS____________________________________________________________'''

def interpolate_small_gaps(x:np.ndarray, y:np.ndarray, missing:np.ndarray, max_gap:int=1, max_dist_m:float=0.02):
    """
    Conditionally interpolate NaN gaps in x,y trajectories.
    Only interpolate if the gap is shorter than `max_gap`
    and the Euclidean distance across the gap is less than `max_dist_m`.
    """

    # Load data
    x = np.array(x, copy=True)
    y = np.array(y, copy=True)
    missing = np.array(missing, copy = True)
    isnan = np.isnan(x) | np.isnan(y)

    if not np.any(isnan):
        return x, y, missing  # Nothing to do

    # Find start and end indices of NaN runs
    diffs = np.diff(np.concatenate([[0], isnan.astype(int), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    s_idx = 0
    for s, e in zip(starts, ends):
        s_idx += 1
        gap_size = e - s
        
        # Skip if at boundary
        if s == 0 or e >= len(x):
            continue

        # Compute distance across the gap
        dx = x[e] - x[s-1]
        dy = y[e] - y[s-1]
        dist_gap = np.sqrt(dx**2 + dy**2)

        if gap_size <= max_gap and dist_gap <= max_dist_m:

            # Linearly interpolate across this small gap
            x[s:e] = np.linspace(x[s-1], x[e], gap_size + 2)[1:-1]
            y[s:e] = np.linspace(y[s-1], y[e], gap_size + 2)[1:-1]
            missing[s:e] = 0 # Value no longer missing

    return x, y, missing

def spline_scipy(x:np.ndarray, degree:int=2, s:float=0.5):
    '''Smooth using the scipy spline method. x is strictly contiguous.'''

    t = np.arange(len(x))

    spline = scipy.interpolate.UnivariateSpline(t, x, k=degree, s=s*len(x)) # s is made to be proportional to tracklet length
    x_hat = spline(t)

    return x_hat

def smooth_nantolerant(x: xr.DataArray, func, params: dict, min_tracklet_length: int):
    """Apply a smoothing function to a 1D array (frame axis) while tolerating NaNs."""

    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    valid = np.isfinite(x)

    # Find contiguous finite segments
    edges = np.diff(valid.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1

    # Add beginning/end if they too are valid
    if valid[0]:
        starts = np.r_[0, starts]
    if valid[-1]:
        ends = np.r_[ends, len(x)]

    # Fit each contiguous segment separately
    for s, e in zip(starts, ends):
        seg = x[s:e]
        n = len(seg)
        if n < min_tracklet_length:  # Too short for interpretation: exclude tracklet
            out[s:e] = np.nan
            continue

        out[s:e] = func(seg, **params)

    return out

def compute_speed(ds:xr.Dataset, smooth_dict:dict, fs: float = 5):
    '''Compute speed using methods specified by speed_types (a dictionary with speed type as keys and parameter dictionaries as values). 
    fs is sample frequency and is used to compute minimum tracklet length and to convert units to /s from /frame.'''
    speed_types = list(smooth_dict.keys())

    # Make this function more general
    if 'x_raw' not in ds:
        x = ds['x']
        y = ds['y']
    else:            
        x = ds['x_raw']
        y = ds['y_raw']

    # Frame-coordinate spacing, expressed in seconds.
    dt = ds['frame'].diff('frame') / fs

    def speed_from_coords(x_coord, y_coord):
        dx = x_coord.diff('frame')
        dy = y_coord.diff('frame')
        return np.hypot(dx, dy) / dt

    if 'raw' in speed_types: # Compute instantaneous speed from raw positions
        ds['v_raw'] = speed_from_coords(x, y)


    if 'spline' in speed_types: # Compute spline smoothed speed
        print('Fitting spline')
        spline_params = smooth_dict['spline']
        print(spline_params)
        # min_tracklet_length = 10*(spline_params['degree'] + 1)
        # min_tracklet_length = 3*(spline_params['degree'] + 1)
        min_tracklet_length = spline_params['degree'] + 1 # Minimum tracklet length for spline smoothing is degree + 1

        ds['x_spline'] = xr.apply_ufunc(smooth_nantolerant, x, input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs={'func': spline_scipy, 'params': spline_params, 'min_tracklet_length': min_tracklet_length}, dask='parallelized')
        ds['y_spline'] = xr.apply_ufunc(smooth_nantolerant, y, input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs={'func': spline_scipy, 'params': spline_params, 'min_tracklet_length': min_tracklet_length}, dask='parallelized')

        # Speed magnitude
        ds['v_spline'] = speed_from_coords(ds['x_spline'], ds['y_spline'])

    return ds

def impose_borders(ds:xr.Dataset, arena_center_m:np.ndarray, arena_radius_m:float): 
    '''Sets missing = 1 for individuals outside a circular region and nan for all other variables'''
    ds_copy = ds.copy()
    mask = (ds_copy['x_raw'] - arena_center_m[0])**2 + (ds_copy['y_raw'] - arena_center_m[1])**2 > arena_radius_m**2 # Mask out individuals outside of the arena

    # Set all data variables to np.nan where mask is True
    ds_copy = xr.where(mask, np.nan, ds_copy)

    # Set missing = 1 where mask is True
    ds_copy['missing'] = xr.where(mask, 1, ds_copy['missing'])

    return ds_copy

def compute_tracklet_lengths(missing_1d):
    """
    Compute per-frame tracklet lengths for a 1D boolean array indicating missing values.

    Parameters
    ----------
    missing_1d : array-like of bool or nan
        True = missing, False = valid (or NaN treated as missing)

    Returns
    -------
    lengths : np.ndarray
        Array of same shape as input; contains the total length of each
        contiguous valid segment, NaN where missing.
    """

    # Prepare valid array
    missing_1d = np.array(missing_1d, copy = True)
    missing_1d[np.isnan(missing_1d)] = 1  # Treat NaNs as missing
    present = ~missing_1d.astype(bool)

    n = len(present)
    lengths = np.full(n, np.nan)

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

    # Fill in lengths where valid
    lengths[present] = seg_len_map[seg_id[present]]

    # Convert missing entries back to NaN
    lengths[~present] = np.nan

    return lengths

def preprocess_ds(ds:xr.Dataset, smooth_dict:dict, fill_gaps:bool, interp_dict:dict | None = None, center_only:bool = True, arena_center_m:np.ndarray | None = None, arena_radius_m:float | None = None, fs:float = 5):
    '''Preprocess a (dewarped) dataset.'''

    # STEP 1: Exclude detections outside of the arena
    t1 = time.time()
    if center_only:

        ds = impose_borders(ds, arena_center_m, arena_radius_m)

        t2 = time.time()
        print(f'Detections outside of the arena excluded in {(t2 - t1):.2f} s.')
        t1 = t2

    # STEP 2: Interpolate gaps
    
    if fill_gaps:
        interpolated_arrs = xr.apply_ufunc(interpolate_small_gaps, ds['x_raw'], ds['y_raw'], ds['missing'], input_core_dims=[['frame'], ['frame'], ['frame']], 
                                           output_core_dims=[['frame'], ['frame'], ['frame']], vectorize=True, kwargs=interp_dict, dask='allowed', output_dtypes=[float, float, float])

        ds['x_raw'], ds['y_raw'], ds['missing'] = interpolated_arrs
    
        t2 = time.time()
        print(f'Tracklets interpolated in {(t2 - t1):.2f} s.')
        t1 = t2

    # STEP 3: Compute velocities

    ds = compute_speed(ds, smooth_dict, fs)

    t2 = time.time()
    print(f'Speed computed in {(t2 - t1):.2f} s.')
    t1 = t2

    # STEP 4: Compute tracklet lengths

    tracklet_lengths = xr.apply_ufunc(compute_tracklet_lengths, ds['missing'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, dask='allowed', output_dtypes=[float])   
    ds['tracklet_length'] = tracklet_lengths

    t2 = time.time()
    print(f'Tracklet lengths computed in {(t2 - t1):.2f} s.')

    return ds

def preprocess_and_save_all_batches(n_batches:int, h5_dir:str, smooth_dict:dict, fill_gaps:bool, interp_dict:dict | None = None, center_only:bool = True, arena_center_m:np.ndarray | None = None, arena_radius_m:float | None = None, fs:float = 5):

    # Iterate over batches
    for i in range(n_batches):

        # Load unprocessed .h5 file
        ds_name = f'{h5_dir}/batch_{i}_5.0Hz'
        ds = load_preprocessed_data(ds_name + '.hdf5')

        # Preprocess ds
        ds = preprocess_ds(ds, smooth_dict, fill_gaps, interp_dict, center_only, arena_center_m, arena_radius_m, fs)

        # Save ds
        params = {'smooth_dict': smooth_dict, 'fill_gaps': fill_gaps, 'interp_dict': interp_dict, 'center_only': center_only, 'radius': arena_radius_m, 'speed_units': 'm/s', 'done_by': 'maya_dagher'}
        save_ds(ds, ds_name, params)

        # Manage memory
        del ds