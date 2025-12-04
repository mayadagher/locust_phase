'''_____________________________________________________IMPORTS____________________________________________________________'''

from matplotlib.ticker import FixedLocator
import numpy as np
import math
import xarray as xr
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import seaborn as sns
import pynumdiff
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import scipy.stats as st
import time

'''_____________________________________________________COMPUTATION FUNCTIONS____________________________________________________________'''

# def check_trex_speeds(ds, plot_speeds = False): # Try to understand speed distribution from TRex

#     if plot_speeds:
#         # Look at histograms of speeds to look for instances of id jumping
#         speed_vals = ds['trex_speed'].values.ravel()
#         valid_speeds = speed_vals[np.isfinite(speed_vals)]

#         sns.histplot(valid_speeds)
#         plt.axvline(x=100, color='k', linestyle='--') # Plotting a line when speed is 100 because that is the cut-off in TRex settings for keeping identity (allegedly)
#         plt.yscale('log')
#         plt.xlabel('Speed')
#         plt.ylabel('Frequency')
#         plt.savefig(f'plots/{system}/batch_{batch_num}/trex_speed_histogram.png') # Anything over 100 is suspect, but transition in not as harsh as one would expect

#     # Check if speed is always nan when detection is missing (answer: YES)
#     print(ds['missing'].sum())
#     print(ds['trex_speed'].isnull().sum())
#     print((ds['missing'].astype(bool) & ds['trex_speed'].isnull()).sum())

#     # Why, then, is speed sometimes so large? Check to see if these follow or precede missing values

#     mask = ds['trex_speed'] > 100

#     # Shift mask by offset to see if 'missing' values in the past or future cause high speed instances
#     offset = 3 # +: values forward in time, -: values backward in time
#     shifted_mask = mask.shift(frame = offset).fillna(False)

#     for id in ds['id'].values:
#         # Look at whether different 'missing' values correlate with high speed instances
#         num_high_speeds = shifted_mask.sel(id = id).sum().values
#         num_missings = (ds.sel(id = id, frame = shifted_mask.sel(id = id))['missing'] == 1).sum().values
#         print('ID', id, ', high speed counts:', num_high_speeds, ', missing counts:', num_missings)

#     # Hm, it seems like the high speeds aren't occurring at any fixed frame from the missing values. What could be causing it?
#     # I also checked and these high speeds are not very close together (not consecutive).

#     # Let's calculate our own speeds

#     # Look at times when speed is above 100 and plot the before and after position... ideally, there would be some evidence of pairs of IDs switching, maybe?

def interpolate_small_gaps(x, y, missing, max_gap=3, max_dist=10):
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

    for s, e in zip(starts, ends):
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

def sg_derivative(x, window_length=7, polyorder=2, deriv=0, delta=1.0):
    """NaN-tolerant Savitzky-Golay derivative for 1D array (frame axis)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    notnan = ~np.isnan(x)

    # find contiguous segments of not-nan
    edges = np.diff(notnan.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if notnan[0]:
        starts = np.r_[0, starts]
    if notnan[-1]:
        ends = np.r_[ends, len(x)]

    for s, e in zip(starts, ends):
        seg = x[s:e]
        n = len(seg)
        if n < 3:  # too short: fallback to finite difference or copy
            if n == 1:
                out[s:e] = np.nan
            else:
                # central diff for n==2
                out[s:e] = np.gradient(seg, delta)
            continue
        wl = min(window_length, n if (n % 2 == 1) else n - 1)
        if wl < polyorder + 2:
            wl = polyorder + 2
            if wl % 2 == 0:
                wl += 1
            if wl > n:
                wl = n if n % 2 == 1 else n - 1
        try:
            out[s:e] = savgol_filter(seg, window_length=wl, polyorder=polyorder, deriv=deriv, delta=delta, mode = 'interp')
        except Exception:
            # fallback: numerical gradient
            print('Error using savgol_filter, using np.gradient instead.')
            out[s:e] = np.gradient(seg, delta)

    return out

def compute_speed(ds, speed_dict):
    '''Compute speed using methods specified by speed_types (a dictionary with speed type as keys and parameter dictionaries as values).'''
    speed_types = list(speed_dict.keys())

    if 'raw' in speed_types or 'moving_avg' in speed_types or 'moving_med' in speed_types: # Compute instantaneous speed from raw positions
        ds['vx_raw'], ds['vy_raw'] = ds['x_raw'].diff(dim = 'frame'), ds['y_raw'].diff(dim = 'frame')
        ds['v_raw'] = np.hypot(ds['vx_raw'], ds['vy_raw']) / ds['frame'].diff(dim = 'frame')

    if 'high_ord' in speed_types: # Compute high-order derivatives
        high_ord_params = speed_dict['high_ord']
        ds['x_high_ord'], ds['vx_high_ord'] = xr.apply_ufunc(pynumdiff.finite_difference.finitediff, ds['x_raw'], input_core_dims=[['frame']], output_core_dims=[['frame'],['frame']], vectorize = True,  kwargs=high_ord_params, dask = 'parallelized')
        ds['y_high_ord'], ds['vy_high_ord'] = xr.apply_ufunc(pynumdiff.finite_difference.finitediff, ds['y_raw'], input_core_dims=[['frame']], output_core_dims=[['frame'],['frame']], vectorize = True,  kwargs=high_ord_params, dask = 'parallelized')
        
        # Get speed magnitude
        ds['v_high_ord'] = np.hypot(ds['vx_high_ord'], ds['vy_high_ord'])

    if 'moving_avg' in speed_types: # Compute moving average speed
        moving_params = speed_dict['moving']
        ds['v_moving_avg'] = ds['v_raw'].rolling(frame=moving_params['window_length'], center=moving_params['center'], min_periods = 1).mean()

    if 'moving_med' in speed_types: # Compute moving median speed
        moving_params = speed_dict['moving_med']
        ds['v_moving_med'] = ds['v_raw'].rolling(frame=moving_params['window_length'], center=moving_params['center'], min_periods = 1).median()

    if 'sg' in speed_types: # Compute Savitzky-Golay smoothed speed
        sg_params = speed_dict['sg']
        # Ensure deriv key is 0 at first, for first order sg calculation
        sg_params['deriv'] = 0

        # Smooth x and y (vectorized over ids)
        ds['x_sg'] = xr.apply_ufunc(sg_derivative, ds['x_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs=sg_params, dask = 'parallelized')
        ds['y_sg'] = xr.apply_ufunc(sg_derivative, ds['y_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs=sg_params, dask = 'parallelized')

        # Get velocity
        sg_params['deriv'] = 1
        ds['vx_sg'] = xr.apply_ufunc(sg_derivative, ds['x_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs = sg_params, dask = 'parallelized')
        ds['vy_sg'] = xr.apply_ufunc(sg_derivative, ds['y_raw'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, kwargs = sg_params, dask = 'parallelized')

        # Speed magnitude
        ds['v_sg'] = np.hypot(ds['vx_sg'], ds['vy_sg'])

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

# def sg_circular(ds, window_length=7, polyorder=2, deriv=0, delta=1.0):
#     """
#     Apply NaN-tolerant Savitzky–Golay smoothing to circular data (like orientation).

#     Parameters
#     ----------
#     ds : xarray.Dataset
#         Dataset containing `var_name` to smooth.
#     var_name : str
#         Name of the variable to smooth (e.g. 'orientation' or 'ang_speed').
#     window_length : int
#         Length of the filter window (must be odd).
#     polyorder : int
#         Polynomial order for the Savitzky–Golay filter.
#     frame_dim : str
#         Name of the frame dimension along which to apply smoothing.

#     Returns
#     -------
#     smoothed : xarray.DataArray
#         Smoothed variable with same shape and coords as input.
#     """

#     def smooth_1d(angle_1d):
#         """Apply NaN-safe Savitzky–Golay smoothing to 1D circular data."""
#         # Handle NaNs: create a mask
#         mask = np.isnan(angle_1d)
#         if np.all(mask):
#             return angle_1d  # all NaNs → unchanged

#         # Temporarily unwrap angles to remove jumps at ±π
#         unwrapped = np.unwrap(angle_1d[~mask])

#         # Apply Savitzky–Golay filter on valid data only
#         smoothed = savgol_filter(unwrapped, window_length = min(window_length, unwrapped.size - (1 - (unwrapped.size % 2))), polyorder=min(polyorder, unwrapped.size - 1), deriv = deriv, delta = delta)

#         # Re-wrap to [-π, π]
#         smoothed_wrapped = (smoothed + np.pi) % (2 * np.pi) - np.pi

#         # Reinsert NaNs
#         result = np.full_like(angle_1d, np.nan, dtype=float)
#         result[~mask] = smoothed_wrapped
#         return result

#     # Apply vectorized across all non-frame dims
#     smoothed = xr.apply_ufunc(smooth_1d, ds['raw_orientation'], input_core_dims=[['frame']], output_core_dims=[['frame']], vectorize=True, dask='allowed', output_dtypes=[float])

#     return smoothed

def compute_theta(ds, speed_dict): # Speeds should be pre-computed
    ''' Compute heading direction and instantaneous change in heading direction.'''
    speed_types = list(speed_dict.keys())

    # Compute (smoothed) orientations
    ds['theta_raw'] = np.arctan2(ds['vy_raw'], ds['vx_raw']) # Compute theta from high ord data

    smth_fns = {'high_ord': pynumdiff.finite_difference.finitediff, 'sg': savgol_filter}
    for speed in speed_types:
        if speed in ['high_ord', 'sg']:
            if speed == 'sg':
                speed_dict[speed]['deriv'] = 1 # Ensure first difference is taken
            ds[f'vtheta_{speed}'] = smooth_circular(ds, smth_fns[speed], speed_dict[speed])

        else:
            print(f'Warning: {speed} not yet implemented for angular speed.')

    # ds['theta_sg'] = sg_circular(ds, speed_dict['window_length'], speed_dict['polyorder'], 0, speed_dict['delta']) # Smoothed orientations

    # # Compute (smoothed) angular speed
    # ds['vtheta_sg'] = sg_circular(ds, speed_dict['window_length'], speed_dict['polyorder'], 1, speed_dict['delta']) # Smoothed angular speed

    return ds

def compute_dist_from_center(ds, center=(1920/2, 1920/2)):
    ''' Compute distance from center for each individual at each frame using high_ord smoothed positions.'''
    ds_copy = ds.copy()
    ds_copy['dist_from_center'] = np.hypot(ds_copy['y_high_ord'] - center[1], ds_copy['x_high_ord'] - center[0])
    return ds_copy

def preprocess_data(ds, speed_dict, fill_gaps = False, interp_dict = None, center_only = False, radius = None):
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

# More piecing together? Across identities? Comp. expensive, maybe covered by just training a better model.
# Try to isolate discontinuities by looking at angular speed (to make sure tracklets are actually just one individual)

'''_____________________________________________________ANIMATION FUNCTIONS____________________________________________________________'''
def animate_trajs_lined(ds, video_path: str, buffer = 150, start_frame=0, end_frame=-1, interval=50, trail=10):
    """
    Animate trajectories from an xarray.Dataset over a subsection of a video. buffer specifies the size of the window to use for the animation.
    start_frame and end_frame should be relative to the actual start of the batch (since subsequent batches won't start at 0).
    """

    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Manually find start and end frame in dataset if not specified
    first_frame = int(ds.frame.min())
    last_frame = int(ds.frame.max())
    num_frames = last_frame - first_frame + 1

    if end_frame < 0:
        end_frame = num_frames

    # Check appropriate start and end frame inputs
    assert start_frame < num_frames, f"Start frame is too large for the number of frames for this batch, which is {num_frames}."
    assert end_frame <= num_frames, f"End frame is too large for the number of frames for this batch, which is {num_frames}."
    assert start_frame < end_frame, f"Start frame must be less than end frame."

    # Define frames in animation (consecutive)
    start_frame += first_frame
    end_frame += first_frame
    frames = np.arange(start_frame, end_frame)

    # Define ids
    ids = ds.id.values

    # Subset dataset
    ds_sub = ds.sel(frame = slice(start_frame, end_frame - 1))

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_axis_off()

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first video frame.")
    img_artist = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    lines = {i: ax.plot([], [], '-', lw=1)[0] for i in ids}
    for line in lines.values():
        line.set_zorder(2)
    img_artist.set_zorder(1)

    ax.set_xlim([1920/2 - buffer, 1920/2 + buffer])
    ax.set_ylim([1920/2 - buffer, 1920/2 + buffer])

    def init():
        for line in lines.values():
            line.set_data([], [])
        return [img_artist, *lines.values()]

    def update(idx):
        frame_num = frames[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return [img_artist, *lines.values()]
        img_artist.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        recent = frames[max(0, idx - trail):idx + 1]
        for i in ids:
            ds_recent = ds_sub.sel(id=i).sel(frame=recent)
            lines[i].set_data(ds_recent.x_high_ord, ds_recent.y_high_ord)
        return [img_artist, *lines.values()]

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(f'plots/{system}/batch_{batch_num}/locust_tracks_lined.gif')
    cap.release()
    return

def animate_trajs_coloured(ds, video_path: str, colours: xr.DataArray, cbar_name: str, start_frame=0, end_frame=-1, interval=50):
    """
    Scatter points with colours from 'colours' DataArray over video frames. Colours should be per-id, per-frame. Colours elements are assumed to be scalars, not tuples.
    start_frame and end_frame should be relative to the actual start of the batch (since subsequent batches won't start at 0).
    """

    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Manually find start and end frame in dataset if not specified
    first_frame = int(ds.frame.min())
    last_frame = int(ds.frame.max())
    num_frames = last_frame - first_frame + 1

    if end_frame < 0:
        end_frame = num_frames

    # Check appropriate start and end frame inputs
    assert start_frame < num_frames, f"Start frame is too large for the number of frames for this batch, which is {num_frames}."
    assert end_frame <= num_frames, f"End frame is too large for the number of frames for this batch, which is {num_frames}."
    assert start_frame < end_frame, f"Start frame must be less than end frame."

    # Define frames in animation (consecutive)
    start_frame += first_frame
    end_frame += first_frame
    frames = np.arange(start_frame, end_frame)

    # Subset dataset
    ds_sub = ds.sel(frame = slice(start_frame, end_frame - 1))
    colours_sub = colours.sel(frame = slice(start_frame, end_frame - 1))

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_axis_off()

    # Initialize scatter plot artist outside update function
    valid_colours = colours_sub.values[(ds_sub['missing'].values != 1) & ~np.isnan(colours_sub.values)] # Excludes points due to boundaries and due to other third axis specific filtering
    vmax = float(np.nanmax(valid_colours))
    vmin = float(np.nanmin(valid_colours))
    norm = plt.Normalize(vmin, vmax)
    scat = ax.scatter([], [], c=[], cmap = 'viridis', s=0.5, norm=norm)
    scat.set_zorder(2)

    cbar = fig.colorbar(scat, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_ticks(np.linspace(vmin, vmax, 7))
    cbar.set_label(cbar_name)

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first video frame.")
    img_artist = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_artist.set_zorder(1)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return [img_artist, scat]

    def update(idx):
        frame_num = frames[idx]

        # Show frame of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return [img_artist, scat]
        img_artist.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Get positions and tracklet lengths for current frame
        try:
            frame_data = ds_sub.sel(frame=frame_num)
            frame_colours = colours_sub.sel(frame=frame_num).values

            x_vals = frame_data['x_high_ord'].values
            y_vals = frame_data['y_high_ord'].values
            
            # Remove NaN values or those excluded from outside the arena
            valid = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isnan(frame_colours) | frame_data['missing'].values.astype(bool))
            
            # Update scatter plot
            scat.set_offsets(np.c_[x_vals[valid], y_vals[valid]])
            scat.set_array(frame_colours[valid])

        except KeyError:
            # Frame not in dataset
            scat.set_offsets(np.empty((0, 2)))

        if idx % 100 == 0:
            print(f'Processed frame {idx+1}/{len(frames)}')
        
        return [img_artist, scat]

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(f'plots/{system}/batch_{batch_num}/locust_tracks_coloured_{'_'.join(cbar_name.lower().split(' '))}.gif')
    cap.release()
    return

'''_____________________________________________________PLOT FUNCTIONS____________________________________________________________'''
def plot_tracks(ds, t_slice):

    plt.figure(figsize=(10, 10))
    for i in ds['id'].values:
        plt.plot(ds['x_raw'].sel(id=i, frame=t_slice), ds['y_raw'].sel(id=i, frame=t_slice), marker='.', label=f'ID {i}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Locust tracks')
    if len(ds['id'].values) <= 10:
        plt.legend()
    plt.savefig(f'plots/{system}/batch_{batch_num}/original_tracks.png')

def plot_smoothed_coords(ds, id: int, speed_names: list, t_slice): # Takes preprocessed xarray.Dataset as input that has raw and sg computed
    # Look at raw versus smooth x and y data for a single id
    _, axs = plt.subplots(len(speed_names)+2, 1, figsize=(26, 12))
    coords = ['x', 'y']
    # print(ds['v_raw'].isel(id=id, frame=t_slice).values)
    # print(ds['v_sg'].isel(id=id, frame=t_slice).values)

    for i in range(2):
        axs[i].plot(ds['frame'].isel(frame=t_slice), ds[coords[i]].isel(id=id, frame=t_slice), label='Original')
        axs[i].plot(ds['frame'].isel(frame=t_slice), ds[coords[i]+'_high_ord'].isel(id=id, frame=t_slice), label='Smoothed', linestyle='--')
        axs[i].set_xlabel('Frame', fontsize = 17)
        axs[i].set_ylabel(coords[i], fontsize = 17)
        # if i !=2:
        #     axs[i].set_ylim(1200, 1500)

    for i, name in enumerate(speed_names):
        axs[i + 2].plot(ds['frame'].isel(frame=t_slice), ds['v_raw'].isel(id=id, frame=t_slice), label = 'Original')
        axs[i + 2].plot(ds['frame'].isel(frame=t_slice), ds['v_' + name].isel(id=id, frame=t_slice), label = 'Smoothed', linestyle = '--')
        axs[i + 2].set_xlabel('Frame', fontsize = 17)
        axs[i + 2].set_ylabel(name.capitalize(), fontsize = 17)
    axs[-1].legend()

    plt.tight_layout()
    plt.savefig(f'plots/{system}/batch_{batch_num}/smoothed_speeds.png')

def fit_mixture_model(data, ax=None, opt = True):
    '''
    Fit two distributions to speed data.

    Parameters
    ----------
    data : array-like
        1D array of speed values (must be non-negative integers or will be rounded)
    ax : matplotlib axis, optional
        Axis to plot on. If None, no plot is created.
    '''

    def exp_pdf(x, r): # func1
        '''PDF of Exponential distribution.'''
        return r * np.exp(-r * x)
    
    def delta_pdf(x, _): # func1
        '''PDF of Delta distribution at zero.'''
        return np.where((x >= bin_edges[0]) & (x < bin_edges[1]), 1.0, 0.0)

    def gamma_pdf(x, alpha, lam, _): # func2
        '''PDF of Gamma distribution.'''
        return (x**(alpha - 1) * np.exp(-x / lam)) / (lam**alpha * math.gamma(alpha))
    
    def lognorm_pdf(x, mu, sigma, _): # func2
        '''PDF of Lognormal distribution.'''
        return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2 * sigma**2))
    
    def weibull_pdf(x, k, lam, _): # func2
        '''PDF of Weibull distribution.'''
        return (k / lam) * (x / lam)**(k - 1) * np.exp(-(x / lam)**k)
    
    def logt_pdf(x, nu, mu, sigma): # func2
        '''PDF of Log-Student's t distribution.'''
        y = np.log(x)
        return (1.0 / x) * st.t.pdf(y, df=nu, loc=mu, scale=sigma)

    def mixture_pdf(x, p11, p21, p22, p23, weight):
        '''PDF of mixture of two distributions.'''
        return weight * norm * func1(x, p11) + (1 - weight) * norm * func2(x, p21, p22, p23)
    
    func1 = exp_pdf
    func2 = logt_pdf

    # Initial parameter guesses
    p0 = np.array([25, 100, 1e-15, 1.2, 0.15])  # r, p21, p22, p23, weight

    # Fit histogram with this gamma mixture pdf
    bins = np.linspace(0, np.max(data), 400)
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_width = np.diff(bin_edges)[0]
    bin_centers = bin_edges[:-1] + bin_width / 2
    norm = 1/(len(bin_centers)*bin_width) # Ensure distribution sums to 1

    # Normalize counts to create a probability distribution for fitting
    normalized_counts = counts / counts.sum()

    if opt:
        try:
            p0, _ = curve_fit(
                mixture_pdf,
                bin_centers,
                normalized_counts,
                p0=p0,
                bounds=((len(p0) - 1)*[0] + [0.14], (len(p0) - 1)*[1000] + [1]),
                maxfev=10000
            )
            print('Fitted parameters:', p0)

        except RuntimeError:
            # Fallback: return initial guess if fitting fails
            print("Warning: Zero-inflated fitting did not converge. Using initial guess.")

    else:
        print('Using initial parameters:', p0)

    # Plot if axis provided
    if ax is not None:
        # Plot histogram
        ax.bar(bin_centers, normalized_counts, width = bin_width, label='Data', color='steelblue')

        # Plot fitted mixture
        x_plot = np.linspace(bin_centers[0], bin_centers[-1], 800)

        pdf_mix = mixture_pdf(x_plot, *p0)
        ax.plot(x_plot, pdf_mix, 'r-', linewidth=2, label='Mixture')

        # ax.set_xscale('log')
        ax.set_xlabel('Speed (rounded)')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.set_title(f'{func1.__name__.replace("_pdf","").capitalize()} + {func2.__name__.replace("_pdf","").capitalize()} mixture fit')

        return

def plot_speed_hists(ds, speed_names: list, fit_speed: bool = False): # Takes preprocessed xarray.Dataset as input

    # Look at all speeds for each ID
    _, axs = plt.subplots(len(speed_names), 1, figsize=(26, 12), sharey = True, sharex = True)
    
    # Ensure axs is always a list for consistent indexing
    if len(speed_names) == 1:
        axs = [axs]     

    for i in range(len(speed_names)):
        speed_vals = ds[f'v_{speed_names[i]}'].values.ravel()
        valid_speeds = speed_vals[np.isfinite(speed_vals)]
        
        # Fit mixed distributions if requested
        if fit_speed:
            fit_mixture_model(valid_speeds, ax=axs[i], opt = False)
        else:
            sns.histplot(valid_speeds, ax=axs[i], bins=500)
            axs[i].set_title(f'{speed_names[i].capitalize()} speed')

        axs[i].axvline(x=20, color='k', linestyle='--') # Plotting a line when speed is 20 because that is the cut-off in TRex settings for keeping identity (in px/frame)
        axs[i].axvline(x=threshes[batch_num], color='k', linestyle='--') # Plotting a line at the activity threshold for this batch
    
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Speed')
    plt.ylabel('Frequency')
    plt.tight_layout()

    plt.savefig(f'plots/{system}/batch_{batch_num}/speed_histograms.png')

def plot_tracklet_lengths_hist(ds_raw, speed_dict: dict, interp_dict: dict, radius: float, n_bins = 15):

    # Initiate plot
    _, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (8, 5))
    labels = [f'None', f'{round(radius)}']

    # Iterate over no interpolation, with interpolation
    for i in range(2):
        lengths_list = []
        # Iterate over all data, centered
        for j in range(2):
            print(2*i + j + 1)
            ds = preprocess_data(ds_raw, speed_dict, fill_gaps = bool(i), interp_dict = interp_dict, center_only = bool(j), radius = radius)

            # Broadcast id coordinate to the same shape as tracklet_id
            ids = xr.broadcast(ds['id'], ds['tracklet_id'])[0].values.ravel()  # shape (id, frame)

            # Compute tracklet lengths
            tids = ds['tracklet_id'].values.ravel()

            mask = ~np.isnan(tids) # Make mask of non nan tracklet ids
            valid_ids = ids[mask].astype(int)
            valid_tids = tids[mask].astype(int)

            # Count length of tracklets
            counts = np.bincount(np.ravel_multi_index((valid_ids, valid_tids), (int(valid_ids.max()+1), int(valid_tids.max()+1))))
            lengths_list.append(counts)

        max_length = np.max([np.max(lengths) for lengths in lengths_list])
        bins = np.logspace(0, np.log10(max_length), n_bins)

        for k, lengths in enumerate(lengths_list):
            sns.histplot(lengths, ax=ax[i,0], bins=bins, label=labels[k])
            
        counts1, _ = np.histogram(lengths_list[0], bins = bins)
        counts2, _ = np.histogram(lengths_list[1], bins = bins)
        widths = np.diff(bins)
        bar_width = np.median(widths / bins[:-1]) * bins[:-1]  # fraction of local bins
        ax[i,1].bar(bins[:-1], counts1 - counts2, color = 'gray', width = bar_width, align = 'edge')

        ax[i,0].set_ylabel('Count', fontsize = 15)
        ax[i,1].set_ylabel('Excluded', fontsize = 15)

        for k in range(2):
            ax[i,k].set_title('Interpolated' if i == 1 else 'Not interpolated', fontsize = 15)
            if i:
                ax[i,k].set_xlabel('Tracklet length (frames)', fontsize = 13)
            if not k:
                ax[i,k].legend(title = 'Radius')

    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'plots/{system}/batch_{batch_num}/tracklet_length_hists/rad_{str(int(radius))}.png')

def plot_ang_speed(ds, t_slice):
    # Look at angular speeds over time for each ID

    _, axs = plt.subplots(2, 1, figsize=(26, 12), sharey = True)
    column_names = ['vtheta_high_ord', 'vtheta_sg']

    for id in range(2):
        for i in range(len(column_names)):
            fitted_speed = ds[f'{column_names[i]}'].isel(id=id, frame = t_slice)
            axs[i].plot(fitted_speed['frame'], fitted_speed, label=f'ID {id}')

    for i in range(len(column_names)):
        axs[i].set_xlabel('Frame')
        axs[i].set_ylabel(f'{column_names[i].capitalize()}')
        if len(ds['id'].values) < 10:
            axs[i].legend()
    plt.tight_layout()
    plt.savefig(f'plots/{system}/batch_{batch_num}/angular_speed_over_time.png')

def plot_num_tracklets_over_time(ds):
    # Look at number of tracklets in any given instance to see how many individuals are missing. This assumes all are tracked at one point.

    perc_tracklets = np.mean(ds['missing'] != 1, axis = 0) # Includes interpolated points if they exist

    plt.figure(figsize=(12, 6))
    plt.plot(perc_tracklets['frame'], perc_tracklets)
    plt.xlabel('Frame')
    plt.ylabel('Individuals tracked (%)') # Assumes TReX at one point tracked all individuals
    plt.title('Tracked Individuals Over Time')
    plt.savefig(f'plots/{system}/batch_{batch_num}/num_tracklets_over_time.png')

'''_____________________________________________________LOAD AND SAVE FUNCTIONS____________________________________________________________'''
def load_trex_data(batch_num, file_name, load_num_ids=None):
    """
    Load TReX .npz data into an xarray.Dataset.
    
    Dimensions: id × frame
    Coordinates: 'id', 'frame'
    Data variables: x_raw, y_raw, (speed, id_prob, num_pixels,) missing
    """

    assert load_num_ids is None or load_num_ids > 0, "load_num_ids must be a positive integer."

    data_dir = f'./locust_data/trex_outputs/batch_{batch_num}/data/'
    num_ids = len(os.listdir(data_dir))
    ids = np.arange(num_ids) if load_num_ids is None else np.arange(load_num_ids)

    datasets = []
    for i in ids:
        path = os.path.join(data_dir, f"{file_name}_id{i}.npz")
        with np.load(path) as d:
            frames = d['frame']
            ds = xr.Dataset(
                {
                    'x_raw': (['frame'], d['X#wcentroid']),
                    'y_raw': (['frame'], d['Y#wcentroid']),
                    # 'trex_speed': (['frame'], d['SPEED#wcentroid']),
                    # 'id_prob': (['frame'], d['visual_identification_p']),
                    # 'num_pixels': (['frame'], d['num_pixels']),
                    'missing': (['frame'], d['missing']),
                },
                coords={'frame': frames, 'id': i},
            )
            datasets.append(ds)

    # Concatenate along 'id' dimension
    full_ds = xr.concat(datasets, dim='id', join = 'outer')
    full_ds = full_ds.where(np.isfinite(full_ds), np.nan)
    
    return len(ids), full_ds

def load_preprocessed_data(load_name): # Load pre-processed data from h5s
    """
    Load an xarray Dataset from an HDF5 file.
    """
    ds = xr.open_dataset(load_name, engine="h5netcdf")
    return ds.load()

def save_data(ds, save_name): # Save pre-processed data to h5s
    for var in ds.data_vars:
        if ds[var].dtype == np.float64: # Losing some precision here, but saves a lot of space
            ds[var] = ds[var].astype(np.float32)
    ds.close()  # Ensure any open files are closed before saving
    encoding = {var: {'compression': 'gzip', 'compression_opts': 4} for var in ds.data_vars}
    ds.to_netcdf(f'{save_name}.h5', engine="h5netcdf", encoding=encoding)

'''_____________________________________________________PARAMETERS____________________________________________________________'''
# Loading parameters
batch_num = 1
exp_name = '20230329'
num_ids = None # None means all, for loading from TReX outputs

# Activity thresholds for each computed batch (index is batch number)
threshes = [0.1483, 0.1373]

# Smoothing and interpolating parameters
# speed_dict = {'raw': None, 'moving_avg': {'window_length': 5, 'center': True}, 'moving_med': {'window_length': 5, 'center': True}, 'sg': {'window_length': 5, 'polyorder': 2, 'deriv': 0, 'delta': 1.0}}
speed_dict = {'raw': None, 'high_ord':{'dt': 1, 'num_iterations': 1, 'order': 4}} # , 'moving_med': {'window_length': 5, 'center': True}, 'sg': {'window_length': 5, 'polyorder': 2, 'deriv': 0, 'delta': 1.0}}
interp_dict= {'max_gap': 5, 'max_dist': 10}
fill_gaps = True

# Visualizing and animating parameters
vid_path = './locust_data/trex_inputs/20230329.mp4'
system = 'locusts'
'''_____________________________________________________RUN CODE____________________________________________________________'''



# print(ds)
# plot_tracks(ds, t_slice = slice(0, 500), system = system)
# animate_trajectories_with_video(ds, np.arange(num_ids), vid_path, system, start_frame=0, end_frame=500, interval=200, trail=10)
# check_trex_speeds(ds)

if __name__ == "__main__":

    # Load data
    print('Make sure batch number is not excluded in dockerignore.')
    num_ids, ds_raw = load_trex_data(batch_num, exp_name, num_ids) # Last integer specifies how many IDs to include
    print('TRex data loaded. Number of IDs:', num_ids)

    # Pre-process data
    ds = preprocess_data(ds_raw, speed_dict, fill_gaps=fill_gaps, interp_dict=interp_dict, center_only=True, radius=960) # Center only to exclude stray detections near borders
    print('Data pre-processed.')

    # Save pre-processed data
    save_name = system + '_batch_' + str(batch_num) + '_' + exp_name
    save_data(ds, save_name)
    print('Data saved.')

    # Load previously pre-processed data
    # load_name = system + '_batch_' + str(batch_num) + '_' + exp_name + '_2.h5'
    # ds = load_preprocessed_data(load_name)
    # print('Pre-processed data loaded.')

    # Plot speed histograms
    # plot_speed_hists(ds, speed_names = ['raw', 'high_ord', 'sg'], fit_speed = False)
    # print('Plotted speed histograms.')

    # print(np.nanstd(ds['sg_speed'].values))

    # Plot smoothed coordinates
    # plot_smoothed_coords(ds, id = 0, speed_names = ['high_ord', 'moving_med', 'sg'], t_slice = slice(0, 200))
    # print('Plotted smoothed coordinates.')

    # Plot histograms of track lengths
    # plot_tracklet_lengths_hist(ds_raw, speed_dict, interp_dict, radius=800)
    # print('Plotted tracklet length histograms.')

    # Plot orientations and angular speed over time
    # plot_ang_speed(ds, t_slice = slice(0, 500))
    # print('Plotted orientations and angular speed over time.')

    # Plot number of tracklets over time
    # plot_num_tracklets_over_time(ds)
    # print('Plotted number of tracklets over time.')

    # Animate tracklet lengths
    # animate_trajs_coloured(ds, vid_path, colours=ds['tracklet_length'], cbar_name='Tracklet length', start_frame=0, end_frame=-1, interval=50)
    # print('Animated tracklet lengths.')
    pass
