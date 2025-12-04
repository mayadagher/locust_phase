'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import seaborn as sns

from numpy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks

from clean_tracks import load_preprocessed_data, animate_trajs_coloured, save_data

'''_____________________________________________________COMPUTATION FUNCTIONS____________________________________________________________'''

# Use KDE to smooth PDF of speeds and find minima
def fast_kde_1d(x, xmin, xmax, bw, n_bins):
    """
    Fast 1D FFT-based Gaussian KDE using a Silverman bandwidth.
    Returns:
        grid: evaluation grid
        pdf: estimated density on grid
    """
    x = np.asarray(x)

    # Bin data
    grid = np.linspace(xmin, xmax, n_bins)
    hist, edges = np.histogram(x, bins=n_bins, range=(xmin, xmax), density=False)
    dx = edges[1] - edges[0]

    # Create Gaussian kernel on the grid
    freqs = fftfreq(n_bins, dx)
    kernel_ft = np.exp(-2 * (np.pi**2) * (bw**2) * (freqs**2))

    # FFT-based convolution
    hist_ft = fft(hist)
    pdf = np.real(ifft(hist_ft * kernel_ft))

    # Normalize to integrate to 1
    pdf /= (pdf.sum() * dx)

    return grid, pdf

def kde_threshold(speeds, xmin, xmax, bw, n_bins):

    # Get grid and PDF from FFT-based KDE
    grid, pdf = fast_kde_1d(speeds, xmin, xmax, bw, n_bins=n_bins)

    # Find peaks of the KDE
    peaks, _ = find_peaks(pdf)
    if len(peaks) < 2:
        return np.nan

    # Get the two highest peaks
    top_two = peaks[np.argsort(pdf[peaks])[-2:]]

    # Sort them left→right
    p1, p2 = np.sort(top_two)

    # Threshold = minimum between peaks
    valley_idx = np.argmin(pdf[p1:p2+1]) + p1
    threshold = grid[valley_idx]

    return threshold

def bootstrap_threshold(x, bw_factor = 1, n_boot=500, n_bins=5000, n_samples = None, seed=0):
    """
    Bootstraps the KDE-based threshold. bw_factor is used to test the sensitivity of the threshold to bandwidth selection.
    Returns:
        thresh        = point estimate from full data
        boot_samples  = array of bootstrap thresholds
    """
    rng = np.random.default_rng(seed)

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    bw = bw_factor * 1.06 * x.std() * len(x)**(-0.2) # Silverman bandwidth for KDE
    thresh = kde_threshold(x, xmin, xmax, bw, n_bins)

    boot = []

    if n_samples is None: # Don't sub-sample
        n_samples = len(x)
    elif n_samples > len(x): # Don't re-sample
        n_samples = len(x)

    for _ in tqdm(range(n_boot)):
        xb = rng.choice(x, size=n_samples, replace=True)
        tb = kde_threshold(xb, xmin, xmax, bw, n_bins = n_bins)
        boot.append(tb)

    boot = np.array(boot)
    return thresh, boot

def validate_bootstrap_threshold(x, n_samples):
    '''Test whether bootstrap threshold is sensitive to small changes in bandwidth. In addition, see whether fraction of active individuals varies greatly across confidence interval. 
    Also see if grid resolution or number of samples varies results.'''

    # Test different bandwidth factors
    bw_factors = [0.5, 1, 1.5, 2]
    bw_results = {}
    print('Checking bandwidth sensitivity')
    for bw_factor in bw_factors:
        thresh, boot = bootstrap_threshold(x, bw_factor=bw_factor, n_boot=500, n_bins = 5000, n_samples=100000)
        ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
        bw_results[bw_factor] = {
            'threshold': thresh,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'proportion_above_ci_low': np.nanmean(x > ci_low),
            'proportion_above_ci_high': np.nanmean(x > ci_high)}
        print('Bandwidth factor: ', bw_factor)
        print(bw_results[bw_factor])

    # Test different grid resolutions
    bin_vals = [2000, 5000, 8000]
    bin_results = {}

    for n_bins in bin_vals:
        thresh, boot = bootstrap_threshold(x, bw_factor=1, n_boot=500, n_bins=n_bins, n_samples=100000)
        ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
        bin_results[n_bins] = {
            'threshold': thresh,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'proportion_above_ci_low': np.mean(x > ci_low),
            'proportion_above_ci_high': np.mean(x > ci_high)}
        print('Grid resolution: ', n_bins)
        print(bin_results[n_bins])

    # Test different grid resolutions
    sample_vals = [20000, 100000, 150000]
    sample_results = {}

    for n_samples in sample_vals:
        thresh, boot = bootstrap_threshold(x, bw_factor=1, n_boot=500, n_bins=5000, n_samples=n_samples)
        ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
        sample_results[n_samples] = {
            'threshold': thresh,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'proportion_above_ci_low': np.mean(x > ci_low),
            'proportion_above_ci_high': np.mean(x > ci_high)}
        print('Number of samples: ', n_samples)
        print(sample_results[n_samples])

    return bw_results, bin_results, sample_results

def compute_active_state(ds, speed_name: str, activity_thresh: float):
    '''Compute active state based on speed and activity threshold.'''
    ds['active'] = ds[speed_name] > activity_thresh
    ds['active'] = xr.where(ds['missing'] == 1, np.nan, ds['active']) # Includes interpolated values
    return ds

def compute_bouts(ds):
    '''Compute active and inactive bouts based on speed and activity threshold.
    Returns: arrays with bout ids separated by NaNs.'''
    n_ids, n_frames = len(ds.id), len(ds.frame)

    # Compute changes in activity
    is_active = ds['active'].astype(float) # Frame: 0 to n_frames
    is_active = xr.where(np.isnan(is_active), 10, is_active) # Dummy value to stop nan propogation after nans in diff

    changes = (is_active.diff(dim='frame') != 0).astype(float) # Any change along the frame dimension is counted (start or end of active bout, or transition to or from nan)
    changes = xr.where((is_active == 10)[:,1:], np.nan, changes) # Frame: 1 to n_frames

    # Create a bout index
    bout_ids = changes.cumsum(dim='frame', skipna=True) # Frame: 1 to n_frames
    bout_ids += (bout_ids[:,0] == 0).astype(int) # Making bout ids start at 1 for ease
    bout_id_offset = bout_ids[:-1, -1].cumsum(dim='id')
    bout_ids[1:] += bout_id_offset.assign_coords(id=slice(1,n_ids)) # Ensuring all bouts have unique ids
    bout_ids = xr.where(np.isnan(changes), np.nan, bout_ids)

    # print('Percent time points nan: ', round(np.sum(np.isnan(bout_ids).values)/(n_ids*n_frames), 4))

    # Determine which bouts are active and which are inactive (but tracked)
    active_bouts = bout_ids.where((is_active == 1)[:,1:])
    inactive_bouts = bout_ids.where((is_active == 0)[:,1:]) # Not exactly the inverse of active, due to NaNs

    # Assign to ds
    ds['active_bout_id'] = active_bouts
    ds['inactive_bout_id'] = inactive_bouts

    # Find length of each bout
    active_ids, active_lengths = np.unique(np.array(active_bouts)[~np.isnan(active_bouts)], return_counts = True)
    inactive_ids, inactive_lengths = np.unique(np.array(inactive_bouts)[~np.isnan(inactive_bouts)], return_counts = True)

    # Build lookup dicts
    active_len_dict = {int(i): int(l) for i, l in zip(active_ids, active_lengths)}
    inactive_len_dict = {int(i): int(l) for i, l in zip(inactive_ids, inactive_lengths)}

    # Helper to map IDs → lengths using vectorized `np.vectorize`
    def map_lengths(arr, length_dict):
        arr = xr.where(np.isnan(arr), 0, arr)
        mapper = np.vectorize(lambda x: length_dict.get(int(x), np.nan))
        return xr.apply_ufunc(mapper, arr, vectorize=True, dask="parallelized", output_dtypes=[float])

    # Apply mapping
    ds["active_bout_length"] = xr.where(~np.isnan(active_bouts), map_lengths(active_bouts, active_len_dict), np.nan)
    ds["inactive_bout_length"] = xr.where(~np.isnan(inactive_bouts), map_lengths(inactive_bouts, inactive_len_dict), np.nan)

    return ds

'''_____________________________________________________VISUALIZATION FUNCTIONS____________________________________________________________'''

def activity_over_time(ds, activity_thresh: float, plot_name: str):

    num_tracked = np.sum(ds['missing'] != 1, axis = 0) # Includes interpolated points if they exist
    perc_active = ds['active'].sum(axis = 0)/num_tracked

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(perc_active)), perc_active)
    plt.xlabel('Frame')
    plt.ylabel('Individuals active (%)') # Assumes TReX at one point tracked all individuals
    plt.title(f'Activity threshold: {round(activity_thresh, 4)}')
    plt.savefig(f'pre_process_plots/{system}/batch_{batch_num}/activity_over_time.png')

def active_inactive_bout_lengths(ds, activity_thresh: float):
    '''Compute and visualize active/inactive bout lengths.'''
    
    # Count lengths of active and inactive bouts
    def count_runs(arr):
        """Count lengths of consecutive identical non-nan values."""
        x = arr.values.ravel()
        x = x[~np.isnan(x)]

        # If empty:
        if x.size == 0:
            raise ValueError('Array is empty.')

        # Find run boundaries
        changes = np.diff(x) != 0
        idx = np.concatenate(([0], np.where(changes)[0] + 1, [len(x)]))

        # Run lengths:
        lengths = np.diff(idx).astype(int)

        return lengths

    active_lengths = count_runs(ds['active_bout_id'])
    inactive_lengths = count_runs(ds['inactive_bout_id'])
    
    # Generate histograms of active and inactive bouts
    bins = np.logspace(0, np.log10(max(np.max(active_lengths), np.max(inactive_lengths))), 15)

    plt.hist(active_lengths, bins=bins, align='mid', alpha=0.7, label = 'Active')
    plt.hist(inactive_lengths, bins=bins, align='mid', alpha=0.7, label = 'Inactive')
    plt.xlabel('Length of bouts')
    plt.ylabel('Counts')
    plt.xscale('log')
    # plt.yscale('log')
    plt.title(f'Activity threshold: {round(activity_thresh, 4)}')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(f'pre_process_plots/{system}/batch_{batch_num}/bout_lengths.png') # We expect there to be fewer long active bouts due to tracking error

def active_inactive_psd(ds, speed_names: list, activity_thresh: float):
    '''Look at the power spectral density of x and y during active and inactive bouts for the raw data, as well as smoothed data.'''

    active_bouts, inactive_bouts = compute_bouts(ds) # Computes active and inactive bouts using activity threshold (computed on some smoothed data... should it be raw data?)

    # Compute FFT for active and inactive bouts

'''_____________________________________________________PARAMETERS____________________________________________________________'''
# Loading parameters
batch_num = 1
exp_name = '20230329'
system = 'locusts'

# Activity thresholds for each computed batch (index is batch number)
# threshes = [0.1483, 0.1373] # from sg speed
threshes = [0.3352, 0.3243] # High order diff speed

# Visualizing and animating parameters
vid_path = './locust_data/trex_inputs/20230329.mp4'
'''_____________________________________________________RUN CODE____________________________________________________________'''


if __name__ == "__main__":

    # Load previously pre-processed data
    load_name = system + '_batch_' + str(batch_num) + '_' + exp_name + '_2.h5'
    ds = load_preprocessed_data(load_name)
    print('Pre-processed data loaded.')

    # print(ds.data_vars)
    # ids, frames = ds.sizes
    # print(ids, frames)

    # Compute activity threshold
    # valid_speeds = ds['v_sg'].values[np.isfinite(ds['v_sg'].values)]
    valid_speeds = ds['v_high_ord'].values[np.isfinite(ds['v_high_ord'].values)]
    bw = 1.06 * valid_speeds.std() * len(valid_speeds)**(-0.2) # Silverman bandwidth for KDE
    activity_thresh = kde_threshold(valid_speeds, valid_speeds.min(), valid_speeds.max(), bw, n_bins = 5000)
    print('Computed activity threshold: ', round(activity_thresh, 4))

    # Compute uncertainty of activity threshold
    bw_results, _, _ = validate_bootstrap_threshold(valid_speeds, n_samples=100000)
    print('Threshold: ', round(bw_results[1]['threshold'], 4))
    print('95% CI: ', round(bw_results[1]['ci_low'], 4), round(bw_results[1]['ci_high'], 4))
    print('Methodological uncertainty (0.5 - 1.5 Silverman bandwidths): ', round(bw_results[0.5]['threshold'], 4), round(bw_results[1.5]['threshold'], 4))

    # Append activity column to ds
    # activity_thresh = threshes[batch_num] # Using this to save time during testing
    ds = compute_active_state(ds, speed_name='v_high_ord', activity_thresh=activity_thresh)
    print('Computed active state.')

    # Compute bout ids for active and inactive states
    ds = compute_bouts(ds)
    print('Assigned active and inactive bout ids.')

    # Re-save data, since bout computation takes some time
    save_name = system + '_batch_' + str(batch_num) + '_' + exp_name
    save_data(ds, save_name)
    print('Saved updated data.')

    # Animate active bout lengths:
    # animate_trajs_coloured(ds, vid_path, colours = ds['active_bout_length'], cbar_name = 'Active bout length', start_frame=0, end_frame=-1, interval=50)
    # print('Animated active bout lengths.')

    # Visualize percent active over time
    # activity_over_time(ds, activity_thresh=activity_thresh)
    # print('Plotted activity over time.')

    # Visualize histograms of active and inactive bout lengths
    # active_inactive_bout_lengths(ds, activity_thresh=activity_thresh)
    # print('Plotted histograms of active and inactive bout lengths.')
