'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import matplotlib.pyplot as plt

from visualize_preprocessed import animate_trajs_coloured

'''_____________________________________________________VISUALIZATION FUNCTIONS____________________________________________________________'''

def activity_over_time(ds, activity_thresh: float, exp_name: str, batch_num: int):

    num_tracked = np.sum(ds['missing'] != 1, axis = 0) # Includes interpolated points if they exist
    perc_active = ds['active'].sum(axis = 0)/num_tracked

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(perc_active)), perc_active)
    plt.xlabel('Frame')
    plt.ylabel('Individuals active (%)') # Assumes TReX at one point tracked all individuals
    plt.title(f'Activity threshold: {round(activity_thresh, 4)}')
    plt.savefig(f'pre_process_plots/{exp_name}/batch_{batch_num}/activity_over_time.png')

def active_inactive_bout_lengths(ds, activity_thresh: float, exp_name: str, batch_num: int):
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
    plt.savefig(f'pre_process_plots/{exp_name}/batch_{batch_num}/bout_lengths.png') # We expect there to be fewer long active bouts due to tracking error

# def active_inactive_psd(ds, speed_names: list, activity_thresh: float, exp_name: str, batch_num: int):
#     '''Look at the power spectral density of x and y during active and inactive bouts for the raw data, as well as smoothed data.'''

#     active_bouts, inactive_bouts = compute_bouts(ds) # Computes active and inactive bouts using activity threshold (computed on some smoothed data... should it be raw data?)

    # Compute FFT for active and inactive bouts