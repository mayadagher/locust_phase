'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import matplotlib.pyplot as plt

from visualize_preprocessed import animate_trajs_coloured
from helper_fns import *

'''_____________________________________________________VISUALIZATION FUNCTIONS____________________________________________________________'''

def activity_over_time(ds, activity_thresh: float, exp_name: str, batch_num: int):

    num_tracked = np.sum(ds['missing'] != 1, axis = 0) # Includes interpolated points if they exist
    perc_active = ds['active'].sum(axis = 0)/num_tracked

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(perc_active)), perc_active)
    plt.xlabel('Frame')
    plt.ylabel('Individuals active (%)') # Assumes TReX at one point tracked all individuals
    plt.title(f'Activity threshold: {round(activity_thresh, 4)}')
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/activity_over_time.png')

def active_inactive_bout_lengths(ds, activity_thresh: float, exp_name: str, batch_num: int):
    '''Compute and visualize active/inactive bout lengths.'''

    active_lengths = run_lengths_labeled(ds['active_bout_id'])
    inactive_lengths = run_lengths_labeled(ds['inactive_bout_id'])

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
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/bout_lengths.png') # We expect there to be fewer long active bouts due to tracking error

def active_inactive_bout_lengths_quants(active_lengths, inactive_lengths, f_min, quantiles: list, min_tracklet_length: int, exp_name: str, batch_num: int):
    '''Compute and visualize active/inactive bout lengths.'''
    
    # Use bins that align with log ticks
    max_val = max(np.max(active_lengths), np.max(inactive_lengths))
    max_order = np.floor(np.log10(max_val))
    max_bin = round(max_val, -int(max_order))
    if max_bin < max_val:
        max_bin += 10**(int(max_order))
    
    bins = [np.linspace(10**(i), min(10**(i + 1) - 10**(i), max_bin), min(9, int(max_bin/10**(i)))) for i in range(int(max_order) + 1)]
    bins = np.concatenate([arr for arr in bins])

    # Generate histograms of active and inactive bouts
    plt.hist(active_lengths, bins=bins, alpha=0.7, label = 'Active')
    plt.hist(inactive_lengths, bins=bins, alpha=0.7, label = 'Inactive')
    plt.xlabel('Length of bouts')
    plt.ylabel('Counts')
    plt.xscale('log')
    plt.title(f'Activity quantiles: {[round(q, 2) for q in quantiles]}')
    plt.axvline(min_tracklet_length, color='k', linestyle='--', label = 'Minimum tracklet length')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/activity_quantiles/bout_lengths_quants_{round(quantiles[0], 2)}_{round(quantiles[1], 2)}_fmin_{f_min}.png') # We expect there to be fewer long active bouts due to tracking error

def plot_psds(psd_dict, f_min, exp_name: str, batch_num: int, smooth_names: list, actives = False, quants = None, normalize = False):
    if actives:
        data = []
        data.append(psd_dict[str(f_min)]['active'][str(quants)])
        data.append(psd_dict[str(f_min)]['inactive'][str(quants)])

    else:
        data = []
        data.append(psd_dict[str(f_min)]['all'])

    _, ax = plt.subplots(len(smooth_names), 2, figsize=(10, round(1.5*len(smooth_names))), sharex = True, sharey = True)

    for smooth_i, smooth_name in enumerate(smooth_names):
        for i in range(2): # x/y
            coord = ['x', 'y'][i]
            for j, data_j in enumerate(data):
                d = data_j[coord][smooth_name]
                if normalize:
                    d['mean'] = d['mean'] / d['mean'].sum()

                ax[smooth_i, i].plot(d['freq'], d['mean'], label = ['Active', 'Inactive'][j])

                ax[smooth_i, i].set_xscale('log')
                ax[smooth_i, i].set_yscale('log')
                ax[smooth_i, i].set_xlabel('Frequency (Hz)')
                ax[smooth_i, i].set_ylabel('Power')
                ax[smooth_i, i].set_title(coord + ' ' + smooth_name)
                if actives:
                    ax[smooth_i, i].legend()

    plt.tight_layout()
    fig_name = f'plots/{exp_name}/batch_{batch_num}/psd_fmin_{f_min}_active_quants_{quants}_norm_{int(normalize)}.png' if actives else f'plots/{exp_name}/batch_{batch_num}/psd_all_fmin_{f_min}.png'
    plt.savefig(fig_name)

def plot_autocorr(autocorrs, taus, speed_name : str, exp_name: str, batch_num: int):
    plt.figure(figsize=(10, 5))

    plt.plot(taus, autocorrs)
    plt.xlabel('Lag (frames)')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.title(speed_name)
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/autocorr_all.png')

def plot_activity_autocorr(autocorrs, taus, quants: list, speed_name: str, exp_name: str, batch_num: int):
    plt.figure(figsize=(10, 5))

    for i, autocorr in enumerate(autocorrs):
        plt.plot(taus, autocorr, label = ['Inactive', 'Active'][i])

    plt.xlabel('Lag (frames)')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.title(speed_name)
    plt.legend()
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/autocorr_activity_quants_{quants}.png')

def plot_activity_autocorr_psd(powers, freq, quants: list, speed_name: str, exp_name: str, batch_num: int):

    for i, power in enumerate(powers):
        plt.plot(freq, power, label = ['Inactive', 'Active'][i])

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.title('Autocorrelation of ' +speed_name)
    plt.legend()
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/autocorr_psd_{speed_name}_activity_quants_{quants}.png')