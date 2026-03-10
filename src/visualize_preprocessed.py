'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import os

cwd = os.getcwd()
if cwd.endswith('src'):
    from helper_fns import *
else:
    from src.helper_fns import *


'''_____________________________________________________PLOT FUNCTIONS____________________________________________________________'''

def plot_tracks(ds, output_dir: str, ids: np.array, start_frame:int = 0, end_frame:int | None = None,):

    frames = get_frame_slice(ds, start_frame, end_frame)

    plt.figure(figsize=(10, 10))
    for i in ids:
        plt.plot(ds['x_raw'].sel(id=i, frame=frames), ds['y_raw'].sel(id=i, frame=frames), marker='.', label=f'ID {i}')
    plt.xlabel('x', fontsize = 17)
    plt.ylabel('y', fontsize = 17)
    plt.title('Locust tracks', fontsize = 17)
    if len(ds['id'].values) <= 10:
        plt.legend()
    plt.savefig(output_dir + 'original_tracks.png')

def plot_smoothed_coords(ds, output_dir: str, id: int, smooth_names: list, start_frame: int = 0, end_frame: int | None = None): # Takes preprocessed xarray.Dataset as input that has raw and sg computed
    
    frames = get_frame_slice(ds, start_frame, end_frame)

    # Look at raw versus smooth x and y data for a single id
    coords = ['x', 'y', 'v']
    _, axs = plt.subplots(len(coords), len(smooth_names), figsize = (8, 10), sharex = True)
    
    for i, name in enumerate(smooth_names):
        for j, coord in enumerate(coords):
            axs[j][i].plot(ds['frame'].isel(frame=frames), ds[coord + '_raw'].isel(id=id, frame=frames), label = 'Original')
            axs[j][i].plot(ds['frame'].isel(frame=frames), ds[coord + '_' + name].isel(id=id, frame=frames), label = 'Smoothed', linestyle = '--')
            if j == len(coords) - 1:
                axs[j][i].set_xlabel('Frame', fontsize = 17)
            if i == 0:
                axs[j][i].set_ylabel(coord, fontsize = 17)
            if j == 0:
                axs[j][i].set_title(name, fontsize = 17)
    axs[-1][-1].legend()

    plt.tight_layout()
    plt.savefig(output_dir + 'smoothed_speeds.png')

def plot_speed_hists(ds, output_dir: str, smooth_names: list): # Takes preprocessed xarray.Dataset as input

    # Look at all speeds for each ID
    _, axs = plt.subplots(len(smooth_names), 1, figsize=(26, 12), sharey = True, sharex = True)
    
    # Ensure axs is always a list for consistent indexing
    if len(smooth_names) == 1:
        axs = [axs]     

    for i in range(len(smooth_names)):
        speed_vals = ds[f'v_{smooth_names[i]}'].values.ravel()
        valid_speeds = speed_vals[np.isfinite(speed_vals)]
        sns.histplot(valid_speeds, ax=axs[i], bins=500)
        axs[i].set_title(f'{smooth_names[i].capitalize()} speed')
    
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Speed')
    plt.ylabel('Frequency')
    plt.tight_layout()

    plt.savefig(output_dir + 'speed_histograms.png')

def corr_speed_tracklet_length(ds, output_dir: str, smooth_name: str):
    plt.figure(figsize=(10, 10))

    plt.plot(ds[f'v_{smooth_name}'].values.ravel(), ds['tracklet_length'].values.ravel(), 'o', markersize=1)
    plt.xlabel('Speed')
    plt.ylabel('Tracklet length')
    plt.savefig(output_dir + f'corr_speed_tracklet_length_{smooth_name}.png')

def corr_speed_pos_in_tracklet(ds, smooth_name: str, output_dir: str):
    plt.figure(figsize=(10, 10))

    rel_pos = within_tracklet_pos(ds[f'v_{smooth_name}'].values)
 
    plt.plot(ds[f'v_{smooth_name}'].values.ravel(), rel_pos.flatten(), 'o', markersize=1)
    plt.xlabel('Speed', fontsize = 17)
    plt.ylabel('Position in tracklet relative to its end', fontsize = 17)  
    plt.savefig(f'{output_dir}/corr_speed_pos_in_tracklet_{smooth_name}.png')

def plot_single_tracklet_lengths(ds, var: str, output_dir: str):
    plt.figure(figsize=(10, 10))

    # Get tracklet lengths
    tracklet_lengths = run_lengths(ds[var].values)

    # Plot histogram of tracklet lengths
    plt.hist(tracklet_lengths, bins = 11)
    plt.xlabel('Tracklet length', fontsize = 17)
    plt.ylabel('Frequency', fontsize = 17)
    plt.yscale('log')
    # plt.xscale('log')
    plt.savefig(output_dir + f'tracklet_lengths_{var}.png')

# def plot_tracklet_lengths_hist(ds_raw, speed_dict: dict, interp_dict: dict, radius: float, exp_name: str, batch_num: int, n_bins = 15):

#     # Initiate plot
#     _, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (8, 5))
#     labels = [f'None', f'{round(radius)}']

#     # Iterate over no interpolation, with interpolation
#     for i in range(2):
#         lengths_list = []
#         # Iterate over all data, centered
#         for j in range(2):
#             print(2*i + j + 1)
#             ds = preprocess_data(ds_raw, speed_dict, fill_gaps = bool(i), interp_dict = interp_dict, center_only = bool(j), radius = radius)

#             # Broadcast id coordinate to the same shape as tracklet_id
#             ids = xr.broadcast(ds['id'], ds['tracklet_id'])[0].values.ravel()  # shape (id, frame)

#             # Compute tracklet lengths
#             tids = ds['tracklet_id'].values.ravel()

#             mask = ~np.isnan(tids) # Make mask of non nan tracklet ids
#             valid_ids = ids[mask].astype(int)
#             valid_tids = tids[mask].astype(int)

#             # Count length of tracklets
#             counts = np.bincount(np.ravel_multi_index((valid_ids, valid_tids), (int(valid_ids.max()+1), int(valid_tids.max()+1))))
#             lengths_list.append(counts)

#         max_length = np.max([np.max(lengths) for lengths in lengths_list])
#         bins = np.logspace(0, np.log10(max_length), n_bins)

#         for k, lengths in enumerate(lengths_list):
#             sns.histplot(lengths, ax=ax[i,0], bins=bins, label=labels[k])
            
#         counts1, _ = np.histogram(lengths_list[0], bins = bins)
#         counts2, _ = np.histogram(lengths_list[1], bins = bins)
#         widths = np.diff(bins)
#         bar_width = np.median(widths / bins[:-1]) * bins[:-1]  # fraction of local bins
#         ax[i,1].bar(bins[:-1], counts1 - counts2, color = 'gray', width = bar_width, align = 'edge')

#         ax[i,0].set_ylabel('Count', fontsize = 15)
#         ax[i,1].set_ylabel('Excluded', fontsize = 15)

#         for k in range(2):
#             ax[i,k].set_title('Interpolated' if i == 1 else 'Not interpolated', fontsize = 15)
#             if i:
#                 ax[i,k].set_xlabel('Tracklet length (frames)', fontsize = 13)
#             if not k:
#                 ax[i,k].legend(title = 'Radius')

#     plt.xscale('log')
#     plt.yscale('log')
#     plt.tight_layout()
#     plt.savefig(f'plots/{exp_name}/batch_{batch_num}/preprocess/tracklet_length_hists/rad_{str(int(radius))}.png')

def plot_ang_speed(ds, output_dir: str, smooth_name: str, start_frame: int = 0, end_frame: int | None = None):
    # Look at angular speeds over time for a few IDs

    # Get frame slice
    frames = get_frame_slice(ds, start_frame, end_frame)

    _, axs = plt.subplots(2, 1, figsize=(12, 12), sharex = True)
    column_names = [f'theta_{smooth_name}', f'vtheta_{smooth_name}']

    # Initialize max, min for periodic lines
    min_n = 0
    max_n = 0

    num_ids = 3
    for id in range(num_ids):
        for i in range(len(column_names)):
            fitted_speed = ds[f'{column_names[i]}'].sel(id=id, frame = frames)
            copy = fitted_speed.copy()
            if not i:
                copy[~np.isnan(fitted_speed)] = np.unwrap(fitted_speed[~np.isnan(fitted_speed)]) # Unwrap to make it continuous
                min_n = min(np.nanmin(copy), min_n)
                max_n = max(np.nanmax(copy), max_n)
            axs[i].plot(fitted_speed['frame'], copy, label=f'ID {id}')
            if num_ids < 10:
                axs[i].legend()

    # Round min and max to nearest multiple of 2π
    min_n = -1*np.ceil(abs(min_n) / (2 * np.pi))
    max_n = np.ceil(max_n / (2 * np.pi))

    for n in range(int(min_n), int(max_n)):
        axs[0].axhline(n * 2 * np.pi, color='black', linestyle='--', linewidth=0.5)

    for i in range(len(column_names)):
        axs[i].set_xlabel('Frame', fontsize = 17)
        axs[i].set_ylabel(f'{column_names[i]}', fontsize = 17)

    plt.tight_layout()
    plt.savefig(output_dir + f'ang_speed_over_time_{smooth_name}.png')

def plot_num_tracklets_over_time(ds:xr.Dataset, output_dir:str):
    # Look at number of tracklets in any given instance to see how many individuals are missing. This assumes all are tracked at one point.

	if 'missing' in ds.data_vars:
		num_tracklets = np.sum(ds['missing'] != 1, axis = 1) # Includes interpolated points if they exist
	else:
		print('Dataset does not have "missing" array.')
		num_tracklets = np.sum(~np.isnan(ds['x']), axis = 1)
	
	frames = np.unique(ds.frame.values)
		
	plt.figure(figsize=(12, 6))
	plt.plot(frames, num_tracklets)
	plt.xlabel('Frame', fontsize = 17)
	plt.ylabel('Number of tracked individuals', fontsize = 17) # Assumes TReX at one point tracked all individuals
	plt.title('Tracked individuals over time', fontsize = 17)
	plt.savefig(output_dir + 'num_tracklets_over_time.png')

def corr_tracklet_length_density(ds, output_dir: str, nbrs, radius: float):
    # Look at correlation between tracklet length and density of neighbours

    # Tracklet lengths
    tracklet_lengths = ds['tracklet_length'].values.T.ravel()

    # Get neighbour densities
    nbr_offsets = nbrs['metric'][str(radius)]['nbrs']['offsets']
    print(np.diff(nbr_offsets))
    nbr_densities = np.diff(nbr_offsets)
    nbr_densities[np.isnan(tracklet_lengths)] = np.nan # Set densities to nan where tracklet lengths are nan (i.e. where there are boundaries or other third axis specific filtering)

    plt.figure(figsize=(10, 10))
    plt.plot(nbr_densities, tracklet_lengths, 'o', markersize=1)
    plt.xlabel(f'Neighbour density (radius = {radius})', fontsize = 17)
    plt.ylabel('Tracklet length (frames)', fontsize = 17)
    plt.savefig(output_dir + f'corr_tracklet_length_r_{radius}.png')
    print('Rsquared:', np.corrcoef(nbr_densities[~np.isnan(tracklet_lengths)], tracklet_lengths[~np.isnan(tracklet_lengths)])[0, 1]**2)

def corr_tracklet_length_centrality(ds, output_dir: str):
    # Look at correlation between tracklet length and density of neighbours

    # Tracklet lengths
    tracklet_lengths = ds['tracklet_length'].values.ravel()

    # Get distance from center
    dists = ds['dist_from_center'].values.ravel()

    plt.figure(figsize=(10, 10))
    plt.plot(dists, tracklet_lengths, 'o', markersize=1)
    plt.xlabel(f'Distance to center of arena (px))', fontsize = 17)
    plt.ylabel('Tracklet length (frames)', fontsize = 17)
    plt.savefig(output_dir + 'corr_tracklet_length_centrality.png')
    print('Rsquared:', np.corrcoef(dists[~np.isnan(dists)], tracklet_lengths[~np.isnan(dists)])[0, 1]**2)