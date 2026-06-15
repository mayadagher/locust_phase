'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Literal

from correlation_length import get_global_voronoi_corrs, metric_corr_vectors
from helper_fns import get_frame_slice, fft_timeseries
from phase import *

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def plot_phase(ds:xr.Dataset, x_var:str, y_var:str, output_dir:str, labels:list[str], title:str, x_factor:int = 1, gridsize:int = 30):
    """
    Plots a 2D heatmap of x_var vs. y_var.
    """

    # Get flattened valid values for x and y
    valid_mask = (~np.isnan(ds[x_var].values)) & (~np.isnan(ds[y_var].values)) & (ds[x_var].values < np.nanquantile(ds[x_var].values, 0.999))
    x_valid = ds[x_var].values[valid_mask].flatten()*x_factor # Rescale if necessary
    y_valid = ds[y_var].values[valid_mask].flatten()

    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Create the hexbin heatmap
    # mincnt=1 ensures we don't color empty areas
    hb = ax.hexbin(x_valid, y_valid, gridsize=gridsize, cmap='viridis', mincnt=1) #, bins='log') # Log scale for better contrast
    
    # Add Colorbar
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Occupancy', size=12)
    
    # Labeling the 2D space
    ax.set_xlabel(labels[0], fontsize=17)
    ax.set_ylabel(labels[1], fontsize=17)
    ax.set_title(title, fontsize=17, pad=15)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir + f'phase_space_{x_var}_{y_var}.png')

def plot_distribution_over_time(ds: xr.Dataset, y_var: str, y_label:str, output_dir: str, title:str, y_factor:int = 1, fps:int = 5, start_frame:int = 0, end_frame:int = None, y_bins:int =50, time_bins:int =200, subsample:int = 1):
    """
    Plots a 2D histogram showing the evolution of the population's distribution.
    
    density_array: np.ndarray of shape (n_frames, n_ids)
    fps: Frames per second for the time axis

    density_bins: Resolution of the y-axis (density values)
    time_bins: Resolution of the x-axis (how much to aggregate time)
    """
    # Get absolute frame values
    abs_frames, ds_idcs = get_frame_slice(ds, rel_start = start_frame, rel_end = end_frame, in_function_subsample = subsample)

    dist_array = ds[y_var].values[ds_idcs, :]
    _, n_ids = dist_array.shape
    
    # 1. Create coordinates for every single data point
    x = np.repeat(ds_idcs, n_ids) / fps # Convert frames to seconds
    
    # Flatten the density values
    y = dist_array.flatten()*y_factor # Rescale if necessary
    
    # 2. Mask out NaNs (critical for tracking data)
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    # 3. Plot
    plt.figure(figsize=(12, 7))
    
    # We use LogNorm so we can see rare outliers and the main crowd simultaneously
    h = plt.hist2d(x_clean, y_clean, bins=[time_bins, y_bins], cmap='magma', norm=LogNorm(), cmin=1)
    
    # Add styling
    cb = plt.colorbar(h[3])
    cb.set_label('Num. individuals', rotation=270, fontsize = 15, labelpad=15)
    plt.title(title, fontsize = 17)
    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir + f'{y_var}_hist_over_time_{abs_frames[0]}_{abs_frames[1]}_fs_{fps/round(np.diff(abs_frames)[0])}.png')

def plot_voronoi_corr_single_frame(ds:xr.Dataset, param:str, frame_idx:int, arena_center:np.ndarray, arena_radius:float, output_dir:str, title:str, tolerance:float= 1e-6, param_type: Literal["scalar", "circular"] = "scalar", max_layer:int | None = None, min_pairs_per_layer:int = 5, subsample:int | None = None):

    # Define position array to save time from accessing ds
    positions = np.stack([ds['centroid_x'], ds['centroid_y']])[:,frame_idx,:].T # (max_ids, 2)

    # Define value of variable to be correlated
    z_vals = ds[param].values[frame_idx,:] # (max_ids,)

    # Filter out detections outside of arena
    dist_from_center = np.sqrt((positions[:,0] - arena_center[0])**2 + (positions[:,1] - arena_center[1])**2) # (max_ids,)
    outside_arena_mask = dist_from_center > arena_radius

    # Define valid mask
    valid_mask = (~np.isnan(positions).any(axis = 1)) & (~np.isnan(z_vals)) & (~outside_arena_mask) # (max_ids,)

    # Get valid positions and z values
    positions = positions[valid_mask] # (N,)
    z_vals = z_vals[valid_mask] # (N,)

    # Create voronoi tessellation
    vor = Voronoi(positions)

    # Iterate over each point to collect VALID polygons and their z values
    cells = []
    all_z = []

    for point_idx, z_val in enumerate(z_vals):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]

        # Exclude regions with no points
        if len(region) == 0:
            continue

        # Clip and order vertices
        vertices = vor.vertices[np.array(region)[(np.array(region) != -1).astype(bool)].astype(int)]
        poly = Polygon(clip_voronoi_region(vertices, arena_center, arena_radius))

        if poly.is_empty:
            continue

        # Append valid polygons and their corresponding z values to new lists
        cells.append(poly)
        all_z.append(z_val)

    # Get correlation vectors for each individual
    ks, C, C_se, n_pairs, ks_all, mat = get_global_voronoi_corrs(cells, np.array(all_z), tolerance, param_type, max_layer, min_pairs_per_layer, subsample)

    # Plot EACH correlation vector as well as the mean to see if there are different sub-populations
    _, ax = plt.subplots()
    
    for vec in mat:
        ax.plot(ks_all, vec, alpha = 0.1)
    ax.plot(ks, C, color = 'k')
    ax.set_xlabel('Voronoi layer', fontsize = 17)
    ax.set_ylabel('Correlation', fontsize = 17)
    ax.set_title(title, fontsize = 19)
    ax.grid()

    abs_frame = ds.frame.values[frame_idx]

    plt.savefig(output_dir + f'correlation_lengths/voronoi_corr_{param}_frame_{abs_frame}_subsample_{subsample}.png')

def compare_corrs_single_frame(ds:xr.Dataset, param:str, frame_idx:int, arena_center:np.ndarray, arena_radius:float, output_dir:str, title:str, tolerance:float= 1e-6, param_type: Literal["scalar", "circular"] = "scalar", max_layer:int | None = None, min_pairs_per_layer:int = 5, subsample:int | None = None, metric_n_bins:int = 50, distance_factor:float = 1):
    
    # Plot EACH correlation vector as well as the mean to see if there are different sub-populations for different distance metrics
    _, ax = plt.subplots(1, 4, figsize = (18, 5))

    # PART 1: VORONOI CORRELATION

    # Define position array to save time from accessing ds
    positions = np.stack([ds['centroid_x'], ds['centroid_y']])[:,frame_idx,:].T # (max_ids, 2)

    # Define value of variable to be correlated
    z_vals = ds[param].values[frame_idx,:] # (max_ids,)

    # Filter out detections outside of arena
    dist_from_center = np.sqrt((positions[:,0] - arena_center[0])**2 + (positions[:,1] - arena_center[1])**2) # (max_ids,)
    outside_arena_mask = dist_from_center > arena_radius

    # Define valid mask
    valid_mask = (~np.isnan(positions).any(axis = 1)) & (~np.isnan(z_vals)) & (~outside_arena_mask) # (max_ids,)

    # Get valid positions and z values
    positions = positions[valid_mask] # (N,)
    z_vals = z_vals[valid_mask] # (N,)

    # Create voronoi tessellation
    vor = Voronoi(positions)

    # Iterate over each point to collect VALID polygons and their z values
    valid_indices = []

    for point_idx, z_val in enumerate(z_vals):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]

        # Exclude regions with no points
        if len(region) == 0:
            continue

        # Clip and order vertices
        vertices = vor.vertices[np.array(region)[(np.array(region) != -1).astype(bool)].astype(int)]
        poly = Polygon(clip_voronoi_region(vertices, arena_center, arena_radius))

        if poly.is_empty: # Use this method to do one last check and see if the vertices are valid
            continue

        # Append valid indices to new list
        valid_indices.append(point_idx)

    # Get correlation vectors for each individual
    ks, C, C_se, n_pairs, ks_all, mat = get_global_voronoi_corrs(vor, np.array(valid_indices), z_vals, tolerance, param_type, max_layer, min_pairs_per_layer, subsample)
    
    for vec in mat:
        ax[0].plot(ks_all, vec, alpha = 0.1)
    ax[0].plot(ks, C, color = 'k')
    ax[0].set_xlabel('Voronoi layer', fontsize = 17)
    ax[0].set_ylabel('Correlation', fontsize = 17)
    ax[0].grid()

    # PART 2: METRIC CORRELATION
    metric_results = metric_corr_vectors(positions, z_vals, param_type=param_type, n_bins=metric_n_bins, min_pairs_per_bin=min_pairs_per_layer, subsample=subsample, axis = None)
    metric_x_results = metric_corr_vectors(positions, z_vals, param_type=param_type, n_bins=metric_n_bins, min_pairs_per_bin=min_pairs_per_layer, subsample=subsample, axis = 0)
    metric_y_results = metric_corr_vectors(positions, z_vals, param_type=param_type, n_bins=metric_n_bins, min_pairs_per_bin=min_pairs_per_layer, subsample=subsample, axis = 1)

    metrics = [metric_results, metric_x_results, metric_y_results]
    for i in range(1, 4):

        indvs = metrics[i - 1]['individual_corrs']
        bin_centers = metrics[i - 1]['bin_centers']*distance_factor
        C = metrics[i - 1]['C']
        for indv in indvs:
            if np.sum(np.isfinite(indv)):
                ax[i].plot(bin_centers, indv, alpha = 0.1)
        ax[i].plot(bin_centers, C, color = 'k')
        ax[i].set_xlabel(['Metric distance (m)', 'Displacement x (m)', 'Displacement y (m)'][i - 1], fontsize = 17)
        ax[i].grid()

    abs_frame = ds.frame.values[frame_idx]
    plt.suptitle(title, fontsize = 19)
    plt.tight_layout()
    plt.savefig(output_dir + f'correlation_lengths/compare_corr_{param}_frame_{abs_frame}_subsample_{subsample}_2.png')

def define_cycle(ds: xr.Dataset, output_dir: str, fps:int = 5, start_frame:int = 0, end_frame:int = None, subsample:int = 1, density_factor:float = 1):

    # Get frames
    abs_frames, ds_idcs = get_frame_slice(ds, start_frame, end_frame, subsample)

    # Take the median values of polarization and density across individuals over each frame
    pols = np.nanmedian(ds['polarization_voronoi_None'].values[ds_idcs,:], axis = 1)
    dens = np.nanmedian(ds['density_voronoi_None'].values[ds_idcs,:], axis = 1)

    # Take an FFT of the polarization median
    freqs, power, phase, dominant_freq_pol = fft_timeseries(pols, fps)

    _, ax = plt.subplots(2, 2)
    ax[0,0].plot(abs_frames, pols)
    ax[0,0].set_ylabel('Median polarization', fontsize = 14)
    ax[0,0].grid()
    
    ax[0,1].plot(freqs, power)
    ax[0,1].axvline(x = dominant_freq_pol, linestyle = '--', c = 'k')
    ax[0,1].set_ylabel('Power', fontsize = 17)
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    print('Polarization phase at maximum power: ', phase[np.argmax(power)])

    ax[1,0].plot(abs_frames, dens*density_factor)
    ax[1,0].set_xlabel('Frame', fontsize = 17)
    ax[1,0].set_ylabel(r'Median density ($n/m^2$)', fontsize = 14)
    ax[1,0].grid()

    # Take an FFT of the density median
    freqs, power, phase, dominant_freq = fft_timeseries(dens, fps)

    ax[1,1].plot(freqs, power)
    ax[1,1].axvline(x = dominant_freq, linestyle = '--', c = 'k')
    ax[1,1].set_ylabel('Power', fontsize = 17)
    ax[1,1].set_xlabel('Frequency (Hz)', fontsize = 17)
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')
    print('Density phase at maximum power: ', phase[np.argmax(power)])

    plt.tight_layout()
    output_name = output_dir + f'marching_cycle_start_{abs_frames[0]}_end_{abs_frames[-1]}_fs_{fps/round(np.diff(abs_frames)[0])}.png'
    plt.savefig(output_name)
    print('Saved to', output_name)
