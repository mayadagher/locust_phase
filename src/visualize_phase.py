'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from phase import *

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def plot_phase(ds:xr.Dataset, x_var:str, y_var:str, output_dir:str, labels:list[str], title:str, x_factor:int = 1, gridsize:int = 30):
    """
    Plots a 2D heatmap of x_var vs. y_var.
    """

    # Get flattened valid values for x and y
    valid_mask = (~np.isnan(ds[x_var].values)) & (~np.isnan(ds[y_var].values))
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
    
    # # Optional: Add common state labels
    # ax.text(np.min(density), 0.1, " Swarm", color='white', alpha=0.7)
    # ax.text(np.max(density), 0.9, " School/Milling ", color='white', ha='right', alpha=0.7)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir + f'phase_space_{x_var}_{y_var}.png')

def plot_distribution_over_time(ds: xr.Dataset, y_var: str, y_label:str, output_dir: str, title:str, y_factor:int = 1, fps:int = 5, y_bins:int =50, time_bins:int =200):
    """
    Plots a 2D histogram showing the evolution of the population's distribution.
    
    density_array: np.ndarray of shape (n_frames, n_ids)
    fps: Frames per second for the time axis

    density_bins: Resolution of the y-axis (density values)
    time_bins: Resolution of the x-axis (how much to aggregate time)
    """
    dist_array = ds[y_var].values
    n_frames, n_ids = dist_array.shape
    
    # 1. Create coordinates for every single data point
    # Frame indices repeated for each individual
    frame_indices = np.arange(n_frames)
    x = np.repeat(frame_indices, n_ids) / fps # Convert frames to seconds
    
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
    plt.savefig(output_dir + f'{y_var}_hist_over_time.png')
