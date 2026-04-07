'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

from entropy import *

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def bin_occupation_histogram(occupancy_data:np.ndarray, output_dir:str, k_layers:np.ndarray, r_max:float=100, n_focals:int | None = None):

    bins = np.arange(0, np.max(occupancy_data) + 1).astype(int)
    counts = np.bincount(occupancy_data.flatten().astype(int))
    plt.bar(bins, counts)
    plt.savefig(output_dir + f'entropy/bin_occupation_hist_{k_layers}_{r_max}_{n_focals}.png')
    print(f'% invalid: {np.sum(counts[2:])/(np.sum(counts))}')

def occupants_histogram(occupancy_data:np.ndarray, output_dir:str, k_layers:np.ndarray, r_max:float, n_focals:int | None = None):

    n_occupants = np.sum(occupancy_data, axis = 2).flatten()

    plt.hist(n_occupants, bins = np.max(n_occupants))
    plt.savefig(output_dir + f'entropy/n_occupants_hist_{k_layers}_{r_max}_{n_focals}.png')

def entropy_over_time(occupancy_data:np.ndarray, null_occupancy_data:np.ndarray, null_name:str, output_dir:str, k_layers:np.ndarray, n_samples:int, r_max:float=100, n_focals:int | None = None):

    s_t, ret_t, unique_t, counts_t, degen_t = calculate_entropy(occupancy_data, k_layers, n_samples)
    null_s_t, null_ret_t, _, _, null_degen_t = calculate_entropy(null_occupancy_data, k_layers, n_samples)

    _, ax = plt.subplots(3, 1, sharex=True, figsize = (10, 8))
    t = np.arange(len(s_t))
    ax[0].plot(t, s_t)
    ax[0].plot(t, null_s_t)
    ax[0].set_ylabel('Entropy (bits)', fontsize = 17)
    
    ax[1].plot(t, ret_t)
    ax[1].plot(t, null_ret_t)
    ax[1].set_ylabel('Valid states (%)', fontsize = 17)

    ax[2].plot(t, degen_t, label = 'Data')
    ax[2].plot(t, null_degen_t, label = f'{null_name} null')
    ax[2].set_ylabel('Num. degenerate states', fontsize = 17)
    ax[2].set_xlabel('Frame', fontsize = 17)
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir + f'entropy/s_over_time_{k_layers}_{r_max}_{n_focals}_{null_name}.png')

def state_hist(occupancy_data:np.ndarray, output_dir:str, k_layers:np.ndarray, n_samples:int, r_max:float=100, n_focals:int | None = None):

    # Reshape to (Samples, Bins)
    flat_data = occupancy_data.reshape(-1, occupancy_data.shape[2])

    # Keep only states where max occupancy < 2
    clean_states = flat_data[np.max(flat_data, axis=1) < 2, :]

    # Count unique states, treating each grid as a single state
    unique_states, counts = np.unique(clean_states, axis=0, return_counts=True)
        
    # Check for degenerates and get state ids
    new_ids, new_counts = compute_degenerates_fast(unique_states, counts, k_layers)
    print(new_ids)
    print(new_counts)

    fig = plt.figure()
    plt.bar(new_ids, new_counts)
    plt.xlabel('Configuration ID', fontsize = 17)
    plt.ylabel('Counts', fontsize = 17)
    plt.savefig(output_dir + f'entropy/state_hist_{k_layers}_{r_max}_{n_focals}.png')

def plot_pca(occupancy_data:np.ndarray, output_dir:str,  k_layers:np.ndarray, r_max:float=100, n_focals:int | None = None):

    X_pca, pca = get_pca(occupancy_data)

    _ = plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.grid(True)
    plt.savefig(output_dir + f'entropy/pca_variance_{k_layers}_{r_max}_{n_focals}.png')

def plot_bin_heatmap(occupancy_data: np.ndarray, output_dir: str, k_layers: np.ndarray, r_max: float):
    # 1. Clean and Process Data
    # Reshape to (Samples, Bins)
    flat_data = occupancy_data.reshape(-1, occupancy_data.shape[2])
    # Keep only states where max occupancy < 2
    clean_states = flat_data[np.max(flat_data, axis=1) < 2, :]
    probabilities = np.mean(clean_states, axis=0)

    # 2. Setup the Plot (Standard Cartesian, NOT Polar)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    n_layers = len(k_layers)
    r_edges = np.linspace(0, r_max, n_layers + 1)
    dr = r_max / n_layers
    
    patches = []
    values = []
    current_idx = 0

    for l in range(n_layers):
        r_outer = r_edges[l+1]
        n_ang = k_layers[l]
        
        # Define angles. 

        # We subtract a 90-degree offset so 0 degrees (forward) is "UP".
        theta_edges = np.linspace(0, 360, n_ang + 1)
        
        for i in range(n_ang):
            t_start = theta_edges[i] - 90
            t_end = theta_edges[i+1] - 90
            
            # width must be the THICKNESS of the ring
            # radius is the OUTER edge
            wedge = Wedge(center=(0, 0), r=r_outer, theta1=t_start, theta2=t_end, width=dr)
            
            patches.append(wedge)
            values.append(probabilities[current_idx])
            current_idx += 1

    # 3. Collection for efficient plotting
    p = PatchCollection(patches, cmap='viridis', edgecolors='white', linewidths=0.5)
    p.set_array(np.array(values))
    ax.add_collection(p)
    
    # 4. Focal Heading Indicator (Pointing UP to match the +90 shift)
    ax.plot(0, 0, marker='^', color='red', markersize=15, label='Focal heading')
    
    # 5. Final Formatting
    ax.set_xlim(-r_max - 5, r_max + 5)
    ax.set_ylim(-r_max - 5, r_max + 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Occupation probability', size=14)
    plt.title("Egocentric occupancy", fontsize=16)
    
    plt.legend(loc='upper right')
    plt.savefig(f"{output_dir}/entropy/bin_heatmap_{k_layers}.png", bbox_inches='tight')
    plt.show()

def plot_density_bin_heatmap(occupancy_data: np.ndarray, output_dir: str, k_layers: np.ndarray, r_max: float):
    # 1. Clean and Process Data
    flat_data = occupancy_data.reshape(-1, occupancy_data.shape[2])
    clean_states = flat_data[np.max(flat_data, axis=1) < 2, :] # (N, n_bins)

    # THRESHOLD
    n_occupants = np.sum(clean_states, axis = 1).flatten()
    low_thresh = np.quantile(n_occupants, 0.25, method='lower')
    high_thresh = np.quantile(n_occupants, 0.75, method='higher')

    # Split microstates by number of occupants
    occ_low = clean_states[n_occupants < low_thresh]
    occ_high = clean_states[n_occupants > high_thresh]

    # Get average config for low and high density
    probs = [np.mean(occ_low, axis = 0), np.mean(occ_high, axis = 0)]

    # 2. Setup the Plot (Standard Cartesian, NOT Polar)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    n_layers = len(k_layers)
    r_edges = np.linspace(0, r_max, n_layers + 1)
    dr = r_max / n_layers

    for i in range(2):
    
        patches = []
        values = []
        current_idx = 0

        for l in range(n_layers):
            r_outer = r_edges[l+1]
            n_ang = k_layers[l]
            
            # Define angles. 

            # We subtract a 90-degree offset so 0 degrees (forward) is "UP".
            theta_edges = np.linspace(0, 360, n_ang + 1)
            
            for k in range(n_ang):
                t_start = theta_edges[k] - 90
                t_end = theta_edges[k+1] - 90
                
                # width must be the THICKNESS of the ring
                # radius is the OUTER edge
                wedge = Wedge(center=(0, 0), r=r_outer, theta1=t_start, theta2=t_end, width=dr)
                
                patches.append(wedge)
                values.append(probs[i][current_idx])
                current_idx += 1

        # 3. Collection for efficient plotting
        p = PatchCollection(patches, cmap='viridis', edgecolors='white', linewidths=0.5)
        p.set_array(np.array(values))
        # p.set_clim(np.min(probs), np.max(probs))
        ax[i].add_collection(p)
        
        # 4. Focal Heading Indicator (Pointing UP to match the +90 shift)
        ax[i].plot(0, 0, marker='^', color='red', markersize=15, label='Focal heading')
        
        # 5. Final Formatting
        ax[i].set_title(['Low local density', 'High local density'][i], fontsize = 17)
        ax[i].set_xlim(-r_max - 5, r_max + 5)
        ax[i].set_ylim(-r_max - 5, r_max + 5)
        ax[i].set_aspect('equal')
        ax[i].axis('off')
    
        cbar = plt.colorbar(p, ax=ax[i], fraction=0.046, pad=0.04)
        cbar.set_label('Occupation probability', size=14)
    plt.suptitle(f"Locality: {round(r_max//50)} BL", fontsize=16)
    
    ax[1].legend(loc='upper right')
    plt.savefig(f"{output_dir}/entropy/bin_heatmap_by_density_{k_layers}_{r_max}.png", bbox_inches='tight')
    plt.show()

