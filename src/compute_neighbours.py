'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree
from scipy.spatial import Voronoi
import time
import os
import h5py

from data_handling import *

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

# Compute angular region assignment for neighbours of one focal
def vectorized_region_assignment(x, y, theta, nbr_list, num_regions):
    """
    x, y, theta : shape (agent,)
    nbr_list : Python list of list of neighbour indices
    num_regions : int
    """

    # Pad each neighbour list
    max_nbrs = max((len(arr) for arr in nbr_list), default=0)
    if max_nbrs == 0:
        # No neighbours at this timestep
        return [np.array([], dtype=int) for _ in range(len(x))]

    padded = -np.ones((len(nbr_list), max_nbrs), dtype=int)
    for i, arr in enumerate(nbr_list):
        padded[i, :len(arr)] = arr

    # padded: shape (agent, max_nbrs), where -1 marks padded entries

    # Gather neighbour coords
    # neighbour_x[i, j] = x of j-th neighbour of agent i
    neighbour_x = np.where(padded >= 0, x[padded], np.nan)
    neighbour_y = np.where(padded >= 0, y[padded], np.nan)

    # Broadcast focal → (agent, max_nbrs)
    focal_x = x[:, None]
    focal_y = y[:, None]
    focal_theta = theta[:, None]

    # Compute relative angle
    rel = np.arctan2(neighbour_y - focal_y, neighbour_x - focal_x) - focal_theta
    rel = (rel + np.pi) % (2*np.pi) - np.pi   # normalize to [-pi, pi]

    # Assign regions
    width = 2*np.pi / num_regions
    offset = np.pi / num_regions

    regions = np.floor((rel + offset) / width).astype("float")
    regions = np.mod(regions, num_regions)

    # Remove padded entries
    region_list = []
    for i in range(len(nbr_list)):
        k = len(nbr_list[i])
        region_list.append(regions[i, :k].astype(float))

    return region_list

# Compute neighbours for one timestep (metric)
def metric_step(positions, radius):
    valid_mask = ~np.isnan(positions).any(axis=1)
    positions_valid = positions[valid_mask]

    tree = BallTree(positions_valid)
    new_nbr_indices = tree.query_radius(positions_valid, r=radius)

    # Find original indices of neighbours
    original_indices = np.where(valid_mask)[0]
    nbr_indices = [original_indices[new_inds] for new_inds in new_nbr_indices]

    # Reinsert empty lists for invalid agents
    new_idx = 0
    nbr_full = []
    for valid in valid_mask:
        if valid:
            nbr_full.append(nbr_indices[new_idx])
            new_idx += 1
        else:
            nbr_full.append(np.array([], dtype=int))

    # Remove self from each neighbour list
    nbr_full = [inds[inds != i] for i, inds in enumerate(nbr_full)]

    return nbr_full

# Compute kNN neighbours for one timestep (topological)
def knn_step(positions, k):
    valid_mask = ~np.isnan(positions).any(axis=1)
    positions_valid = positions[valid_mask]

    tree = BallTree(positions_valid)
    _, indices = tree.query(positions_valid, k=k+1)
    
    # Find original indices of neighbours
    original_indices = np.where(valid_mask)[0]
    nbr_indices = [original_indices[new_inds] for new_inds in indices]

    # Reinsert empty lists for invalid agents
    new_idx = 0
    nbr_full = []
    for valid in valid_mask:
        if valid:
            nbr_full.append(nbr_indices[new_idx])
            new_idx += 1
        else:
            nbr_full.append(np.array([], dtype=int))

    # Remove self from each neighbour list
    nbr_full = [inds[inds != i] for i, inds in enumerate(nbr_full)]

    return nbr_full

# Compute Voronoi neighbours for one frame
def voronoi_step(positions, n_agents):
    try:
        vor = Voronoi(positions)
    except Exception:
        return [[] for _ in range(n_agents)]

    nbrs = {i: set() for i in range(n_agents)}
    for i1, i2 in vor.ridge_points:
        nbrs[i1].add(i2)
        nbrs[i2].add(i1)
    return [sorted(list(v)) for v in nbrs.values()]

# Main function: compute neighbours for each frame, and optionally the regions they occupy relative to the focal's heading
def compute_nbrs(ds, interaction: str, interaction_param=None):
    """
    ds: xarray.Dataset with dims ('id', 'frame')
         variables: x_high_ord, y_high_ord, theta_raw (should probably be smoothed for future use)
    """

    # Count length of dimensions
    n_ids = len(ds.id.values)
    frames = ds.frame.values

    # Initialize output lists
    nbrs_container = []

    # Iterate over frames
    for f in frames:

        sub = ds.sel(frame=f)
        positions = np.column_stack([sub.x_sg.values, sub.y_sg.values])

        # Exclude agents who are outside of boundary
        n_agents = np.sum(sub['missing'] != 1) # Check how many active tracklets there are

        if n_agents < 2: # No neighbours possible
            nbrs_container.append([[] for _ in range(n_ids)])
            print('Warning: frame with less than 2 agents.')
            continue

        # Choose interaction rule
        if interaction == "metric":
            nbr_list = metric_step(positions, interaction_param)

        elif interaction == "topo":
            nbr_list = knn_step(positions, interaction_param)

        elif interaction == "voronoi":
            if n_agents < 3: # Voronoi requires at least 2 neighbours
                nbr_list = [np.array([], dtype=int) for _ in range(n_agents)]
            else:
                nbr_list = voronoi_step(positions, n_agents)

        else:
            raise ValueError(f"Interaction {interaction} not implemented. Interaction must be either 'metric', 'topo', or 'voronoi'.")
    
        nbrs_container.append(nbr_list)

    return nbrs_container

def compute_regions(ds, nbr_values, nbr_offsets, num_regions: int):
    '''Compute what regions each neighbour lies in, relative to the orientation of the focal. Uses CSR notation for neighbour lists. Inputs are x/y_high_ord and theta_raw, but this can be switched out
    once smoothing approach has been established.'''


    # Count length of dimensions
    n_ids = len(ds.id.values)
    frames = ds.frame.values
    n_frames = len(frames)

    # Check nbr_offsets has correct length (probably correct for this ds)
    assert len(nbr_offsets) == n_frames * n_ids + 1

    # Initialize output lists
    regions_container = []

    # Iterate over frames
    for fi in range(1, n_frames):

        # Create sub-dataset for this frame
        sub = ds.sel(frame=frames[fi])

        # Get list of neighbours for this frame
        nbrs_list = []
        for ni in range(n_ids):
            start_idx = nbr_offsets[fi * n_ids + ni]
            end_idx = nbr_offsets[fi * n_ids + ni + 1]
            nbrs_list.append(nbr_values[int(start_idx):int(end_idx)])
        
        # Compute what regions each neighbour lies in, relative to the orientation of the focal
        region_vals = vectorized_region_assignment(x=sub.x_sg.values, y=sub.y_sg.values, theta=sub.theta_sg.values, nbr_list=nbrs_list, num_regions=num_regions)
        regions_container.append(region_vals)

    return regions_container

def ragged_to_csr(container):
    """
    container: list of length T
        each element is a list of length N
            each element is a 1D numpy array of variable length

    Returns:
        values  : 1D numpy array (flattened)
        offsets : 1D numpy array of length (T*N + 1)

    To index values correctly, use frame_idx*n_ids + id_idx.
    """
    n_frames = len(container)
    n_ids = len(container[0])
    total = sum(len(arr) for frame in container for arr in frame) # Total length of all neighbours of all ids in all frames

    # Initialize storage arrays
    values = np.empty(total, dtype=np.float32)
    offsets = np.zeros(n_frames * n_ids + 1, dtype=np.float32)

    # Initialize index counters
    k = 0   # flat values index
    p = 0   # flat (t, i) index

    for t in range(n_frames):
        for i in range(n_ids):
            arr = np.asarray(container[t][i], dtype=np.float32)
            values[k:k+len(arr)] = arr
            k += len(arr)
            offsets[p+1] = offsets[p] + len(arr)
            p += 1

    return values, offsets

def create_nbrs_h5(ds, inter_dict, exp_name: str, batch_num: int, do_regions: bool = False, num_regions = None):
    '''Compute neighbours according to different interaction rules and save to h5 files using values-offsets indexing.
    Value: ids of neighbours.
    Offsets: index at which next focal starts.'''

    h5_path = f'./preprocessed/{exp_name}/batch_{batch_num}/nbrs.h5'

    # We load the CSR arrays (values/offsets) for nbrs into memory so they can be passed to compute_regions if neighbours are present but regions are not.
    existing = load_neighbours_hdf5(h5_path)

    # Compute new neighbours/regions for any interaction-param combos not already computed
    for interaction in list(inter_dict.keys()):
        for inter_param in inter_dict[interaction]:

            # Determine whether nbrs/regions already exist in the HDF5
            param_key = str(inter_param)
            has_nbrs = (interaction in existing and param_key in existing[interaction] and 'nbrs' in existing[interaction][param_key])
            has_regions = (interaction in existing and param_key in existing[interaction] and 'regions' in existing[interaction][param_key])

            if has_nbrs: # If we have neighbours, don't have regions, AND want regions, load vals and offsets to be used to compute regions
                print(f'Neighbours for {interaction}/{param_key} already exist.')
                if (not has_regions and do_regions):
                    # Load existing nbrs CSR arrays
                    nbr_vals = existing[interaction][param_key]['nbrs']['values']
                    nbr_offsets = existing[interaction][param_key]['nbrs']['offsets']
                    print(f"Loaded existing neighbours for {interaction}/{param_key} from {h5_path}")
            else:
                print(f"Computing neighbours for {interaction} ({param_key})...")

                # Compute neighbours fresh
                t1 = time.time()
                nbrs_out = compute_nbrs(ds, interaction = interaction, interaction_param = inter_param)
                nbr_vals, nbr_offsets = ragged_to_csr(nbrs_out)
                t2 = time.time()
                print(f'{interaction} ({inter_param}) completed in {round(t2 - t1, 2)} seconds.')

            if (do_regions and not has_regions):
                # If neighbours exist (either loaded or just computed), compute regions
                regions_out = compute_regions(ds, nbr_vals, nbr_offsets, num_regions)
                r_vals, r_offsets = ragged_to_csr(regions_out)

            # Save vals, offsets to an h5 file (appends to existing, or creates a new file)
            if not has_nbrs:
                save_neighbours_hdf5(h5_path, interaction, inter_param, nbr_vals, nbr_offsets, 'nbrs')
                print(f"Saved neighbours for {interaction} ({inter_param}) to {h5_path}.")
                del nbr_vals, nbr_offsets

            if (not has_regions and do_regions):
                save_neighbours_hdf5(h5_path, interaction, inter_param, r_vals, r_offsets, 'regions')
                print(f"Saved regions for {interaction} ({inter_param}) to {h5_path}.")
                del r_vals, r_offsets