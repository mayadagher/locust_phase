'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree
from scipy.spatial import Voronoi
import time
import awkward as ak

from clean_tracks import load_preprocessed_data, save_data

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

# Compute angular region assignment for neighbours of one focal
def vectorized_region_assignment(x, y, theta, nbr_list, num_regions):
    """
    x, y, theta : shape (agent,)
    nbr_list : Python list of numpy arrays of neighbour indices
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
        region_list.append(regions[i, :k].astype(int))

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
    tree = BallTree(positions)
    _, indices = tree.query(positions, k=k+1)
    return [inds[1:] for inds in indices]  # drop self

# Compute Voronoi neighbours for one frame
def voronoi_step(positions, n_agents):
    try:
        vor = Voronoi(positions)
    except Exception:
        return [np.array([], dtype=int) for _ in range(n_agents)]

    nbrs = {i: set() for i in range(n_agents)}
    for i1, i2 in vor.ridge_points:
        nbrs[i1].add(i2)
        nbrs[i2].add(i1)
    return [np.array(sorted(list(v))) for v in nbrs.values()]

# Main function: compute neighbours for each frame, and optionally the regions they occupy relative to the focal's heading
def compute_nbrs(ds, interaction: str, interaction_param=None, regions: bool=False, num_regions=None):
    """
    ds: xarray.Dataset with dims ('id', 'frame')
         variables: x_high_ord, y_high_ord, theta_raw
    """

    # Count length of dimensions
    n_ids = len(ds.id.values)
    frames = ds.frame.values
    n_frames = len(frames)

    # Initialize output arrays: object dtype
    nbrs = xr.DataArray(np.empty((n_ids, n_frames), dtype=object), coords=dict(id=ds.id, frame=ds.frame), name="nbrs")

    region_da = None
    if regions:
        region_da = xr.DataArray(np.empty((n_ids, n_frames), dtype=object), coords=dict(id=ds.id, frame=ds.frame),
            name="regions",)

    # Iterate over frames
    for fi, f in enumerate(frames):

        sub = ds.sel(frame=f)
        positions = np.column_stack([sub.x_high_ord.values, sub.y_high_ord.values])

        # Exclude agents who are outside of boundary
        n_agents = np.sum(sub['missing'] != 1) # Check how many active tracklets there are

        if n_agents < 2:
            # no neighbours possible
            nbrs[:, fi] = np.array([np.array([], dtype=int)] * n_agents, dtype=object)
            if regions:
                region_da[:, fi] = np.array([np.array([], dtype=int)] * n_agents, dtype=object)
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
    
        nbrs[:, fi] = nbr_list
        # Optional region assignment
        if regions and fi > 0: # Orientation can't be computed on first frame
            region_vals = vectorized_region_assignment(x=sub.x_high_ord.values, y=sub.y_high_ord.values, theta=sub.theta_raw.values, nbr_list=nbr_list, num_regions=num_regions)
            region_da[:, fi] = region_vals

    # Attach to dataset
    ds_out = ds.copy()
    ds_out[f"nbrs_{interaction}_{interaction_param}"] = nbrs.astype(str)
    if regions:
        ds_out["regions"] = region_da

    return ds_out

'''_____________________________________________________PARAMETERS____________________________________________________________'''
# Loading parameters
batch_num = 1
exp_name = '20230329'
system = 'locusts'

# Interaction parameters
# inter_dict = {'metric': [15, 50, 100], 'topo': [1, 3, 7], 'voronoi': [None]}
inter_dict = {'metric': [15]}

'''_____________________________________________________RUN CODE____________________________________________________________'''

if __name__ == "__main__":

    # Load previously pre-processed data
    load_name = system + '_batch_' + str(batch_num) + '_' + exp_name + '.h5'
    ds = load_preprocessed_data(load_name)
    print('Pre-processed data loaded.')

    # Compute neighbours using different conditions
    ds_temp = ds.copy()
    ds.close()  # Close original dataset to free resources (also necessary for overwriting file later)

    for interaction in list(inter_dict.keys()):
        for inter_param in inter_dict[interaction]:
            t1 = time.time()
            ds_temp = compute_nbrs(ds_temp, interaction = interaction, interaction_param = inter_param)
            t2 = time.time() 
            print(f'{interaction}, {inter_param} completed in {round(t2 - t1, 2)} seconds.')
    print(ds_temp)
    
    # Save new columns
    save_name = system + '_batch_' + str(batch_num) + '_' + exp_name # Overwrite previous file
    save_data(ds_temp, save_name)
    print('Saved updated data.')

    # Clean memory
    ds_temp.close()
    del ds_temp, ds