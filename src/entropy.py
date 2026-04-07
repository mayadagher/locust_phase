'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

from helper_fns import *

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def compute_bin_occupancy(ds:xr.Dataset, n_layers:int, n_ang_bins:int, r_max:float, n_focals:int | None):
    """
    Computes neighbour occupancy in egocentric constant-area bins.
    
    Parameters:
    ds: xr.Dataset containing 'centroids_x', 'centroids_y', 'theta'
    n_layers: Number of radial rings
    n_ang_bins: Number of angular slices
    r_max: Maximum distance to consider
    n_focals: Number of individuals to include (in case of subsampling) -> None includes all ids
                -> IDs are not maintained - it just limits the number of computations
    """

    # Define constant area radial bins
    bin_area = (np.pi * r_max**2) / (n_layers * n_ang_bins)
    d_phi = (2 * np.pi) / n_ang_bins
    
    # Get edges for each r value
    r_edges = [0]
    for i in range(n_layers):
        # Solve for next r: Area = 0.5 * (r_next^2 - r_now^2) * d_phi
        next_r = np.sqrt((2 * bin_area / d_phi) + r_edges[-1]**2)
        r_edges.append(next_r)
    r_edges = np.array(r_edges)
    phi_edges = np.linspace(-np.pi, np.pi, n_ang_bins + 1)

    # Extract coordinates (frame, id)
    x = ds.centroid_x.values
    y = ds.centroid_y.values
    theta = ds.theta.values
    n_frames = x.shape[0]

    # Find minimum number of individuals across frames to avoid sampling bias
    complete_mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(theta)) # (n_frames, max(max_ids))
    max_ids = np.sum(complete_mask, axis = 1) # (n_frames,)
    min_ids = np.min(max_ids)
    n_focals = min(min_ids, n_focals if n_focals else max(max_ids))
    
    # Collect occupancies for each frame
    all_occupancies = []

    # Randomize focal ids in case of subsampling

    if n_focals:
        focals = np.random.choice(ds.coords['id'].values, n_focals, replace = False)
    else:
        focals = ds.coords['id'].values
    
    # Iterate over frames
    for t in tqdm(range(n_frames)):
        occupancies_t = []

        # Randomize focal ids in case of subsampling
        focals = np.random.choice(max_ids[t], n_focals, replace = False) # All non-nan up to max_ids

        # Iterate over focals
        for i in focals:
            # Extract focal coordinates
            x_i = x[t, i]
            y_i = y[t, i]
            theta_i = theta[t, i]

            # Get relative positions of all other ids to the focal
            dx = x[t,complete_mask[t,:]] - x_i
            dy = y[t,complete_mask[t,:]] - y_i

            # Rotate coordinates by -theta_i to align with head direction
            # x' = x cos - y sin | y' = x sin + y cos
            cos_i = np.cos(-theta_i)
            sin_i = np.sin(-theta_i)
            
            rel_x = dx * cos_i - dy * sin_i # (n_focals,)
            rel_y = dx * sin_i + dy * cos_i
            
            # Convert to polar coordinates
            dist = np.sqrt(rel_x**2 + rel_y**2) # (n_focals,)
            angle = np.arctan2(rel_y, rel_x)

            # Mask out self (distance is 0) and anything outside r_max
            dist_mask = (dist > 0) & (dist <= r_max)
            
            counts, _, _ = np.histogram2d(dist[dist_mask], angle[dist_mask], bins=[r_edges, phi_edges])
            occupancies_t.append(counts.flatten())

        all_occupancies.append(occupancies_t)

    # Concatenate data across individuals
    occupancy_data = np.array(all_occupancies) # (n_frames, n_focals, n_bins)
    
    return occupancy_data, r_edges, phi_edges

def compute_occupancy_fixed_dr(ds:xr.Dataset, n_layers:int, n_ang0:int, r_max:float, n_focals:int | None, rotation_null:bool, shuffle_null:bool):
    """
    Computes neighbor occupancy in egocentric bins. Radius increments are fixed but number of
    angular bins is dynamic so as to have approxamitely constant areas and aspect ratios.
    
    Parameters:
    ds: xr.Dataset containing 'centroid_x', 'centroid_y', 'theta'
    n_layers: Number of radial rings
    n_ang0: Number of angular bins in first ring
    r_max: Maximum distance to consider
    n_focals: Number of individuals to include (in case of subsampling) -> None includes all ids
                -> IDs are not maintained - it just limits the number of computations
    """

    # 1. Setup grid geometry
    dr = r_max / n_layers

    # k_layers: number of angular bins per radial ring
    k_layers = np.array([int(round(n_ang0 * (2 * l + 1))) for l in range(n_layers)])
    cum_bins = np.concatenate(([0], np.cumsum(k_layers)))
    total_bins = cum_bins[-1]
    
    angle_widths = (2 * np.pi) / k_layers

    # 2. Extract data
    x_raw = ds.centroid_x.values # (time, id)
    y_raw = ds.centroid_y.values
    theta_raw = ds.theta.values
    n_frames, _ = x_raw.shape

    # Pre-allocate output: (frames, n_focals, total_bins)
    # If n_focals is None, we use the minimum number of valid IDs found across frames to avoid sampling bias
    valid_mask = ~np.isnan(x_raw) & ~np.isnan(y_raw) & ~np.isnan(theta_raw)
    min_ids = np.min(np.sum(valid_mask, axis=1))
    
    if n_focals is None or n_focals > min_ids:
        n_focals = min_ids
        
    occupancy_data = np.zeros((n_frames, n_focals, total_bins), dtype=np.uint8)

    # 3. Iterate over frames
    for t in tqdm(range(n_frames)):
        # Get valid indices for this frame
        frame_mask = valid_mask[t]
        valid_indices = np.where(frame_mask)[0]
        
        # Subsample focals for this frame
        current_focals = np.random.choice(valid_indices, n_focals, replace=False)
        # Create a mapping from global ID to our focal index (0 to n_focals-1)
        focal_map = {global_id: i for i, global_id in enumerate(current_focals)}
        
        coords = np.column_stack((x_raw[t, frame_mask], y_raw[t, frame_mask]))
        global_indices = valid_indices # The indices the KDTree is looking at
        
        tree = KDTree(coords)
        
        # Find all pairs within r_max. 
        # result is a set of (i, j) where i < j (local indices in 'coords')
        pairs = tree.query_pairs(r_max)
        
        for local_i, local_j in pairs:
            idx_i = global_indices[local_i]
            idx_j = global_indices[local_j]
            
            # Check if either animal is in our 'focals' list
            # We process i as focal (seeing j) and j as focal (seeing i)
            potential_focals = []
            if idx_i in focal_map: potential_focals.append((idx_i, idx_j))
            if idx_j in focal_map: potential_focals.append((idx_j, idx_i))
            
            for f_id, n_id in potential_focals:

                if shuffle_null:
                    l_idx = np.random.choice(np.arange(n_layers))
                    a_idx = np.random.choice(np.arange(k_layers[l_idx]))

                else:

                    # 4. Egocentric calculation
                    dx = x_raw[t, n_id] - x_raw[t, f_id]
                    dy = y_raw[t, n_id] - y_raw[t, f_id]
                    
                    # Rotate by -theta
                    if rotation_null:
                        th = 2*np.pi*np.random.random()
                    else:
                        th = -theta_raw[t, f_id]

                    rel_x = dx * np.cos(th) - dy * np.sin(th)
                    rel_y = dx * np.sin(th) + dy * np.cos(th)
                    
                    d = np.sqrt(rel_x**2 + rel_y**2)
                    if d == 0: continue # Skip self
                    
                    # 5. Binning logic
                    # Radial part
                    l_idx = int(d // dr)
                    if l_idx >= n_layers:
                        l_idx = n_layers - 1
                    
                    # Angular part: Map [-pi, pi] to [0, 2pi]
                    angle = np.arctan2(rel_y, rel_x)
                    angle_norm = (angle + np.pi) % (2 * np.pi)
                    
                    a_idx = int(angle_norm // angle_widths[l_idx])
                    if a_idx >= k_layers[l_idx]:
                        a_idx = k_layers[l_idx] - 1
                    
                # Global bin index
                bin_idx = cum_bins[l_idx] + a_idx
                
                # Increment occupancy for this focal at this frame
                focal_idx = focal_map[f_id]
                occupancy_data[t, focal_idx, bin_idx] += 1
                
    return occupancy_data, k_layers

def occupancy_fixed_dr_all(ds:xr.Dataset, n_layers:int, n_ang0:int, r_max:float):
    """
    No sub-sampling so that configurations can be associated to actual focals.
    Parameters:
    ds: xr.Dataset containing 'centroid_x', 'centroid_y', 'theta'
    n_layers: Number of radial rings
    n_ang0: Number of angular bins in first ring
    r_max: Maximum distance to consider
    """

    # 1. Setup grid geometry
    dr = r_max / n_layers

    # k_layers: number of angular bins per radial ring
    k_layers = np.array([int(round(n_ang0 * (2 * l + 1))) for l in range(n_layers)])
    cum_bins = np.concatenate(([0], np.cumsum(k_layers)))
    total_bins = cum_bins[-1]
    
    angle_widths = (2 * np.pi) / k_layers

    # 2. Extract data
    x_raw = ds.centroid_x.values # (time, id)
    y_raw = ds.centroid_y.values
    theta_raw = ds.theta.values
    n_frames, max_ids = x_raw.shape
    
    valid_mask = ~np.isnan(x_raw) & ~np.isnan(y_raw) & ~np.isnan(theta_raw)
    occupancy_data = np.zeros((n_frames, max_ids, total_bins), dtype=np.uint8)

    # 3. Iterate over frames
    for t in tqdm(range(n_frames)):
        # Get valid indices for this frame
        frame_mask = valid_mask[t]
        valid_indices = np.where(frame_mask)[0]
        
        coords = np.column_stack((x_raw[t, frame_mask], y_raw[t, frame_mask]))
        
        tree = KDTree(coords)
        
        # Find all pairs within r_max. 
        # result is a set of (i, j) where i < j (local indices in 'coords')
        pairs = tree.query_pairs(r_max)
        
        for f_id, n_id in pairs:

            # 4. Egocentric calculation
            dx = x_raw[t, n_id] - x_raw[t, f_id]
            dy = y_raw[t, n_id] - y_raw[t, f_id]
            
            # Rotate by -theta
            th = -theta_raw[t, f_id]

            rel_x = dx * np.cos(th) - dy * np.sin(th)
            rel_y = dx * np.sin(th) + dy * np.cos(th)
            
            d = np.sqrt(rel_x**2 + rel_y**2)
            if d == 0: continue # Skip self
            
            # 5. Binning logic
            # Radial part
            l_idx = int(d // dr)
            if l_idx >= n_layers:
                l_idx = n_layers - 1
            
            # Angular part: Map [-pi, pi] to [0, 2pi]
            angle = np.arctan2(rel_y, rel_x)
            angle_norm = (angle + np.pi) % (2 * np.pi)
            
            a_idx = int(angle_norm // angle_widths[l_idx])
            if a_idx >= k_layers[l_idx]:
                a_idx = k_layers[l_idx] - 1
            
            # Global bin index
            bin_idx = cum_bins[l_idx] + a_idx
            
            # Increment occupancy for this focal at this frame
            focal_idx = valid_indices[f_id]
            occupancy_data[t, focal_idx, bin_idx] += 1
                
    return occupancy_data, k_layers

def get_min_valids(occ_list:list[np.ndarray]):
    '''Compute the minimum number of valid states across all frames for several occupancy arrays.
    Occupancy arrays have dimensions (n_frames, n_focals, n_bins)'''

    min_valid = np.inf
    for occ in occ_list:
        # Compute number of valid states per frame
        max_per_frame = np.max(occ, axis = 2)
        valid_per_frame = np.sum(max_per_frame < 2, axis = 1)

        # Subsample states to avoid sampling bias
        min_valid = min(min_valid, np.min(valid_per_frame))

    return min_valid

def calculate_entropy(occupancy_data:np.ndarray, k_layers:np.ndarray, n_samples:int):
    """
    occupancy_data: 3D array of shape (n_frames, n_focals, n_bins)
    k_layers: 1D array of number of angular bins in each layer
    Returns: entropy, % of data retained, unique states and their counts, all over time
    """
    num_frames = occupancy_data.shape[0]
    s_t = np.full(num_frames, np.nan)
    ret_t = np.full(num_frames, np.nan)
    unique_t = []
    counts_t = []
    degen_t = []

    # Compute maximum occupancy per frame/focal
    max_per_frame = np.max(occupancy_data, axis = 2)

    # Iterate over time to see if entropy changes over time
    for t in tqdm(range(num_frames)):
        occ_arr = occupancy_data[t,:,:]

        # Exclude instances with occupancies higher than 1 in a bin
        clean_data = occ_arr[max_per_frame[t] < 2, :]
        
        # Subsample states
        state_idcs = np.random.choice(np.arange(clean_data.shape[0]), n_samples, replace = False)

        # Count unique states, treating each grid as a single state
        unique_states, counts = np.unique(clean_data[state_idcs,:], axis=0, return_counts=True)
        
        # Check for degenerate states
        new_ids, new_counts = compute_degenerates(unique_states, counts, k_layers)

        # Calculate Shannon entropy: H = -sum(p * log2(p)) (bits)
        probs = new_counts / np.sum(new_counts)
        s_entropy = entropy(probs, base=2)
        
        retention_pct = (len(clean_data) / len(occ_arr)) * 100

        s_t[t] = s_entropy
        ret_t[t] = retention_pct
        unique_t.append(new_ids)
        counts_t.append(new_counts)
        degen_t.append(len(unique_states) - len(new_ids))
    
    return s_t, ret_t, unique_t, counts_t, degen_t

def get_pca(occupancy_data:np.ndarray):
    """
    occupancy_data: 3D array of shape (n_frames, n_focals, n_bins)
    Returns: occupancy data transformed into pca coordinates, pca object
    """
    # Collapse matrix such that frame index isn't preserved, since we don't need it
    binary_matrix = np.reshape(occupancy_data, (-1, occupancy_data.shape[2]))

    # Exclude configurations that have double or more occupancy
    max_instances = np.max(binary_matrix, axis = 1)
    binary_matrix = binary_matrix[max_instances < 2]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(binary_matrix)

    # Transform macrostates into pca coordinates
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca

def compute_degenerates(unique_states:np.ndarray, counts:np.ndarray, k_layers:np.ndarray):
    """
    unique_states: 2D array of shape (n_unique_states, n_bins)
    counts: 1D array of counts for each state
    k_layers: number of angular bins per layer
    Returns: unique_ids and counts, combining states that are symmetrical across the body axis
    """

    # Determine edges of each layer of bins
    bin_edges = np.concatenate(([0], np.cumsum(k_layers)))

    # Encode states as integers
    unique_ids = [state_to_integer(state) for state in unique_states]
    
    new_ids = []
    new_counts = []
    duplicate_ids = []
    
    for j, state in enumerate(unique_states):

        # Get id for this state
        state_id = unique_ids[j]

        # Check if id in duplicate_states
        if state_id in duplicate_ids: # Skip
            continue

        # Get count for this state
        count = counts[j]

        # Determine flipped state by flipping each layer
        flipped_state = np.array(flatten_deep([np.flip(state[bin_edges[i]:bin_edges[i + 1]]) for i in range(len(bin_edges) - 1)]))

        # Get flipped_id
        flipped_id = state_to_integer(flipped_state)

        if flipped_id in unique_ids:

            # Make sure not to check this duplicate when it comes up
            duplicate_ids.append(flipped_id)

            # Add counts of 'other' to 'state'
            flipped_idx = np.where(unique_ids == flipped_id)[0][0]
            count += counts[flipped_idx]
        
        new_ids.append(state_id)
        new_counts.append(count)

    return np.array(new_ids), np.array(new_counts)

def compute_degenerates_fast(unique_states, counts, k_layers):
    """
    Symmetry folding using lexicographical selection.
    """
    n_unique, n_bins = unique_states.shape
    bin_edges = np.concatenate(([0], np.cumsum(k_layers)))
    
    # 1. Map the indices for a horizontal flip
    flip_idx = np.zeros(n_bins, dtype=int)
    for i in range(len(k_layers)):
        start, end = bin_edges[i], bin_edges[i+1]
        flip_idx[start:end] = np.arange(start, end)[::-1]
    
    flipped_states = unique_states[:, flip_idx]

    # 2. Pick the Canonical State (Lexicographical Choice)
    # We compare rows. For each row, we want a version that is "canonical"
    # Logic: If flipped is "smaller" than original, use flipped.
    # We'll use a loop-free way to find the first index where they differ.
    
    diff = unique_states != flipped_states
    # Find the first column where they differ for each row
    first_diff = np.argmax(diff, axis=1)
    
    # Compare the values at that first difference
    # row_indices is [0, 1, 2, ... n_unique-1]
    row_idx = np.arange(n_unique)
    use_flipped = flipped_states[row_idx, first_diff] < unique_states[row_idx, first_diff]
    
    # Create the canonical array
    canonical_states = unique_states.copy()
    canonical_states[use_flipped] = flipped_states[use_flipped]
    
    # 3. Group and Aggregate using unique rows
    # np.unique with axis=0 is highly optimized C-code
    unique_canon, inverse_indices = np.unique(canonical_states, axis=0, return_inverse=True)
    
    # Sum counts by group
    new_counts = np.bincount(inverse_indices, weights=counts).astype(int)
    
    # 4. Convert to Python 'large' integers for your IDs
    # (Only if you strictly need the 2^70 style IDs)
    powers = 2**np.arange(n_bins, dtype=object)[::-1]
    new_ids = [np.dot(row, powers) for row in unique_canon]

    return np.array(new_ids), new_counts