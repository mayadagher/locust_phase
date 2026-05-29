'_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
from helper_fns import clip_voronoi_region
from shapely.geometry import Polygon


from data_handling import *

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def calculate_polarization(thetas):
    """
    Calculates the polarization (alignment) of a group.
    Returns 1.0 for perfect alignment, 0.0 for perfectly random/opposed, and np.nan for an empty array.
    """
    if len(thetas) == 0:
        return np.nan
    
    # Sum the unit vectors (cos(theta), sin(theta))
    sum_cos = np.sum(np.cos(thetas))
    sum_sin = np.sum(np.sin(thetas))
    
    # Calculate the magnitude of the average vector
    polarization = np.sqrt(sum_cos**2 + sum_sin**2) / len(thetas)
    
    return polarization

def local_env(pos_valid_t:np.ndarray, thetas_valid_t:np.ndarray, nbr_type:str, nbr_param:float | int | None, arena_center:np.ndarray, arena_radius:float):

    # Check if there are any valid positions
    if len(pos_valid_t) < 1:
        return np.array([]), np.array([])

    # Create ball tree for valid positions
    tree = BallTree(pos_valid_t)

    # Find neighbours
    if nbr_type == 'metric':

        # Find all ids within nbr_param of each other
        indcs = tree.query_radius(pos_valid_t, r = nbr_param) # list of arrays of neighbours that's n_valid_ids long

        # Remove self from indcs list
        indcs = [indcs[i][nbrs != i] for i, nbrs in enumerate(indcs)]

        # Compute local density
        density_valid_t = np.array([len(nbrs) for nbrs in indcs])/(np.pi*nbr_param**2)

    elif nbr_type == 'topo':

        # Find all nbr_param-nearest neighbours for each valid id and their distances
        dists, indcs = tree.query(pos_valid_t, k = nbr_param)

        # Remove self from indcs and dists lists
        indcs = [indcs[i][nbrs != i] for i, nbrs in enumerate(indcs)]
        dists = [dists[i][nbrs != i] for i, nbrs in enumerate(dists)]

        # Compute local density
        density_valid_t = nbr_param/np.array([np.pi*np.max(dist)**2 for dist in dists])

    elif nbr_type == 'voronoi':
        
        # Compute Voronoi tessellation
        vor = Voronoi(pos_valid_t)

        # Compute neighbour relationships from Voronoi ridges
        nbrs = {i: set() for i in range(len(pos_valid_t))}
        for i1, i2 in vor.ridge_points:
            nbrs[i1].add(i2)
            nbrs[i2].add(i1)
        indcs = [sorted(list(v)) for v in nbrs.values()]

        # Compute areas of each Voronoi neighbourhood
        areas = np.full(len(pos_valid_t), np.nan)
        for i in range(len(pos_valid_t)):

            # Get correct region index
            region_idx = vor.point_region[i]
            reg = vor.regions[region_idx]

            if len(reg) == 0:
                continue

            # Clip and order vertices
            vertices = vor.vertices[np.array(reg)[(np.array(reg) != -1).astype(bool)].astype(int)]
            poly = Polygon(clip_voronoi_region(vertices, arena_center, arena_radius, 0.05))

            if poly.is_empty:
                continue

            areas[i] = poly.area

        density_valid_t = np.array([len(nbrs) for nbrs in indcs])/areas
    
    else:
        raise ValueError(f"Invalid nbr_type: {nbr_type}. Must be one of 'metric', 'topo', or 'voronoi'.")

    # Compute local polarization, EXCLUDING focal individual
    polarizations_valid_t = np.array([calculate_polarization(thetas_valid_t[nbrs]) for nbrs in indcs])

    return density_valid_t, polarizations_valid_t

def get_local_env(ds:xr.Dataset, nbr_type:str, nbr_param:float | int | None, arena_center:np.ndarray, arena_radius:float):
    '''
    Computes neighbour density and polarization locally.
    
    Parameters:
    ds: xr.Dataset containing 'centroid_x', 'centroid_y', 'theta'
    nbr_type: str indicating which type of neighbourhood to compute density and polarization on; one of 'metric', 'topo'
    nbr_param: int, float, or None parameter for finding neighbours; radius, k, or None for 'metric', 'topo', respectively
    arena_params: list of [arena_center_x, arena_center_y, arena_radius]
    '''

    # Define position and theta array to save time from accessing ds
    positions = np.stack([ds['centroid_x'], ds['centroid_y']]) # (2, n_frames, max_ids)
    thetas = ds['theta'].values # (n_frames, max_ids)

    # Filter out detections outside of arena
    dist_from_center = np.sqrt((positions[0] - arena_center[0])**2 + (positions[1] - arena_center[1])**2) # (n_frames, max_ids)
    outside_arena_mask = dist_from_center > arena_radius

    # Define valid mask
    valid_mask = (~np.isnan(positions).any(axis = 0)) & (~np.isnan(thetas)) & (~outside_arena_mask) # (n_frames, max_ids)

    # Prep results
    densities = np.full(thetas.shape, np.nan)
    polarizations = np.full(thetas.shape, np.nan)

    # Iterate over frames
    n_frames = positions.shape[1]
    print('Compute local environments at each frame.')
    for f in tqdm(range(n_frames)):

        # Find valid positions and thetas for this frame
        pos_valid_t = positions[:, f, valid_mask[f]].T # (n_valid, 2)
        thetas_valid_t = thetas[f, valid_mask[f]] # (n_valid, )

        # Get densities and polarizations for valid ids
        density_valid_t, polarizations_valid_t = local_env(pos_valid_t, thetas_valid_t, nbr_type, nbr_param, arena_center, arena_radius)

        # Update results
        densities[f, valid_mask[f]] = density_valid_t
        polarizations[f, valid_mask[f]] = polarizations_valid_t
    
    # Update ds
    ds[f'density_{nbr_type}_{nbr_param}'] = (('frame', 'id'), densities)
    ds[f'polarization_{nbr_type}_{nbr_param}'] = (('frame', 'id'), polarizations)

    return ds

def compute_g_of_r(points, arena_size, dr=10, max_r=1000):
    """
    Computes the radial distribution function g(r).
    points: (N, 2) array of positions for a single frame.
    arena_size: Area of the arena (used for density normalization).
    dr: Bin width in pixels/cm.
    max_r: Maximum distance to consider.
    """
    n_ids = len(points)
    if n_ids < 2: return None, None
    
    # Global density (rho)
    rho = n_ids / arena_size
    
    # Calculate all pair distances
    dists = pdist(points)
    
    # Bin the distances
    bins = np.arange(0, max_r + dr, dr)
    counts, _ = np.histogram(dists, bins=bins)
    
    # Normalize by the area of the concentric rings
    # g(r) = (counts) / (rho * area_of_ring * n_ids)
    r_centers = (bins[:-1] + bins[1:]) / 2
    # We multiply by 2 because pdist only counts each pair once
    ring_areas = 2 * np.pi * r_centers * dr
    
    g_r = (2 * counts) / (rho * ring_areas * n_ids)
    
    return r_centers, g_r



# def voronoi_correlation_analysis(ds:xr.Dataset, param:str, arena_center:np.ndarray, arena_radius:float, focal_index: int, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1, tolerance: float = 1e-6) -> tuple[list[Polygon], dict[int, int]]:
#     '''
#     Compute correlation length of variable using Voronoi layers as distance metric.
 
#     Parameters
#     ----------
#     ds:
#         xr.Dataset with coordinates (frame, id) and containing variable columns including var
#     param:
#         Variable column in xr.Dataset on which we want to do correlation analysis.
#     arena_center:
#         Numpy array with the coordinates of the arena center.
#     arena_radius:
#         Length of the arena radius.
#     focal_index:
#         Index into *points* of the focal animal / cell.
#     start_frame:
#         Relative index of first frame to be included in the splice for this computation.
#     end_frame:
#         Relative index of last frame to be included in the splice for this computation.
#     subsample:
#         Step size for computing frame splice, useful for covering wide temporal coverage without overwhelming long compute.
#     tolerance:
#         Passed to ``build_adjacency_graph``.
 
#     Returns
#     -------
#     cells : list[Polygon]
#         Clipped Voronoi cells.
#     layers : dict[int, int]
#         Layer assignment for each cell (−1 = unreachable).
#     '''

#     # Define position array to save time from accessing ds
#     positions = np.stack([ds['centroid_x'], ds['centroid_y']]) # (2, n_frames, max_ids)
#     z = ds[param].values # (n_frames, max_ids)

#     # Filter out detections outside of arena
#     dist_from_center = np.sqrt((positions[0,:,:] - arena_center[0])**2 + (positions[1,:,:] - arena_center[1])**2) # (n_frames, max_ids)
#     outside_arena_mask = dist_from_center > arena_radius

#     # Define valid mask
#     valid_mask = (~np.isnan(positions).any(axis = 0)) & (~np.isnan(z)) & (~outside_arena_mask) # (n_frames, max_ids)

#     # Get frames
#     abs_frames, ds_idcs = get_frame_slice(ds, start_frame, end_frame, subsample)

#     # Iterate over frames
#     for f in ds_idcs:
#         # Filter valid positions and z values
#         valid_positions_t = positions[:, f, valid_mask[f]]  # (2, n_ids)

#         if valid_positions_t.shape[1] < 3:
#             continue

#         # Create tessellation
#         vor = Voronoi(valid_positions_t.T)

#         polys_t = []
#         z_t = []

#         # Iterate over each point to ensure order is conserved
#         for point_idx, z_val in enumerate(z[f, valid_mask[f]]):
#             region_idx = vor.point_region[point_idx]
#             region = vor.regions[region_idx]

#             # Exclude regions with no points
#             if len(region) == 0:
#                 continue

#             # Clip and order vertices
#             vertices = vor.vertices[np.array(region)[(np.array(region) != -1).astype(bool)].astype(int)]
#             poly = Polygon(clip_voronoi_region(vertices, arena_center, arena_radius, 0.05))

#             if poly.is_empty:
#                 continue

#             # Append to lists so that indices of polys and zs align
#             polys_t.append(poly)
#             z_t.append(z_val)

#         # Build adjacency graph between vertices
#         graph = build_adjacency_graph(polys_t, tolerance=tolerance)

#         # Compute Voronoi layers for all individuals
#         layers_t:list[dict[int, int]] = []
#         corrs_t
#         for focal_index in range(len(polys_t)):
#             layers = compute_layers(graph, focal_index)

#             # Compute correlations binned by Voronoi distance
#             corrs: dict[int, list] = 


#     return cells, layers

# graph = {0: set([1, 2, 3]), 1: set([0, 2]), 2: set([0, 1, 4]), 3: set([0, 4]), 4: set([2, 3])}
# layers = compute_layers(graph, 1)
# print(layers)        

