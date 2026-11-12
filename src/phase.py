'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
from helper_fns import clip_voronoi_region
from shapely.geometry import Polygon
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


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

def calculate_nematic_order(thetas):
    """
    Calculates the nematic order of a group.
    Returns 1.0 for perfect axial alignment and np.nan for an empty array.
    """
    if len(thetas) == 0:
        return np.nan

    # Double angles so opposite orientations are equivalent
    sum_cos = np.sum(np.cos(2*thetas))
    sum_sin = np.sum(np.sin(2*thetas))

    nematic_order = np.sqrt(sum_cos**2 + sum_sin**2) / len(thetas)

    return nematic_order

def local_env(pos_valid_t:np.ndarray, thetas_valid_t:np.ndarray, nbr_type:str, nbr_param:float | int | None, arena_center:np.ndarray, arena_radius:float):

    # Check if there are any valid positions
    if len(pos_valid_t) < 1:
        return np.array([]), np.array([])

    # Find neighbours
    if nbr_type == 'metric':

        # Create ball tree for valid positions
        tree = BallTree(pos_valid_t)

        # Find all ids within nbr_param of each other
        indcs = tree.query_radius(pos_valid_t, r = nbr_param) # list of arrays of neighbours that's n_valid_ids long

        # Remove self from indcs list
        indcs = [indcs[i][nbrs != i] for i, nbrs in enumerate(indcs)]

        # Compute local density
        density_valid_t = np.array([len(nbrs) for nbrs in indcs])/(np.pi*nbr_param**2)

    elif nbr_type == 'topo':

        # Create ball tree for valid positions
        tree = BallTree(pos_valid_t)

        # Find all nbr_param-nearest neighbours for each valid id and their distances
        dists, indcs = tree.query(pos_valid_t, k = nbr_param + 1)

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

            if poly.is_empty or poly.area < 9e-5: # Area threshold is necessary due to invalid neighbourhoods (200 for px, 9e-5 for m)
                continue

            areas[i] = poly.area

        density_valid_t = 1/areas

    else:
        raise ValueError(f"Invalid nbr_type: {nbr_type}. Must be one of 'metric', 'topo', or 'voronoi'.")

    # Compute local polarization and nematic order, EXCLUDING focal individual
    polarizations_valid_t = np.array([calculate_polarization(thetas_valid_t[nbrs]) for nbrs in indcs])
    nematic_orders_valid_t = np.array([calculate_nematic_order(thetas_valid_t[nbrs]) for nbrs in indcs])

    # Remove individuals whose densities are invalid
    invalid = np.isnan(density_valid_t)
    polarizations_valid_t[invalid] = np.nan
    nematic_orders_valid_t[invalid] = np.nan

    return density_valid_t, polarizations_valid_t, nematic_orders_valid_t

def get_local_env(ds:xr.Dataset, nbr_type:str, nbr_param:float | int | None, arena_center:np.ndarray, arena_radius:float, x_var:str = 'x_raw', y_var:str = 'y_raw', density_factor:float = 1, n_jobs:int = -1, chunk_size:int = 250):
    '''
    Computes neighbour density, polarization, and nematic order locally.
    
    Parameters:
    ds: xr.Dataset containing 'centroid_x', 'centroid_y', 'theta'
    nbr_type: str indicating which type of neighbourhood to compute density and polarization on; one of 'metric', 'topo'
    nbr_param: int, float, or None parameter for finding neighbours; radius, k, or None for 'metric', 'topo', respectively
    arena_params: list of [arena_center_x, arena_center_y, arena_radius]
    '''

    # Define position and theta array to save time from accessing ds
    positions = np.stack([ds[x_var].transpose('frame', 'id').values, ds[y_var].transpose('frame', 'id').values]) # (2, n_frames, max_ids)
    thetas = ds['theta'].transpose('frame', 'id').values # (n_frames, max_ids)

    # Filter out detections outside of arena
    dist_from_center = np.sqrt((positions[0] - arena_center[0])**2 + (positions[1] - arena_center[1])**2) # (n_frames, max_ids)
    outside_arena_mask = dist_from_center > arena_radius

    # Define valid mask
    valid_mask = (~np.isnan(positions).any(axis = 0)) & (~np.isnan(thetas)) & (~outside_arena_mask) # (n_frames, max_ids)

    # Prep results
    densities = np.full(thetas.shape, np.nan)
    polarizations = np.full(thetas.shape, np.nan)
    nematic_orders = np.full(thetas.shape, np.nan)

    def process_frames(frame_indcs):
        '''Compute local environments for a block of frames.'''

        out = []

        for f in frame_indcs:

            # Select valid position and theta values for this frame
            mask = valid_mask[f]
            pos_valid_t = positions[:, f, mask].T
            thetas_valid_t = thetas[f, mask]

            if np.sum(mask) < 1:
                result = None
            else:
                result = local_env(pos_valid_t, thetas_valid_t, nbr_type, nbr_param, arena_center, arena_radius)

            out.append((f, mask, result))

        return out

    # Split frames into blocks to reduce multiprocessing overhead
    n_frames = positions.shape[1]
    frame_chunks = [np.arange(start, min(start + chunk_size, n_frames)) for start in range(0, n_frames, chunk_size)]

    # Compute local environments iterating over chunks of frames in parallel
    with tqdm_joblib(tqdm(desc='Compute local envs. (batches)', total=len(frame_chunks))):
        results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(process_frames)(chunk) for chunk in frame_chunks)

    # Flatten worker outputs
    results = [result for chunk in results for result in chunk]

    # Collect results in original frame order
    for f, mask, result in results:

        if result is None:
            continue

        # Check if there are sufficient arrays that are non-zero
        if len(result) < 3 or np.sum([a.size == 0 for a in result]) > 0:
            continue

        density_t, polarization_t, nematic_order_t = result

        densities[f, mask] = density_t*density_factor
        polarizations[f, mask] = polarization_t
        nematic_orders[f, mask] = nematic_order_t
    
    # Update ds
    ds[f'density_{nbr_type}_{nbr_param}'] = (('frame', 'id'), densities)
    ds[f'polarization_{nbr_type}_{nbr_param}'] = (('frame', 'id'), polarizations)
    ds[f'nematic_order_{nbr_type}_{nbr_param}'] = (('frame', 'id'), nematic_orders)

    return ds

def local_density_gradient(ds: xr.Dataset, *, x_var: str = "x_raw", y_var: str = "y_raw", radius: float, bandwidth: float | None = None) -> np.ndarray:
    """
    Estimate the local density-gradient vector for each id and frame.

    For each focal individual, sum the gradients of Gaussian kernels centered
    on the other individuals within `radius`.

    Parameters
    ----------
    ds : xr.Dataset
        Coordinates must have `id` and `frame` dimensions.
    x_var, y_var : str
        Position variables, in the same length units as `radius`.
    radius : float
        Metric radius for selecting neighbors.
    bandwidth : float, optional
        Gaussian kernel bandwidth. Defaults to radius / 2.

    Returns
    -------
    gradient : ndarray, shape (2, n_ids, n_frames)
        `gradient[0]` is the x component; `gradient[1]` is the y component.
        Invalid focal positions and focal individuals with no neighbors are NaN.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")

    if bandwidth is None:
        bandwidth = radius / 2
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    x = ds[x_var].transpose("id", "frame").values
    y = ds[y_var].transpose("id", "frame").values
    n_ids, n_frames = x.shape

    gradient = np.full((2, n_ids, n_frames), np.nan, dtype=float)
    h2 = bandwidth**2

    for frame_i in range(n_frames):
        valid_ids = np.flatnonzero(np.isfinite(x[:, frame_i]) & np.isfinite(y[:, frame_i]))
        
        if len(valid_ids) < 2:
            continue

        positions = np.column_stack((x[valid_ids, frame_i], y[valid_ids, frame_i]))
        tree = BallTree(positions)
        neighbors = tree.query_radius(positions, r=radius)

        for local_i, local_neighbors in enumerate(neighbors):

            # Remove the focal individual itself.
            local_neighbors = local_neighbors[local_neighbors != local_i]
            
            if local_neighbors.size == 0:
                continue

            # Displacement vectors from focal to neighbors.
            delta = positions[local_neighbors] - positions[local_i]
            r2 = np.sum(delta**2, axis=1)

            # Gradient of the summed 2D Gaussian kernels at the focal position.
            weights = np.exp(-0.5 * r2 / h2) / (2 * np.pi * h2)
            grad = np.sum(delta * (weights / h2)[:, None], axis=0)

            full_id = valid_ids[local_i]
            gradient[:, full_id, frame_i] = grad

    return gradient

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
