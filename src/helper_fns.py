'''_____________________________________________________IMPORTS____________________________________________________________'''


import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

# def get_frame_slice(ds: xr.Dataset, start:int = 0, end:int | None = None):
#     '''Return slice of frames from relative frame numbers (0, 1, 2, ...) in absolute frame numbers.'''

#     first_global, last_global = int(ds.frame.min()), int(ds.frame.max())
#     start = max(first_global, start)

#     global_frames = range(start, end or (last_global + 1))

#     assert len(global_frames), f"Start frame must be less than end frame. Got start={start}, end={end or (last_global + 1)}."

#     return global_frames

def get_frame_slice(ds:xr.Dataset, rel_start:int = 0, rel_end:int | None = None, in_function_subsample:int = 1):
    '''Handle complex combinations of subsampling (ds-level or in-function) and relative start frames. Returns slice of frames in absolute numbers, and indices for selected frames.'''

    # Retrieve all absolute frames recorded in dataset. This handles any subsampling that has already occurred.
    ds_frames = ds.frame.values

    # Correct rel_start and rel_end if they are inappropriate
    rel_start = min(rel_start, len(ds_frames) - 1)
    rel_end = min(len(ds_frames), rel_end or len(ds_frames))

    # Subsample (again) if requested
    indcs = np.arange(rel_start, rel_end, in_function_subsample)

    # Finalize splices
    abs_frames = ds_frames[indcs]
    ds_indcs = indcs

    return abs_frames, ds_indcs


def run_lengths(arr, val = None):
    """
    Return lengths of either:
    a) all consecutive non-NaN runs within rows.
    b) all consecutive runs of a specific value within rows.
    Runs do NOT continue across rows.
    """
    arr = np.asarray(arr)

    if val is None:
        valid = ~np.isnan(arr)
    else:
        valid = (arr == val)

    # Pad with False at both ends along frame axis
    padded = np.pad(valid, ((0, 0), (1, 1)), constant_values=False)

    # Find run start and end indices
    diff = np.diff(padded.astype(int), axis=1)

    starts = diff == 1
    ends   = diff == -1

    # Indices of starts and ends
    start_idx = np.where(starts)
    end_idx   = np.where(ends)

    # Run lengths = end - start
    lengths = end_idx[1] - start_idx[1]

    return lengths

def run_lengths_labeled(arr):
    """
    Count lengths of consecutive identical non-nan values. It is assumed that the last label of one row cannot be the first label of the next row.
    """
    x = arr.values.ravel()
    x = x[~np.isnan(x)]

    # If empty:
    if x.size == 0:
        raise ValueError('Array is empty.')

    # Find run boundaries
    changes = np.diff(x) != 0
    idx = np.concatenate(([0], np.where(changes)[0] + 1, [len(x)]))

    # Run lengths:
    lengths = np.diff(idx).astype(int)

    return lengths

def autocorr_fft(x, tau_max: int = int(1e6), normalize=True, demean=True):

    """
    Fast autocorrelation using FFT.
    
    Parameters
    ----------
    x : 1D array
    normalize : bool
        If True, normalize so R(0) = 1
    demean : bool
        Subtract mean before computing
    
    Returns
    -------
    acf : 1D array
        Autocorrelation for lags >= 0
    """
    x = np.asarray(x)
    n = len(x)

    if demean:
        x = x - np.mean(x)

    # Zero-pad to avoid circular correlation
    nfft = 1 << (2*n - 1).bit_length()
    fx = np.fft.fft(x, n=nfft)
    acf = np.fft.ifft(fx * np.conj(fx)).real
    acf = acf[:n]

    if normalize:
        if np.std(x) != 0:
            acf /= acf[0]
        
    return acf[:tau_max]

def list_long_tracklets(da: xr.DataArray, min_tracklet_length: int = 1):
    '''Takes a DataArray of shape (id, frame) and returns a list of valid tracklets (continuous segments of data that are sufficiently long).'''

    # Loop over ids
    all_valid_trajs = []
    for i in range(da.sizes["id"]):
        vals = da.isel(id=i).values
        frames = da["frame"].values

        # Mask out invalid values, e.g. gaps
        mask = ~np.isnan(vals)

        if not np.any(mask):
            continue

        valid_vals = vals[mask]
        valid_frames = frames[mask]

        # Find continuous frame segments
        breaks = np.where(np.diff(valid_frames) != 1)[0] + 1
        segments = np.split(np.arange(len(valid_frames)), breaks)

        # Add to list of all segments if long enough
        for seg in segments:
            if len(seg) >= min_tracklet_length:
                all_valid_trajs.append(valid_vals[seg])

    print('Number of eligible tracklets:', len(all_valid_trajs), flush=True)

    return all_valid_trajs

def within_tracklet_pos(arr: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    arr : np.ndarray
        Shape (id, frame), NaN-separated tracklets within each row.

    Returns
    -------
    rel_pos : np.ndarray
        Same shape as arr, with positions relative to the end of each tracklet.
        (e.g. -10, ..., -1 for the last point in a tracklet)
    """

    arr = np.asarray(arr)
    n_id, n_frame = arr.shape

    out = np.full_like(arr, np.nan, dtype=float)

    for i in range(n_id):
        valid = ~np.isnan(arr[i])

        if not np.any(valid):
            continue

        # Find tracklet boundaries
        diff = np.diff(valid.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if valid[0]:
            starts = np.r_[0, starts]
        if valid[-1]:
            ends = np.r_[ends, n_frame]

        # Assign relative positions
        for s, e in zip(starts, ends):
            length = e - s
            out[i, s:e] = np.arange(-length + 1, 1)

    return out

def get_metric_density(ds: xr.Dataset, pos_name: str, radius: float) -> xr.Dataset:
    """
    Compute local density (number of neighbours within radius) per id per frame.
    """

    x = ds[f"x_{pos_name}"].values
    y = ds[f"y_{pos_name}"].values
    density = np.full(x.shape, np.nan, dtype=float)

    for f in range(x.shape[1]):
        xf = x[:, f]
        yf = y[:, f]

        mask = ~np.isnan(xf) & ~np.isnan(yf)
        if mask.sum() <= 1:
            continue

        pts = np.column_stack((xf[mask], yf[mask]))

        tree = cKDTree(pts)
        counts = tree.query_ball_tree(tree, r=radius)

        dens = np.array([len(c) - 1 for c in counts])
        density[mask, f] = dens

    ds = ds.copy()
    ds[f"density_r_{radius}"] = (("id", "frame"), density)

    return ds

def flatten_deep(nested):
    result = []
    for item in nested:
        if isinstance(item, list) or isinstance(item, np.ndarray): # or isinstance(np.float64):
            result.extend(flatten_deep(item))
        else:
            result.append(item)
    return result

def state_to_integer(state:np.ndarray):
    return state.dot(1 << np.arange(state.size)[::-1]) # 1 << (bit-wise shift left) essentially turns state into 2**element for each element

def integer_to_state(n:int, state_len:int):
    """
    Converts integer n into a binary array of specified length.
    Uses bit-shifting for high performance.
    """
    # Create an array of bit positions: [2^7, 2^6, ..., 2^0] for length 8
    powers = np.arange(state_len - 1, -1, -1)
    
    # Right shift n by each power and check the last bit
    return (n >> powers) & 1

def voronoi_finite_polygons_2d(vor, arena_x: int, arena_y: int, arena_radius: float):
    """
    Reconstruct infinite Voronoi regions to finite regions.
    Returns:
        regions: list of vertex indices
        vertices: array of vertices
    """

    arena_center = np.array([arena_x, arena_y])

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    # map ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - arena_center, n)) * n

            far_point = vor.vertices[v2] + direction * arena_radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

import matplotlib.pyplot as plt

def clip_voronoi_region(vertices: np.ndarray, arena_center:np.ndarray, arena_radius:float, rads_per_vertex:float = 0.05):

    # Find invalid vertices
    invalid_mask = np.linalg.norm(vertices - arena_center, axis = 1) > arena_radius # (n_verts,)

    # Reject regions with ONLY excluded vertices
    if np.sum(invalid_mask) == len(vertices):
        return None
    
    # Reject regions with less than 3 vertices
    if len(vertices) < 3:
        return None

    # Order vertices around center of region
    center = np.mean(vertices, axis = 0)
    angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])
    order = np.argsort(angles)
    vertices = vertices[order]
    invalid_mask = invalid_mask[order]

    old_vertices = vertices.copy()

    # If all vertices valid and there are enough vertices after clipping infinite points, make polygon
    if np.sum(invalid_mask) == 0:
        if len(invalid_mask) == 3:
            vertices = np.concatenate([vertices, [vertices[0,:]]], axis = 0) # Add first vertex to end, since shapely requires 4 vertices to construct a polygon, for some reason
        return vertices

    # Find transitions between vertices inside and outside of the boundary
    edges = np.abs(np.diff(invalid_mask).astype(int)) # 1indicates that this index as well as the following correspond to an inside-outside ridge line
    trans_idcs = np.where(edges != 0)[0]
    
    # Handle wrap-around ridges
    if len(trans_idcs) < 2:
        edges = np.concat([edges, [1]]) # Missing entry/exit ridge line must wrap around
    else:
        edges = np.concat([edges, [0]])

    def truncate_ridge_line(vert_in: np.ndarray, vert_out: np.ndarray, arena_center: np.ndarray, arena_radius: float):
        '''Returns modified position of invalid vertex such that it is on the boundary but along the same ridge line.'''

        # Compute the line between the two vertices
        if vert_in[0] != vert_out[0]:
            m = (vert_in[1] - vert_out[1])/(vert_in[0] - vert_out[0]) # Get slop of line (1,)
            b = vert_in[1] - m*vert_in[0] # Find intercept of the line (1,)

            # Use the quadratic formula to compute the intercept between the line and the arena boundary
            A = m**2 + 1
            B = 2*(m*(b - arena_center[1]) - arena_center[0])
            C = arena_center[0]**2 + (b - arena_center[1])**2 - arena_radius**2
            determinant = B**2 - 4*A*C

            # Check determinant properties (if determinant is 0, this shouldn't matter downstream)
            no_intercepts = determinant < 0

            assert not no_intercepts, f"Both of the vertices are outside of the arena: {vert_in, vert_out, determinant}."
            
            # Find both solutions
            intercept_a = np.array([(-B + np.sqrt(determinant))/(2*A), m*(-B + np.sqrt(determinant))/(2*A) + b]).T # (x/y,)
            intercept_b = np.array([(-B - np.sqrt(determinant))/(2*A), m*(-B - np.sqrt(determinant))/(2*A) + b]).T # (x/y,)

        else: # The points are on the same vertical line
            intercept_a = np.array([vert_in[0], arena_center[1] + np.sqrt(arena_radius**2 - (vert_in[0] - arena_center[0])**2)]).T # (x/y,)
            intercept_b = np.array([vert_in[0], arena_center[1] - np.sqrt(arena_radius**2 - (vert_in[0] - arena_center[0])**2)]).T # (x/y,)

        # Find solution that's closest to invalid vertex
        dists_a = np.linalg.norm(intercept_a - vert_out) # (1,)
        dists_b = np.linalg.norm(intercept_b - vert_out) # (1,)

        correct_sol = dists_a > dists_b # Takes the index of the minimum of the two arrays

        return [intercept_a, intercept_b][correct_sol.astype(int)]

    # Find intersections between each inside-outside ridge line and the boundaries
    modified_vertices = []
    for i in range(len(vertices)):
        if edges[i]:
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]

            if invalid_mask[i]:
                modified_vertices.append(truncate_ridge_line(v2, v1, arena_center, arena_radius)) # v1 is invalid
            else:
                modified_vertices.append(truncate_ridge_line(v1, v2, arena_center, arena_radius)) # v2 is invalid

    def generate_boundary_verts(vert0:np.ndarray, vert1:np.ndarray, arena_center:np.ndarray, arena_radius:float, rads_per_vert:float):
        '''Generates points between two boundary vertices to smooth curve along boundary.'''

        # Find angles of two initial boundary vertices relative to arena center
        ang0 = np.arctan2(vert0[1] - arena_center[1], vert0[0] - arena_center[0]) % (2*np.pi) # Shifting to [0, 2*np.pi) to avoid wrapping error
        ang1 = np.arctan2(vert1[1] - arena_center[1], vert1[0] - arena_center[0]) % (2*np.pi)

        # Make sure distance between angles is minimal
        if abs(ang1 - ang0) > np.pi:
            temp_angs = np.array([ang0, ang1])
            over_idx = temp_angs > np.pi
            temp_angs[over_idx] = (temp_angs[over_idx] + np.pi)%(2*np.pi) - np.pi # Convert back to [-np.pi/2, np.pi/2)
            ang0 = temp_angs[0]
            ang1 = temp_angs[1]

        # Make a range of angles between the two
        angs = np.linspace(min(ang0, ang1), max(ang0, ang1), 2 + round(abs(ang0 - ang1)/rads_per_vert)) # Using min and max assumes that the smaller arc is always the correct one, which in these experiments is a good assumption but may need to be adapted for different systems

        # Convert these angles into points on the boundary
        points = np.array([arena_radius*np.cos(angs) + arena_center[0], arena_radius*np.sin(angs) + arena_center[1]]).T # (n_points, x/y)

        return points

    # Generate points on boundary between two intercepts to make curve smoother
    new_points = generate_boundary_verts(modified_vertices[0], modified_vertices[1], arena_center, arena_radius, rads_per_vertex)

    # Add new points to original valid points
    vertices = np.concat([vertices[~invalid_mask], new_points], axis = 0)

    # Re-order vertices
    center = np.mean(vertices, axis = 0)
    angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])
    order = np.argsort(angles)
    vertices = vertices[order]

    if len(np.unique(vertices, axis = 0)) < len(vertices):
        print('Warning: redundant vertices for polygon with more than 3 sides.')
        print(old_vertices)

    # fig, ax = plt.subplots()

    # ax.scatter(old_vertices[:,0], old_vertices[:,1])
    # ax.scatter(vertices[:, 0], vertices[:, 1])
    # cir = plt.Circle(arena_center, arena_radius, alpha = 0.3, color= 'g')
    # ax.add_patch(cir)
    # ax.set_xlim([3700, 4400])
    # ax.set_ylim([6700,  6950])
    # plt.savefig('ridge_line_clip_test.png')

    return vertices

