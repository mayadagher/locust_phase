'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from shapely.geometry import Polygon
from tqdm.auto import tqdm
from correlation_length import build_adjacency_graph, compute_voronoi_layers
from scipy.spatial import Voronoi
import collections
from collections import defaultdict
from scipy.signal import find_peaks
from dataclasses import dataclass
from helper_fns import get_frame_slice, clip_voronoi_region, fft_timeseries
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
from data_handling import cluster_stats_to_h5, cluster_ids_to_h5
from memory_profiler import profile
from pathlib import Path
import h5py
import os

'''_____________________________________________________CLASSES____________________________________________________________'''

class Cluster():

    def __init__(self, real_ids: np.ndarray[int], compact_ids: np.ndarray[int], centroids: list[np.ndarray], mean_theta: float, theta_arr: np.ndarray[float], area_list: list[float], density_arr: np.ndarray[float],
                 polarization_arr: np.ndarray[float], all_layers: dict[int, dict[int, int]], v_arr:np.ndarray|None = None):

        self.ids = real_ids
        self.mean_theta = mean_theta
        self.thetas = theta_arr
        self.areas = area_list
        self.densities = density_arr
        self.polarizations = polarization_arr
        self.centroids = centroids
        self.velocities = v_arr

        # Store only direct neighbours (layer == 1) for intra-cluster BFS
        id_set = set(compact_ids)
        self.graph: dict[int, set[int]] = {cell_id: {nbr for nbr, dist in all_layers[cell_id].items() if dist == 1 and nbr in id_set} for cell_id in compact_ids}
                
        self.centrality = self._compute_centrality(compact_ids)
        centroid_idx = np.argmin(self.centrality)
        self.centroid_id = self.ids[centroid_idx]
        self._centroid_compact_id = compact_ids[centroid_idx]
        centroid = centroids[np.argmin(self.centrality)]
        self.centroid = np.array([centroid.x, centroid.y])
        self.layers = self.centroid_distance_array(compact_ids)
        self.max_layer = int(np.nanmax(self.layers))

    def _bfs_distances(self, source_id: int) -> dict[int, int]:
        """
        BFS from source_id over the intra-cluster graph.
        Returns {cell_id: distance} for all reachable cells (excluding source).
        """
        visited = {source_id: 0}
        queue = collections.deque([source_id])

        while queue:
            current = queue.popleft()
            for nbr in self.graph[current]:
                if nbr not in visited:
                    visited[nbr] = visited[current] + 1
                    queue.append(nbr)

        del visited[source_id]
        return visited

    def _compute_centrality(self, compact_ids) -> np.ndarray:
        """
        For each cell, compute its mean topological distance to all other cells
        via BFS over the intra-cluster graph.
        Returns an array (aligned with self.ids) where lower = more central.
        """
        mean_distances = np.zeros(len(compact_ids))

        for i, cell_id in enumerate(compact_ids):
            distances = self._bfs_distances(cell_id)
            dists = list(distances.values())
            mean_distances[i] = np.mean(dists) if dists else 0.0

        return mean_distances

    def centroid_distance_array(self, compact_ids) -> np.ndarray:
        """
        Returns an array (aligned with self.ids) of each cell's topological
        distance from the centroid. Unreachable cells get np.nan.
        """
        centroid_dists = self._bfs_distances(self._centroid_compact_id)
        # centroid itself is distance 0
        centroid_dists[self._centroid_compact_id] = 0
        return np.array([centroid_dists.get(cell_id, np.nan) for cell_id in compact_ids])

    # --- Analytical methods unchanged below ---

    def polarization_vs_centrality(self) -> tuple[np.ndarray, np.ndarray]:
        return self.centrality, self.polarizations

    def density_vs_centrality(self) -> tuple[np.ndarray, np.ndarray]:
        return self.centrality, self.densities

    def shell_medians(self, values: np.ndarray) -> dict[int, float]:
        dist_arr = self.centroid_distance_array()
        shells = {}
        for dist, val in zip(dist_arr, values):
            if np.isnan(dist):
                continue
            shells.setdefault(int(dist), []).append(val)
        return {d: np.median(v) for d, v in sorted(shells.items())}

    def size(self) -> int:
        return len(self.ids)

    def median_polarization(self) -> float:
        return float(np.median(self.polarizations))
    
    def var_polarization(self) -> float:
        return float(np.var(self.polarizations))

    def median_density(self) -> float:
        return float(np.median(self.densities))
    
    def var_density(self) -> float:
        return float(np.var(self.densities))

    def total_area(self) -> float:
        return np.sum(self.areas)
    
    def theta_var(self) -> float:
        """
        Circular variance of an array of angles in radians.
        Returns a value in [0, 1], where 0 = all angles identical, 1 = maximally dispersed.
        """
        return 1 - np.abs(np.mean(np.exp(1j * self.thetas)))
    
@dataclass
class FrameClusterStats:
    """All cluster-level statistics for one frame, ready for aggregation."""
    abs_frame:      int
    centroids: np.ndarray
    areas:          np.ndarray
    ns:             np.ndarray
    median_pols:      np.ndarray
    var_pols:       np.ndarray
    median_densities: np.ndarray
    var_densities:  np.ndarray
    mean_thetas:    np.ndarray
    var_thetas:     np.ndarray
    p_by_layer:     np.ndarray                 # (n_clusters, max_layer+1), nan-padded
    d_by_layer:     np.ndarray
    p_from_edge:    np.ndarray                 # flipped version
    d_from_edge:    np.ndarray
'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def find_clusters(all_polarizations: np.ndarray, all_thetas: np.ndarray, all_densities: np.ndarray, all_layers: dict[int, dict[int, int]], polys: list[Polygon],
                  pol_thresh: float = 0.8, angle_thresh: float = np.pi/4, v_thresh: float = 0.01, min_cluster_size: int = 2, all_velocities: np.ndarray | None = None, valid_indices: np.ndarray | None = None):
    
    def angular_diff(a: float, b: float) -> float:
        """Smallest angle between two headings, result in [0, π]."""
        diff = abs(a - b) % (2 * np.pi)
        return diff if diff <= np.pi else 2 * np.pi - diff

    def add_to_cluster( id: int, cluster_mean_theta: float, cluster: list[int], visited_this_bfs: set[int]) -> tuple[list[int], set[int], float]:

        layer_dict = all_layers[id]
        nbrs = np.array(list(layer_dict.keys()))[np.array(list(layer_dict.values())) == 1]

        # Neighbours must be: polarized, optionally active, unassigned globally, not yet seen in this BFS
        if all_velocities is not None:
            # Do not exclude ids that have nan velocities, but exclude non-moving ids
            candidate_nbrs = [int(i) for i in nbrs if all_polarizations[i] >= pol_thresh and (all_velocities[i] >= v_thresh or np.isnan(all_velocities[i])) and i not in assigned and i not in visited_this_bfs]
        else:
            candidate_nbrs = [int(i) for i in nbrs if all_polarizations[i] >= pol_thresh and i not in assigned and i not in visited_this_bfs]

        # Among candidates, keep only those aligned with the cluster's current mean heading
        aligned_nbrs = [i for i in candidate_nbrs if angular_diff(all_thetas[i], cluster_mean_theta) <= angle_thresh]

        new_cluster_nbrs = [i for i in aligned_nbrs if i not in cluster]
        cluster += new_cluster_nbrs
        visited_this_bfs.update(new_cluster_nbrs)

        # Update cluster mean heading using circular mean
        if new_cluster_nbrs:
            cluster_mean_theta = np.arctan2(np.mean(np.sin(all_thetas[cluster])), np.mean(np.cos(all_thetas[cluster])))

        # Continue breadth-first-search for newly added cluster members (recursive)
        for i in new_cluster_nbrs:
            cluster, visited_this_bfs, cluster_mean_theta = add_to_cluster(i, cluster_mean_theta, cluster, visited_this_bfs)

        return cluster, visited_this_bfs, cluster_mean_theta

    assigned: set[int] = set()   # cells that have been assigned to a completed cluster
    all_clusters = []
    all_cluster_mean_thetas = []

    for id in all_layers.keys():

        # Skip if below polarization threshold or already in a cluster
        if all_polarizations[id] < pol_thresh or id in assigned:
            continue

        if all_velocities is not None: # Do not exclude nans, but exclude non-moving locusts
            if np.isfinite(all_velocities[id]) and (all_velocities[id] < v_thresh):
                continue

        cluster = [id]
        visited_this_bfs = {id}
        seed_theta = all_thetas[id] # Theta of first individual used as cluster mean theta until more individuals added

        cluster, visited_this_bfs, cluster_mean_theta = add_to_cluster(id, seed_theta, cluster, visited_this_bfs)

        # Only mark cells as assigned once the cluster is complete
        assigned.update(cluster)
        all_clusters.append(cluster)
        all_cluster_mean_thetas.append(cluster_mean_theta)

    final_clusters = []
    for i, cluster in enumerate(all_clusters):
        if len(cluster) >= min_cluster_size:
            real_ids = None if valid_indices is None else valid_indices.astype(int)[cluster]
 
            if all_velocities is not None:
                final_clusters.append(Cluster(real_ids, cluster, [polys[c].centroid for c in cluster], all_cluster_mean_thetas[i], all_thetas[cluster], [polys[c].area for c in cluster], all_densities[cluster], all_polarizations[cluster], all_layers, all_velocities[cluster]))
            else:
                final_clusters.append(Cluster(real_ids, cluster, [polys[c].centroid for c in cluster], all_cluster_mean_thetas[i], all_thetas[cluster], [polys[c].area for c in cluster], all_densities[cluster], all_polarizations[cluster], all_layers))

    del polys
    return final_clusters

def clusters_single_frame(vor: Voronoi, valid_indices:np.ndarray, cells: list[Polygon], pol_vals: list[float], theta_vals: list[float], density_vals: list[float], v_vals: list[float] | None = None, tolerance:float = 1e-6, pol_thresh:float = 0.8, angle_thresh:float = np.pi/4, v_thresh:float = 0.01, min_cluster_size:int = 2):

    # Get adjacency graph which will be used to construct layers for each individual
    graph = build_adjacency_graph(vor, valid_indices, tolerance)

    # Convert original Voronoi point IDs to compact array indices.
    original_to_compact = {int(original_idx): compact_idx for compact_idx, original_idx in enumerate(valid_indices)}

    # Convert graph ids to compact ids
    graph = {original_to_compact[point_idx]: {original_to_compact[nbr] for nbr in neighbors if nbr in original_to_compact}
            for point_idx, neighbors in graph.items() if point_idx in original_to_compact}

    # Get dictionary of neighbour layer (compact) ids relative to each individual
    all_layers = {f: compute_voronoi_layers(graph, f, max_layers = 1) for f in graph}

    # Get clusters
    return find_clusters(all_polarizations = pol_vals, all_thetas = theta_vals, all_densities = density_vals, all_velocities = v_vals, all_layers = all_layers, polys = cells, pol_thresh = pol_thresh, angle_thresh = angle_thresh, v_thresh = v_thresh, min_cluster_size = min_cluster_size, valid_indices = valid_indices)

def extract_frame_cluster_stats(positions:np.ndarray, pol_vals:np.ndarray, theta_vals:np.ndarray, density_vals:np.ndarray, abs_frame: int, arena_center: np.ndarray, arena_radius: float, tolerance: float = 1e-6, pol_thresh: float = 0.8,
                                min_cluster_size: int = 2, area_factor: float = 1) -> FrameClusterStats | None:

    # Compute voronoi tesselation
    vor = Voronoi(positions)
    cells, all_pols, all_thetas, all_ds, valid_indices = [], [], [], [], []

    # Filter out invalid neighbourhoods
    for point_idx, (p_val, t_val, d_val) in enumerate(zip(pol_vals, theta_vals, density_vals)):
        region_idx = vor.point_region[point_idx]
        region     = vor.regions[region_idx]

        if len(region) == 0:
            continue

        vertices = vor.vertices[np.array(region)[(np.array(region) != -1)].astype(int)]
        poly     = Polygon(clip_voronoi_region(vertices, arena_center, arena_radius))

        if poly.is_empty:
            continue

        cells.append(poly)
        all_pols.append(p_val)
        all_thetas.append(t_val)
        all_ds.append(d_val)
        valid_indices.append(point_idx)

    # Compute clusters
    clusters = clusters_single_frame(vor=vor, valid_indices=np.asarray(valid_indices), cells=cells, pol_vals=all_pols, 
                                     theta_vals=all_thetas, density_vals=all_ds, tolerance=tolerance,
                                     pol_thresh=pol_thresh, min_cluster_size=min_cluster_size)

    del vor

    if not clusters:
        return None

    # Store cluster results
    centroids = np.array([c.centroid for c in clusters])
    areas      = np.array([c.total_area() for c in clusters]) * area_factor
    ns         = np.array([c.size() for c in clusters])
    med_pols  = np.array([c.median_polarization() for c in clusters])
    var_pols  = np.array([c.var_polarization() for c in clusters])
    med_ds    = np.array([c.median_density() for c in clusters])
    var_ds      = np.array([c.var_density() for c in clusters])
    mean_thetas = np.array([c.mean_theta for c in clusters])
    var_thetas = np.array([c.theta_var() for c in clusters])

    pol_layer_dicts = [c.shell_medians(c.polarizations) for c in clusters]
    d_layer_dicts   = [c.shell_medians(c.densities) for c in clusters]

    max_layers      = max(c.max_layer for c in clusters)

    p_by_layer = np.full((len(clusters), max_layers + 1), np.nan)
    d_by_layer = np.full((len(clusters), max_layers + 1), np.nan)

    for i, c in enumerate(clusters):
        p_by_layer[i, :(c.max_layer + 1)] = [pol_layer_dicts[i][j] for j in range(c.max_layer + 1)]
        d_by_layer[i, :(c.max_layer + 1)] = [d_layer_dicts[i][j] / area_factor for j in range(c.max_layer + 1)]

    # Flip for edge-relative view
    p_from_edge = np.full_like(p_by_layer, np.nan)
    d_from_edge = np.full_like(d_by_layer, np.nan)

    for i in range(len(clusters)):
        last = np.where(np.isnan(p_by_layer[i]))[0]
        last = last[0] if len(last) else p_by_layer.shape[1]
        p_from_edge[i, :last] = np.flip(p_by_layer[i, :last])
        d_from_edge[i, :last] = np.flip(d_by_layer[i, :last])

    return FrameClusterStats(
        abs_frame=abs_frame, centroids=centroids,
        areas=areas, ns=ns, median_pols=med_pols, var_pols=var_pols, median_densities=med_ds, var_densities=var_ds, mean_thetas=mean_thetas, var_thetas=var_thetas,
        p_by_layer=p_by_layer, d_by_layer=d_by_layer, p_from_edge=p_from_edge, d_from_edge=d_from_edge)

def find_reflections(ds: xr.Dataset, fps:int = 5, start_frame:int = 0, end_frame:int | None = None, arena_center_m:np.ndarray = np.array([0.01, 0]), subsample:int = 1):
    
    # Get frames
    abs_frames, ds_idcs = get_frame_slice(ds, start_frame, end_frame, subsample)

    # Get median value of polarizations as indicator for reflections (mostly sinusoidal)
    pols = np.nanmedian(ds['polarization_voronoi_None'].values[ds_idcs,:], axis = 1)

    # Get dominant frequency
    _, _, _, dominant_freq = fft_timeseries(pols, fps)

    # Find number of frames per period (approximately)
    period_frames = int(round(fps / dominant_freq))

    # Invert signal and find peaks (= troughs of original)
    trough_frames, _ = find_peaks(
        -pols,
        distance=period_frames * 0.5,   # troughs at least half a period apart
        prominence=0.1,                  # ignore shallow noise fluctuations
    )

    # Classify each reflect as left or right by checking how median x relates to center of arena
    x_var = 'centroid_x'
    if 'centroid_x' not in ds:
        x_var = 'x_raw'
    xs = np.nanmedian(ds[x_var].values[ds_idcs, :], axis = 1)
    sides = np.where(xs[trough_frames] > arena_center_m[0], 'right', 'left')

    return trough_frames, abs_frames[trough_frames], period_frames, sides


def _frame_worker(args):
    """Wrapper for using ProcessPoolExecutor."""
    pos, pols, dens, thetas, abs_frame, arena_center, arena_radius, tolerance, pol_thresh, min_cluster_size, area_factor = args
    return abs_frame, extract_frame_cluster_stats(pos, pols, thetas, dens, abs_frame, arena_center, arena_radius, tolerance, pol_thresh, min_cluster_size, area_factor)

def whole_batch_cluster_analysis(ds: xr.Dataset, output_dir: str, arena_center:np.ndarray, arena_radius:float, fps:int = 5, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1,tolerance: float = 1e-6,
    pol_thresh: float = 0.8, min_cluster_size: int = 2, area_factor: float = 1, n_workers:int = 8):
    
    # Find absolute and relative (to ds) frame indices
    abs_frames, rel_frames = get_frame_slice(ds, start_frame, end_frame, subsample)

    # Pre-slice dataset to prepare position, polarization, density arrays
    positions = np.array([ds['centroid_x'].values, ds['centroid_y'].values]) # (x/y, n_frames, max_n)
    pol_vals     = ds['polarization_voronoi_None'].values # (n_frames, max_n)
    density_vals = ds['density_voronoi_None'].values # (n_frames, max_n)
    theta_vals = ds['theta'].values # (n_frames, max_n)

    # Exclude points that are outside arena or non-finite
    dist_from_center  = np.linalg.norm(positions - arena_center[:, None, None], axis=0) # (n_frames, max_n)
    outside_arena     = dist_from_center > arena_radius # (n_frames, max_n)
    valid_mask        = (~np.isnan(positions).any(axis=0)) & (~np.isnan(pol_vals)) & (~np.isnan(density_vals)) & (~outside_arena) # (n_frames, max_n)

    # Build flat list of arguments for all (event, rel_frame) tasks
    tasks = []
    for i, rel_frame in enumerate(rel_frames):
        mask = valid_mask[rel_frame]
        tasks.append((positions[:, rel_frame, mask].T, pol_vals[rel_frame, mask], density_vals[rel_frame, mask], theta_vals[rel_frame, mask], abs_frames[i], arena_center, arena_radius, tolerance, pol_thresh, min_cluster_size, area_factor))

    # Run in parallel, collect into a plain list to avoid shared-state issues
    results: list[tuple[int, FrameClusterStats]] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_frame_worker, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Frames"):
            abs_frame, stats = future.result()
            if stats is not None:
                results.append((abs_frame, stats))

    # Merge into aggregated dict
    aggregated: dict[int, list[FrameClusterStats]] = defaultdict(list)
    for abs_frame, stats in results:
        aggregated[abs_frame].append(stats)

    # Save data to .h5 file
    output_file = output_dir + f"clusters/cluster_data_start_{abs_frames[0]}_end_{abs_frames[-1]}_subsample_{subsample}_pol_thresh_{pol_thresh}_min_cluster_size_{min_cluster_size}.h5"
    cluster_stats_to_h5(aggregated, output_file)

def _frame_cluster_worker(args):
    """Wrapper for using ProcessPoolExecutor."""
    pos, pols, dens, thetas, vels, abs_frame, arena_center, arena_radius, tolerance, pol_thresh, ang_thresh, v_thresh, min_cluster_size = args
    
    # Compute voronoi tesselation
    vor = Voronoi(pos)
    cells, all_pols, all_thetas, all_ds, all_vs, valid_indices = [], [], [], [], [], []

    # Filter out invalid neighbourhoods
    for point_idx, (p_val, t_val, d_val, v_val) in enumerate(zip(pols, thetas, dens, vels)):
        region_idx = vor.point_region[point_idx]
        region     = vor.regions[region_idx]

        if len(region) == 0:
            continue

        vertices = vor.vertices[np.array(region)[(np.array(region) != -1)].astype(int)]
        poly     = Polygon(clip_voronoi_region(vertices, arena_center, arena_radius))

        if poly.is_empty:
            continue

        cells.append(poly)
        all_pols.append(p_val)
        all_thetas.append(t_val)
        all_ds.append(d_val)
        all_vs.append(v_val)
        valid_indices.append(point_idx)

    # Compute clusters
    clusters = clusters_single_frame(vor = vor, valid_indices = np.array(valid_indices), cells = cells, pol_vals = np.asarray(all_pols), theta_vals = np.asarray(all_thetas), density_vals = np.asarray(all_ds), v_vals = np.asarray(all_vs), tolerance = tolerance, pol_thresh = pol_thresh, angle_thresh = ang_thresh, v_thresh = v_thresh, min_cluster_size = min_cluster_size)
    
    return abs_frame, clusters

def compute_whole_batch_clusters(ds: xr.Dataset, output_dir: str, 
                                 x_var:str = 'x_raw', y_var:str = 'y_raw', density_var:str = 'density_voronoi_None',
                                 polarization_var:str = 'polarization_voronoi_None', v_var:str = 'v_high_ord',  
                                 arena_center:np.ndarray = np.array([0.01, 0]), arena_radius:float = 2.38, fps:int = 5, 
                                 start_frame:int = 0, end_frame:int | None = None, subsample:int = 1, tolerance: float = 1e-6, 
                                 angle_thresh: float = np.pi/4, pol_thresh: float = 0.8, v_thresh:float = 0.01, min_cluster_size: int = 2, 
                                 n_workers:int = 4, save_batch:int = 500):
    '''Make sure to use the 'both' dataset which includes velocities.'''


    # Find absolute and relative (to ds) frame indices
    abs_frames, rel_frames = get_frame_slice(ds, start_frame, end_frame, subsample)

    # Pre-slice dataset to reduce computation time
    # Ensure coordinate order is consistent
    positions = np.array([ds[x_var].transpose('frame', 'id').values, ds[y_var].transpose('frame', 'id').values]) # (x/y, n_frames, max_n)
    pol_vals     = ds[polarization_var].transpose('frame', 'id').values # (n_frames, max_n)
    density_vals = ds[density_var].transpose('frame', 'id').values # (n_frames, max_n)
    theta_vals = ds['theta'].transpose('frame', 'id').values # (n_frames, max_n)
    v_vals = ds[v_var].transpose('frame', 'id').values # (n_frames, max_n)

    # Define params for saving to .h5 file
    h5_name =  f"{abs_frames[0]}_{abs_frames[-1]}_{(fps*subsample):.1f}Hz"
    params = {'x_var': x_var, 'y_var': y_var, 'density_var': density_var, 'polarization_var': polarization_var,
                'v_var': v_var, 'arena_center': arena_center.tolist(), 'arena_radius': arena_radius, 'fps': fps,
                'start_frame_abs': int(abs_frames[0]), 'end_frame_abs': int(abs_frames[-1]), 'subsample': subsample,
                'tolerance': tolerance, 'pol_thresh': pol_thresh, 'angle_thresh': angle_thresh, 'v_thresh': v_thresh,
                'min_cluster_size': min_cluster_size}
    save_batch = 5 # Save to file every 500 frames

    # Check for existing results file
    h5_name = f"{abs_frames[0]}_{abs_frames[-1]}_{(fps * subsample):.1f}Hz"
    target_h5 = (Path(output_dir) / "clusters" / f"{h5_name}.h5")
    target_h5.parent.mkdir(parents=True, exist_ok=True)
    save_dir = str(target_h5.parent) + os.sep
    save_name = target_h5.stem

    # Read completed frames
    def frame_group_is_complete(grp):
        required = {"ids", "layers", "idx_ptr"}
        if not required.issubset(grp.keys()):
            return False

        ids = grp["ids"][:]
        layers = grp["layers"][:]
        idx_ptr = grp["idx_ptr"][:]

        return (
            len(ids) == len(layers)
            and len(idx_ptr) >= 1
            and idx_ptr[0] == 0
            and idx_ptr[-1] == len(ids)
            and np.all(np.diff(idx_ptr) >= 0)
        )

    completed_frames = set()
    if target_h5.exists():
        with h5py.File(target_h5, "r") as f:
            for name in f.keys():
                if name.isdecimal() and frame_group_is_complete(f[name]):
                    completed_frames.add(int(name))

    print('Completed frames: ', completed_frames)
    frame_pairs = [(int(abs_frame), int(rel_frame)) for abs_frame, rel_frame in zip(abs_frames, rel_frames) if int(abs_frame) not in completed_frames]

    def make_task(abs_frame, rel_frame):
            pos = np.column_stack((positions[0, rel_frame], positions[1, rel_frame]))
            dist = np.linalg.norm(pos - arena_center, axis=1)

            mask = (np.isfinite(pos).all(axis=1) & np.isfinite(pol_vals[rel_frame]) & np.isfinite(density_vals[rel_frame]) & (dist <= arena_radius))

            return (
                pos[mask],
                pol_vals[rel_frame, mask],
                density_vals[rel_frame, mask],
                theta_vals[rel_frame, mask],
                v_vals[rel_frame, mask],
                abs_frame,
                arena_center,
                arena_radius,
                tolerance,
                pol_thresh,
                angle_thresh,
                v_thresh,
                min_cluster_size,
            )

    # Compute clusters in parallel
    with Parallel(n_jobs=n_workers, return_as="generator_unordered", batch_size=1, pre_dispatch=max(1, n_workers)) as parallel:
        
        # Get cluster ids and layers
        completed = parallel(delayed(_frame_cluster_worker)(make_task(abs_frame, rel_frame))
            for abs_frame, rel_frame in frame_pairs)

        results = {}
        frames_in_batch = 0

        # Show progress bar for completed frames
        for abs_frame, frame_clusters in tqdm(completed, total=len(frame_pairs), desc="Completed frames"):
            if frame_clusters:
                results[int(abs_frame)] = frame_clusters

            frames_in_batch += 1

            # Save in batches to prevent memory issues
            if frames_in_batch >= save_batch:
                if results:
                    cluster_ids_to_h5(results, save_dir, save_name, params)
                results = {}
                frames_in_batch = 0

        # Save results from the final, possibly smaller batch.
        if results:
            cluster_ids_to_h5(results, save_dir, save_name, params)
