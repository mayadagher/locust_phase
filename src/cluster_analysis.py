'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from shapely.geometry import Polygon
from tqdm import tqdm
from correlation_length import build_adjacency_graph, compute_voronoi_layers
from scipy.spatial import Voronoi
import collections
from collections import defaultdict
from scipy.signal import find_peaks
from dataclasses import dataclass
from helper_fns import get_frame_slice, clip_voronoi_region, fft_timeseries
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_handling import cluster_stats_to_h5

'''_____________________________________________________CLASSES____________________________________________________________'''

class Cluster():

    def __init__(self, ids: np.ndarray[int], centroids: list[np.ndarray], mean_theta: float, theta_arr: np.ndarray[float], area_list: list[float], density_arr: np.ndarray[float],
                 polarization_arr: np.ndarray[float], all_layers: dict[int, dict[int, int]]):

        self.ids = ids
        self.mean_theta = mean_theta
        self.thetas = theta_arr
        self.areas = area_list
        self.densities = density_arr
        self.polarizations = polarization_arr

        # Store only direct neighbours (layer == 1) for intra-cluster BFS
        id_set = set(ids)
        self.graph: dict[int, set[int]] = {cell_id: {nbr for nbr, dist in all_layers[cell_id].items() if dist == 1 and nbr in id_set} for cell_id in ids}

        self.centrality = self._compute_centrality()
        self.centroid_id = self.ids[np.argmin(self.centrality)]
        centroid = centroids[np.argmin(self.centrality)]
        self.centroid = np.array([centroid.x, centroid.y])
        self.max_layer = int(np.nanmax(self.centroid_distance_array()))

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

    def _compute_centrality(self) -> np.ndarray:
        """
        For each cell, compute its mean topological distance to all other cells
        via BFS over the intra-cluster graph.
        Returns an array (aligned with self.ids) where lower = more central.
        """
        mean_distances = np.zeros(len(self.ids))

        for i, cell_id in enumerate(self.ids):
            distances = self._bfs_distances(cell_id)
            dists = list(distances.values())
            mean_distances[i] = np.mean(dists) if dists else 0.0

        return mean_distances

    def centroid_distance_array(self) -> np.ndarray:
        """
        Returns an array (aligned with self.ids) of each cell's topological
        distance from the centroid. Unreachable cells get np.nan.
        """
        centroid_dists = self._bfs_distances(self.centroid_id)
        # centroid itself is distance 0
        centroid_dists[self.centroid_id] = 0
        return np.array([centroid_dists.get(cell_id, np.nan) for cell_id in self.ids])

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
                  pol_thresh: float = 0.8, angle_thresh: float = np.pi/4, min_cluster_size: int = 2):
    
    def angular_diff(a: float, b: float) -> float:
        """Smallest angle between two headings, result in [0, π]."""
        diff = abs(a - b) % (2 * np.pi)
        return diff if diff <= np.pi else 2 * np.pi - diff

    def add_to_cluster( id: int, cluster_mean_theta: float, cluster: list[int], visited_this_bfs: set[int]) -> tuple[list[int], set[int], float]:

        layer_dict = all_layers[id]
        nbrs = np.array(list(layer_dict.keys()))[np.array(list(layer_dict.values())) == 1]

        # Neighbours must be: polarized, unassigned globally, not yet seen in this BFS
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
            final_clusters.append(Cluster(cluster, [polys[c].centroid for c in cluster], all_cluster_mean_thetas[i], all_thetas[cluster], [polys[c].area for c in cluster], all_densities[cluster], all_polarizations[cluster], all_layers))

    del polys
    return final_clusters

def clusters_single_frame(vor: Voronoi, valid_indices:np.ndarray, cells: list[Polygon], pol_vals: list[float], theta_vals: list[float], density_vals: list[float], tolerance:float = 1e-6, pol_thresh:float = 0.8, min_cluster_size:int = 2):

    # Get adjacency graph which will be used to construct layers for each individual
    graph = build_adjacency_graph(vor, valid_indices, tolerance)

    # Get dictionary of neighbour layer values relative to each individual
    all_layers = {f: compute_voronoi_layers(graph, f, max_layers = 1) for f in graph}

    # Get clusters
    pol_vals = np.array(pol_vals)
    theta_vals = np.array(theta_vals)
    density_vals = np.array(density_vals)
    return find_clusters(pol_vals, theta_vals, density_vals, all_layers, cells, pol_thresh, min_cluster_size)

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
    clusters = clusters_single_frame(vor, np.array(valid_indices), cells, all_pols, all_thetas, all_ds, tolerance, pol_thresh, min_cluster_size)

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

def find_reflections(ds: xr.Dataset, fps:int = 5, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1):
    
    # Get frames
    abs_frames, ds_idcs = get_frame_slice(ds, start_frame, end_frame, subsample)

    # Get median value of polarizations as indicator for reflections (mostly sinusoidal)
    pols = np.nanmedian(ds['polarization_voronoi_None'].values[ds_idcs,:], axis = 1)

    # Get median value of density as indicator for side of reflections (mostly sinusoidal, twice the period, same phase)
    dens = np.nanmedian(ds['density_voronoi_None'].values[ds_idcs,:], axis = 1)

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

    # Classify each reflection as left (density peak) or right (density trough)
    # by checking whether density is above or below its median at each trough frame
    dens_median = np.median(dens[trough_frames])
    sides = np.where(dens[trough_frames] > dens_median, 'left', 'right')

    return trough_frames, abs_frames[trough_frames], period_frames, sides


def _frame_worker(args):
    """Wrapper for using ProcessPoolExecutor."""
    pos, pols, dens, thetas, abs_frame, arena_center, arena_radius, tolerance, pol_thresh, min_cluster_size, area_factor = args
    return abs_frame, extract_frame_cluster_stats(pos, pols, thetas, dens, abs_frame, arena_center, arena_radius, tolerance, pol_thresh, min_cluster_size, area_factor)

def whole_batch_cluster_analysis(ds: xr.Dataset, output_dir: str, arena_center:np.ndarray, arena_radius:float, fps:int = 5, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1,tolerance: float = 1e-6,
    pol_thresh: float = 0.8, min_cluster_size: int = 2, area_factor: float = 1, n_workers:int = 8):
    
    # Find absolute and relative (to ds) frame indices
    abs_frames, rel_frames = get_frame_slice(ds, start_frame, end_frame, subsample)

    # Accumulate across events: rel_frame -> list of per-frame stats objects
    aggregated: dict[int, list[FrameClusterStats]] = defaultdict(list)

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

    # Merge into aggregated dict on the main process — no race conditions
    aggregated: dict[int, list[FrameClusterStats]] = defaultdict(list)
    for abs_frame, stats in results:
        aggregated[abs_frame].append(stats)

    # Save data to .h5 file
    output_file = output_dir + f"clusters/cluster_data_start_{abs_frames[0]}_end_{abs_frames[-1]}_subsample_{subsample}_pol_thresh_{pol_thresh}_min_cluster_size_{min_cluster_size}.h5"
    cluster_stats_to_h5(aggregated, output_file)