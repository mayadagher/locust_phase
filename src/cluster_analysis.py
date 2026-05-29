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
from helper_fns import get_frame_slice, clip_voronoi_region
import time

'''_____________________________________________________CLASSES____________________________________________________________'''

class Cluster():

    def __init__(self, ids: np.ndarray[int], poly_list: list[Polygon], density_arr: np.ndarray[float],
                 polarization_arr: np.ndarray[float], all_layers: dict[int, dict[int, int]]):

        self.ids = ids
        self.polys = poly_list
        self.densities = density_arr
        self.polarizations = polarization_arr

        # Store only direct neighbours (layer == 1) for intra-cluster BFS
        # This is the only thing we need from all_layers going forward
        id_set = set(ids)
        self.graph: dict[int, set[int]] = {
            cell_id: {nbr for nbr, dist in all_layers[cell_id].items() if dist == 1 and nbr in id_set}
            for cell_id in ids
        }

        self.centrality = self._compute_centrality()
        self.centroid_id = self.ids[np.argmin(self.centrality)]
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

    def shell_means(self, values: np.ndarray) -> dict[int, float]:
        dist_arr = self.centroid_distance_array()
        shells = {}
        for dist, val in zip(dist_arr, values):
            if np.isnan(dist):
                continue
            shells.setdefault(int(dist), []).append(val)
        return {d: np.mean(v) for d, v in sorted(shells.items())}

    def size(self) -> int:
        return len(self.ids)

    def mean_polarization(self) -> float:
        return float(np.mean(self.polarizations))

    def mean_density(self) -> float:
        return float(np.mean(self.densities))

    def total_area(self) -> float:
        return sum(p.area for p in self.polys)
    
@dataclass
class FrameClusterStats:
    """All cluster-level statistics for one frame, ready for aggregation."""
    rel_frame:      int                        # offset from event (negative = before)
    abs_frame:      int
    # clusters:       list                       # list[Cluster] — raw objects if needed
    areas:          np.ndarray
    ns:             np.ndarray
    mean_pols:      np.ndarray
    mean_densities: np.ndarray
    p_by_layer:     np.ndarray                 # (n_clusters, max_layer+1), nan-padded
    d_by_layer:     np.ndarray
    p_from_edge:    np.ndarray                 # flipped version
    d_from_edge:    np.ndarray
'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def find_clusters(all_polarizations:np.ndarray, all_densities:np.ndarray, all_layers:dict[int, dict[int, int]], polys: list[Polygon], pol_thresh:float = 0.8, min_cluster_size:int = 2):
    
    def add_to_cluster(id:int, all_polarizations:np.ndarray, all_layers:dict[int, dict[int, int]], pol_thresh:float, cluster:list[int], prev_explored:list[int]):

        layer_dict = all_layers[id]

        # Find direct neighbours of current id (layer 1)
        nbrs = np.array(list(layer_dict.keys()))[np.array(list(layer_dict.values())) == 1]

        # Find neighbours that belong to the cluster (have a high enough polarization)
        cluster_nbrs = nbrs[all_polarizations[nbrs] >= pol_thresh]
        new_cluster_nbrs = [int(i) for i in cluster_nbrs if i not in cluster] # Make sure not to add or explore duplicates

        # Add cluster neighbours to list
        cluster += new_cluster_nbrs

        # Mark current id as 'explored'
        prev_explored.append(id)

        # Explore cluster neighbours
        for i in new_cluster_nbrs:
            if i not in prev_explored:
                cluster, prev_explored = add_to_cluster(i, all_polarizations, all_layers, pol_thresh, cluster, prev_explored)

        return cluster, prev_explored

    # Define list of previously explored ids
    prev_explored = []
    all_clusters = []

    for id in all_layers.keys():
        
        # Exclude ids that are below the polarization threshold or already explored
        if all_polarizations[id] < pol_thresh or id in prev_explored:
            if not id in prev_explored:
                prev_explored.append(id)
            continue

        # Start a new cluster containing this id
        cluster = [id]

        # Use a recursive function to find all ids in this cluster
        cluster, prev_explored = add_to_cluster(id, all_polarizations, all_layers, pol_thresh, cluster, prev_explored)

        # Add cluster to list of all clusters
        all_clusters.append(cluster)

    # Convert clusters of indices into lists of polygons and their polarizations/densities
    final_clusters = []
    for cluster in all_clusters:
        if len(cluster) >= min_cluster_size:

            # final_clusters.append(Cluster(cluster, [polys[c] for c in cluster], all_densities[cluster], all_polarizations[cluster], all_layers))
            final_clusters.append(Cluster([polys[c] for c in cluster], all_densities[cluster], all_polarizations[cluster], all_layers))
            
    return final_clusters

def clusters_single_frame(vor: Voronoi, valid_indices:np.ndarray, cells: list[Polygon], pol_vals: list[float], density_vals: list[float], tolerance:float = 1e-6, pol_thresh:float = 0.8, min_cluster_size:int = 2):

    # Get adjacency graph which will be used to construct layers for each individual
    graph = build_adjacency_graph(vor, valid_indices, tolerance)

    # Get dictionary of neighbour layer values relative to each individual
    all_layers = {f: compute_voronoi_layers(graph, f, max_layers = 1) for f in graph}

    # Get clusters
    pol_vals = np.array(pol_vals)
    density_vals = np.array(density_vals)
    return find_clusters(pol_vals, density_vals, all_layers, cells, pol_thresh, min_cluster_size)

def extract_frame_cluster_stats(positions:np.ndarray, pol_vals:np.ndarray, density_vals:np.ndarray, theta_vals:np.ndarray, frame_idx: int, rel_frame: int, arena_center: np.ndarray, arena_radius: float, tolerance: float = 1e-6, pol_thresh: float = 0.8,
                                min_cluster_size: int = 2, area_factor: float = 1) -> FrameClusterStats | None:

    # Compute voronoi tesselation
    vor = Voronoi(positions)
    cells, all_pols, all_ds, valid_indices = [], [], [], []

    # Filter out invalid neighbourhoods
    for point_idx, (p_val, d_val) in enumerate(zip(pol_vals, density_vals)):
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
        all_ds.append(d_val)
        valid_indices.append(point_idx)

    # Compute clusters
    clusters = clusters_single_frame(vor, np.array(valid_indices), cells, all_pols, density_vals, tolerance, pol_thresh, min_cluster_size)

    del vor

    if not clusters:
        return None

    # Store cluster results
    areas      = np.array([c.total_area() for c in clusters]) * area_factor
    ns         = np.array([c.size() for c in clusters])
    mean_pols  = np.array([c.mean_polarization() for c in clusters])
    mean_ds    = np.array([c.mean_density() for c in clusters]) / area_factor

    pol_layer_dicts = [c.shell_means(c.polarizations) for c in clusters]
    d_layer_dicts   = [c.shell_means(c.densities) for c in clusters]
    max_layers      = max(c.max_layer for c in clusters)

    p_by_layer = np.full((len(clusters), max_layers + 1), np.nan)
    d_by_layer = np.full((len(clusters), max_layers + 1), np.nan)

    for i, c in enumerate(clusters):
        p_by_layer[i, :(c.max_layer + 1)] = [pol_layer_dicts[i][j] for j in range(c.max_layer + 1)]
        d_by_layer[i, :(c.max_layer + 1)] = [d_layer_dicts[i][j] for j in range(c.max_layer + 1)]

    # Flip for edge-relative view
    p_from_edge = np.full_like(p_by_layer, np.nan)
    d_from_edge = np.full_like(d_by_layer, np.nan)

    for i in range(len(clusters)):
        last = np.where(np.isnan(p_by_layer[i]))[0]
        last = last[0] if len(last) else p_by_layer.shape[1]
        p_from_edge[i, :last] = np.flip(p_by_layer[i, :last])
        d_from_edge[i, :last] = np.flip(d_by_layer[i, :last])

    return FrameClusterStats(
        rel_frame=rel_frame, abs_frame=frame_idx,
        clusters=clusters, areas=areas, ns=ns,
        mean_pols=mean_pols, mean_densities=mean_ds,
        p_by_layer=p_by_layer, d_by_layer=d_by_layer,
        p_from_edge=p_from_edge, d_from_edge=d_from_edge,
    )

def find_reflections(ds: xr.Dataset, fps:int = 5, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1):

    def fft_timeseries(time_series, sample_rate=1.0):
        """
        Returns
        -------
        freqs        : array of frequencies (positive only)
        power        : single-sided power spectrum
        phase        : phase at each frequency (radians)
        dominant_freq: frequency with peak power
        """
        ts = np.asarray(time_series, dtype=float)
        ts -= ts.mean()          # remove DC offset
        n = len(ts)

        fft_vals = np.fft.rfft(ts)
        freqs    = np.fft.rfftfreq(n, d=1.0 / sample_rate)

        power    = (np.abs(fft_vals) ** 2) * (2.0 / n)   # single-sided, normalised
        power[0] = 0.0                                     # zero out DC bin
        phase    = np.angle(fft_vals)

        dominant_freq = freqs[np.argmax(power)]

        return freqs, power, phase, dominant_freq
    
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

    return trough_frames, abs_frames[trough_frames], period_frames

def analyse_reflection_events(
    ds: xr.Dataset,
    event_frames: list[int],       # output of your trough-finder
    half_period: int,              # frames either side of event
    arena_center: np.ndarray,
    arena_radius: float,
    tolerance: float = 1e-6,
    pol_thresh: float = 0.8,
    min_cluster_size: int = 2,
    area_factor: float = 1,                      # forwarded to extract_frame_cluster_stats
) -> dict[int, list[FrameClusterStats]]:
    """
    For each event, collect FrameClusterStats for frames in
    [event - half_period, event + half_period].

    Returns {rel_frame: [FrameClusterStats, ...]} aggregated across all events,
    where rel_frame=0 is the reflection frame itself.
    """
    n_frames = ds.sizes['frame']  # adjust to your ds dimension name

    # Accumulate across events: rel_frame -> list of per-frame stats objects
    aggregated: dict[int, list[FrameClusterStats]] = defaultdict(list)

    # Pre-slice dataset to prepare position, polarization, density arrays
    positions = np.array([ds['centroid_x'].values, ds['centroid_y'].values]) # (x/y, n_frames, max_n)
    pol_vals     = ds['polarization_voronoi_None'].values # (n_frames, max_n)
    density_vals = ds['density_voronoi_None'].values # (n_frames, max_n)
    theta_vals = ds['theta'].values # (n_frames, max_n)

    # Exclude points that are outside arena or non-finite
    dist_from_center  = np.linalg.norm(positions - arena_center, axis=0) # (n_frames, max_n)
    outside_arena     = dist_from_center > arena_radius # (n_frames, max_n)
    valid_mask        = (~np.isnan(positions).any(axis=1)) & (~np.isnan(pol_vals)) & (~np.isnan(density_vals)) & (~outside_arena) # (n_frames, max_n)

    # Iterate over reflection events
    for k, event_frame in enumerate(event_frames):
        print(f'Computing reflection event {k + 1}/{len(event_frames)}.')

        # Iterate over frames around reflection events
        for rel_frame in tqdm(range(-half_period, half_period + 1), desc="Relative frames"):
            abs_frame = event_frame + rel_frame

            if abs_frame < 0 or abs_frame >= n_frames:
                continue

            pos = positions[:, rel_frame, valid_mask[rel_frame]]
            pols = pol_vals[rel_frame, valid_mask[rel_frame]]
            dens = density_vals[rel_frame, valid_mask[rel_frame]]
            thetas = theta_vals[rel_frame, valid_mask[rel_frame]]

            stats = extract_frame_cluster_stats(pos, pols, dens, thetas, abs_frame, rel_frame, arena_center, arena_radius, tolerance, pol_thresh, min_cluster_size, area_factor)

            if stats is not None:
                aggregated[rel_frame].append(stats)

    return aggregated



# all_densities = np.arange(8)
# all_polarizations = np.array([0.9,0.85, 0.7, 0.6, 0.9, 0.6, 0.7, 0.95]) # 8 ids
# all_layers = {0: {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:4, 7:4}, 1: {0:1, 1:0, 2:1, 3:2, 4:2, 5:3, 6:4, 7:4}, 
#               2: {0:1, 1:1, 2:0, 3:1, 4:1, 5:2, 6:3, 7:3}, 3: {0:2, 1:2, 2:1, 3:0, 4:1, 5:2, 6:3, 7:3},
#               4: {0:2, 1:2, 2:1, 3:1, 4:0, 5:1, 6:2, 7:2}, 5: {0:3, 1:3, 2:2, 3:2, 4:1, 5:0, 6:1, 7:1},
#               6: {0:4, 1:4, 2:3, 3:3, 4:2, 5:1, 6:0, 7:1}, 7: {0:4, 1:4, 2:3, 3:3, 4:2, 5:1, 6:1, 7:0},
#               }
# polys = [1, 2, 3, 4, 5, 6, 7, 8]
# pol_thresh = 0.8
# find_clusters(all_polarizations, all_densities, all_layers, polys, pol_thresh)