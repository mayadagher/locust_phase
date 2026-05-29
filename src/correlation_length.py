"""
voronoi_correlation.py

Computes the spatial correlation function C(k) of a per-animal parameter
as a function of Voronoi layer k, and estimates a correlation length from
the resulting curve.

Correlation definition
----------------------
For a focal cell f and a cell i at layer k, the pairwise correlation
contribution depends on the parameter type:

  Scalar   –  Pearson-style:  c(f, i) = (x_f - μ)(x_i - μ) / σ²
              where μ, σ² are the global mean and variance.
              C(k) is the mean of c(f, i) over all (f, i) pairs at distance k.
              C(0) = 1 by construction.

  Circular –  Cosine similarity of the unit vectors:
              c(f, i) = cos(θ_f − θ_i)  (equivalently, Re[e^{i(θ_f−θ_i)}])
              C(k) is the mean over all pairs at distance k.
              C(0) = 1 by construction.

Both estimators are unbiased, bounded in [−1, 1], and equal 1 at k=0.
They converge to 0 when cells at layer k are statistically independent.

Global vs. single-focal mode
-----------------------------
  global  –  Every cell is used as a focal cell in turn; pair (f, i) is
             counted once for each ordered focal cell f.  The returned
             curve is the grand mean C(k) ± standard error across all
             focal contributions.  Use this for a population-level
             correlation length.

  focal   –  A single focal cell is fixed; C(k) is the mean over all
             cells at exactly layer k from that focal cell.  Use this
             to inspect the local neighbourhood structure around one
             individual.

Correlation length estimation
------------------------------
The correlation length ξ is estimated by fitting

    C(k) = A · exp(−k / ξ)

to the curve for k ≥ 1 (k = 0 is always 1 and is excluded from the fit
to avoid anchoring bias).  The fit is done in log space (robust linear
regression on log|C(k)|) whenever C(k) > 0 for a sufficient range of k,
and falls back to scipy's curve_fit otherwise.

If the curve does not decay monotonically (e.g. oscillating order), ξ is
reported as the first zero-crossing layer instead, with a warning.

Dependencies
------------
    numpy
    scipy
    (matplotlib – only for the demo)

The functions in this module expect the outputs of voronoi_layers.py:
    cells   : list[shapely.Polygon]
    layers  : dict[int, int]   {cell_index: layer_from_focal}
    graph   : dict[int, set[int]]
"""
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from typing import Literal
import collections
from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
from scipy.spatial import Voronoi


# ---------------------------------------------------------------------------
# Compute layers of Voronoi tessellation
# ---------------------------------------------------------------------------

def build_adjacency_graph(vor: Voronoi, valid_indices: np.ndarray, tolerance: float = 1e-6) -> dict[int, set[int]]:
    """
    Build adjacency graph directly from Voronoi ridge data — O(n) vs O(n²).
    
    vor.ridge_points[k] = [i, j] means points i and j share a Voronoi ridge.
    We only keep ridges where both endpoints are valid (inside arena, non-nan).
    
    Parameters
    ----------
    vor : Voronoi
        The Voronoi tessellation object.
    valid_indices : np.ndarray
        Indices into vor.points that are considered valid cells.
    tolerance : float
        Min ridge length to count as shared edge (filters point-contact ridges).
    """
    valid_set = set(valid_indices)
    graph: dict[int, set[int]] = {i: set() for i in valid_indices}

    for ridge_idx, (i, j) in enumerate(vor.ridge_points):
        if i not in valid_set or j not in valid_set:
            continue

        # Filter ridges that are just point contacts (vertex indices for this ridge)
        ridge_vertices = vor.ridge_vertices[ridge_idx]
        if -1 in ridge_vertices:
            # Infinite ridge — these are boundary ridges, still valid adjacency
            # but have no finite length; include them unless you want to exclude edges
            graph[i].add(j)
            graph[j].add(i)
            continue

        # Finite ridge: check length if tolerance matters
        v0, v1 = vor.vertices[ridge_vertices[0]], vor.vertices[ridge_vertices[1]]
        if np.linalg.norm(v1 - v0) > tolerance:
            graph[i].add(j)
            graph[j].add(i)

    return graph

def compute_voronoi_layers(graph: dict[int, set[int]], focal_index: int, max_layers: int | None = None) -> dict[int, int]:
    """
    Compute the layer (graph distance) of every cell relative to *focal_index*.
    Performs breadth-first search.

    Parameters
    ----------
    graph:
        Adjacency dict as returned by ``build_adjacency_graph``.
    focal_index:
        Index of the focal cell (layer 0).
    max_layers:
        If provided, only cells within this many hops are returned.
        Cells beyond ``max_layers`` (or unreachable) are excluded from output.
        If ``None``, all reachable cells are returned (original behaviour).

    Returns
    -------
    layers : dict[int, int]
        ``layers[i]`` is the shortest-path distance from cell *i* to the
        focal cell. If ``max_layers`` is set, only cells with distance
        <= ``max_layers`` are included. Otherwise, unreachable cells
        are assigned ``-1``.
    """
    if max_layers is not None and max_layers < 0:
        raise ValueError(f"max_layers must be >= 0, got {max_layers}")

    layers: dict[int, int] = {} if max_layers is not None else {i: -1 for i in graph}
    layers[focal_index] = 0

    queue: collections.deque[int] = collections.deque([focal_index])

    while queue:
        current = queue.popleft()
        current_layer = layers[current]

        # Don't expand neighbours beyond the cutoff
        if max_layers is not None and current_layer == max_layers:
            continue

        for neighbor in graph[current]:
            if neighbor not in layers or layers[neighbor] == -1:
                layers[neighbor] = current_layer + 1
                queue.append(neighbor)

    return layers

# ---------------------------------------------------------------------------
# Core pair-correlation helpers
# ---------------------------------------------------------------------------

def _scalar_corr(x: np.ndarray, i: int, j: int, mu: float, sigma2: float) -> float:
    """Normalised scalar covariance contribution for a single pair."""
    if sigma2 == 0:
        return 0.0
    return float((x[i] - mu) * (x[j] - mu) / sigma2)


def _circular_corr(angles: np.ndarray, i: int, j: int) -> float:
    """Cosine of the angular difference for a single pair."""
    return float(np.cos(angles[i] - angles[j]))

def _build_pair_contributions(values: np.ndarray, layers_from_focal: dict[int, int], param_type: Literal["scalar", "circular"], mu: float, sigma2: float) -> dict[int, list[float]]:
    """
    For one fixed focal cell, collect per-layer correlation contributions.

    Returns
    -------
    contribs : dict[layer -> list[float]]
        contribs[k] is the list of pair-correlation values c(focal, i)
        for all i at layer k (k >= 0, including the focal cell itself).
    """
    contribs: dict[int, list[float]] = {}

    for idx, layer in layers_from_focal.items():
        if layer < 0:
            continue  # unreachable cell
        if layer == 0:
            c = 1.0  # self-correlation is always 1
        elif param_type == "scalar":
            focal_idx = next(i for i, lyr in layers_from_focal.items() if lyr == 0)
            c = _scalar_corr(values, focal_idx, idx, mu, sigma2)
        else:
            focal_idx = next(i for i, lyr in layers_from_focal.items() if lyr == 0)
            c = _circular_corr(values, focal_idx, idx)

        contribs.setdefault(layer, []).append(c)

    return contribs


# ---------------------------------------------------------------------------
# Correlation length fitting
# ---------------------------------------------------------------------------

def _fit_correlation_length(
    ks: np.ndarray,
    Ck: np.ndarray,
    Ck_se: np.ndarray,
) -> tuple[float, float, str]:
    """
    Fit an exponential decay C(k) = A * exp(-k / xi) to the curve.

    Fitting is performed for k >= 1 only (k=0 is always 1 by definition
    and would anchor the fit artificially).

    Returns
    -------
    xi, xi_err, fit_type
    """
    mask = ks >= 1
    ks_fit = ks[mask].astype(float)
    Ck_fit = Ck[mask]
    se_fit = Ck_se[mask]

    # --- check for zero-crossing first ---
    if np.any(Ck_fit <= 0):
        first_zero = ks_fit[Ck_fit <= 0][0]
        # interpolate linearly between last positive and first non-positive
        pos_mask = Ck_fit > 0
        if pos_mask.any() and (~pos_mask).any():
            k_last_pos = ks_fit[pos_mask][-1]
            k_first_neg = ks_fit[~pos_mask][0]
            c_last = Ck_fit[pos_mask][-1]
            c_first = Ck_fit[~pos_mask][0]
            if c_last != c_first:
                xi = float(k_last_pos
                           + c_last / (c_last - c_first)
                           * (k_first_neg - k_last_pos))
            else:
                xi = float(first_zero)
        else:
            xi = float(first_zero)
        return xi, float("nan"), "zero_crossing"

    # --- log-linear fit (robust for clean exponentials) ---
    pos = Ck_fit > 0
    if pos.sum() < 3:
        return float("nan"), float("nan"), "failed"

    log_C = np.log(Ck_fit[pos])
    k_pos = ks_fit[pos]

    # Weighted linear regression: log C = log A - k/xi  =>  slope = -1/xi
    sigma_log = se_fit[pos] / Ck_fit[pos] + 1e-12  # delta method
    weights = 1.0 / sigma_log**2

    W = weights.sum()
    Wk = (weights * k_pos).sum()
    WlC = (weights * log_C).sum()
    Wkk = (weights * k_pos**2).sum()
    WklC = (weights * k_pos * log_C).sum()

    denom = W * Wkk - Wk**2
    if abs(denom) < 1e-12:
        return float("nan"), float("nan"), "failed"

    slope = (W * WklC - Wk * WlC) / denom
    # slope = -1/xi  =>  xi = -1/slope
    if slope >= 0:
        # Non-decaying: report half-max layer instead
        above_half = ks_fit[Ck_fit >= 0.5]
        xi = float(above_half[-1]) if len(above_half) else float("nan")
        return xi, float("nan"), "zero_crossing"

    xi = -1.0 / slope

    # Uncertainty propagation
    var_slope = W / denom
    xi_err = xi**2 * np.sqrt(var_slope)

    return float(xi), float(xi_err), "exponential"


def voronoi_corr_vectors(values: np.ndarray, valid_indices:np.ndarray, all_layers: dict[int, dict[int, int]], param_type: Literal["scalar", "circular"] = "scalar", 
                             max_layer: int | None = None, min_pairs_per_layer: int = 5, subsample: int | None = None):
    """
    Compute the population-level layer-correlation curve averaged over
    all focal cells.

    The curve C(k) is computed per focal cell, then averaged; the
    standard error reflects variability *across focal cells*, which is
    more interpretable than the SE across pairs.

    Parameters
    ----------
    values:
        1-D array of length N.
    all_layers:
        ``{focal_index: {cell_index: layer}}`` — pre-computed layer dicts
        for every focal cell.  Build this by calling ``compute_voronoi_layers``
        once per cell:

            from voronoi_layers import compute_voronoi_layers, build_adjacency_graph
            graph = build_adjacency_graph(cells)
            all_layers = {f: compute_voronoi_layers(graph, f) for f in range(N)}

    param_type:
        ``"scalar"`` or ``"circular"``.
    max_layer:
        Truncate the curve at this layer.  Defaults to the 95th percentile
        of maximum reachable layer across all focal cells.
    min_pairs_per_layer:
        Layers with fewer total pairs than this threshold are dropped from
        the output (they are too noisy to be informative).
    subsample:
        Optionally only compute correlation vectors for a random subsample of the individuals. None will sample all individuals.
    """
    values = np.asarray(values, dtype=float)
    N = len(values)

    if param_type == "scalar":
        mu = float(np.mean(values))
        sigma2 = float(np.var(values))
    else:
        mu, sigma2 = 0.0, 1.0

    # Determine max_layer from data if not specified
    if max_layer is None:
        max_layers_per_focal = [max((lyr for lyr in ldict.values() if lyr >= 0), default=0) for ldict in all_layers.values()]
        max_layer = np.max(max_layers_per_focal)

    # For each focal cell f, compute a C_f(k) vector
    # Shape: focal x layer  (sparse – use a dict of arrays)
    focal_curves: dict[int, np.ndarray] = {}   # focal -> C_f[0..max_layer]

    if subsample:
        ids_to_sample = np.random.choice(valid_indices, size = subsample, replace = False)
    else:
        ids_to_sample = valid_indices

    print('Computing correlation vectors for all individuals.')
    for focal_idx in tqdm(ids_to_sample):
        
        layer_dict = all_layers[focal_idx]
        layer_contribs = _build_pair_contributions(values, layer_dict, param_type, mu, sigma2)

        if not layer_contribs:
            continue

        # Build a fixed-length array; NaN for missing layers
        curve = np.full(max_layer + 1, np.nan)

        for k, cs in layer_contribs.items():
            curve[k] = np.mean(cs)

        focal_curves[focal_idx] = curve

    if not focal_curves:
        raise ValueError("No valid focal cells found.")

    # Stack into matrix and compute mean / SE across focal cells
    mat = np.vstack(list(focal_curves.values()))  # (n_focal, max_layer+1)
    ks_all = np.arange(max_layer + 1, dtype=int)

    n_focal_per_layer = np.sum(~np.isnan(mat), axis=0)
    C = np.nanmean(mat, axis=0)
    C_se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(n_focal_per_layer, 1))

    # Count total pairs per layer (summed across all focal cells)
    pair_counts = np.zeros(max_layer + 1, dtype=int)
    for layer_dict in all_layers.values():
        for layer in layer_dict.values():
            if 0 <= layer <= max_layer:
                pair_counts[layer] += 1

    # Drop layers with too few pairs
    keep = pair_counts >= min_pairs_per_layer
    ks = ks_all[keep]
    C = C[keep]
    C_se = C_se[keep]
    n_pairs = pair_counts[keep]

    return ks, C, C_se, n_pairs, ks_all, mat


# ---------------------------------------------------------------------------
# Convenience wrapper: compute all_layers efficiently
# ---------------------------------------------------------------------------

def get_global_voronoi_corrs(vor: Voronoi, valid_indices:np.ndarray, z_vals: np.ndarray, tolerance:float = 1e-6, param_type: Literal["scalar", "circular"] = "scalar", max_layer:int | None = None, min_pairs_per_layer:int = 5, subsample:int | None = None):
    '''
    cells:
        List of Polygons of length N representing Voronoi neighbourhoods -> MUST be in same order as z_vals.
    z_vals:
        Numpy array of length N representing value of parameter that will be correlated - > MUST be in same order as cells.
    tolerance:
        As defined in build_adjacency_graph.
    param_type, max_layer, min_pairs_per_layer
        As defined in voronoi_corr_vectors.
        '''
    
    # Get adjacency graph which will be used to construct layers for each individual
    print('Computing adjacency graph.')
    graph = build_adjacency_graph(vor, valid_indices, tolerance)

    # Get dictionary of neighbour layer values relative to each individual
    print('Computing layers relative to each individual.')
    all_layers = {f: compute_voronoi_layers(graph, f) for f in tqdm(graph)}

    # Get correlation vectors for each individual
    ks, C, C_se, n_pairs, ks_all, mat = voronoi_corr_vectors(z_vals, valid_indices, all_layers, param_type, max_layer, min_pairs_per_layer, subsample)

    return ks, C, C_se, n_pairs, ks_all, mat


def metric_corr_vectors(
    positions: np.ndarray,
    z_vals: np.ndarray,
    param_type: Literal["scalar", "circular"] = "scalar",
    n_bins: int = 20,
    min_pairs_per_bin: int = 5,
    min_partners: int = 3,
    subsample: int | None = None,
    axis: int | None = None) -> dict:
    """
    Compute per-individual correlation vectors over Euclidean distance bins.

    For each individual, correlates its value against partners binned by
    distance — directly analogous to Voronoi layer correlations.

    Returns
    -------
    dict with keys:
        bin_centers      – (M,)        midpoint of each retained bin
        C                – (M,)        mean correlation across individuals
        C_se             – (M,)        SE across individuals
        n_pairs          – (M,)        total pair count per bin
        n_individuals    – (M,)        number of individuals with valid corrs
        individual_ids   – (N,)        index of each individual
        individual_corrs – (N, M)      per-individual correlation vectors;
                                       NaN where an individual had too few
                                       partners in that bin
    """
    N = len(z_vals)
    if N < 2:
        empty = np.array([])
        return dict(bin_centers=empty, C=empty, C_se=empty, n_pairs=empty,
                    n_individuals=empty, individual_ids=empty,
                    individual_corrs=np.full((0, 0), np.nan))

    # Precompute ALL pairwise distances as a full matrix for fast per-individual lookup
    if axis == None:
        diff  = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
        dists = np.sqrt((diff**2).sum(axis=-1))                 # (N, N)
        np.fill_diagonal(dists, np.nan)                         # exclude self
    else:
        # Signed displacement matrix: disp[i,j] = pos[i,axis] - pos[j,axis]
        dists = positions[:, axis, None] - positions[None, :, axis]  # (N, N)
        np.fill_diagonal(dists, np.nan)

    # Global bin edges from the distribution of all pairwise distances
    all_dists = dists[~np.isnan(dists)]
    d_min, d_max = all_dists.min(), all_dists.max()
    edges = np.linspace(d_min, d_max, n_bins + 1)
    n_actual_bins = len(edges) - 1

    # Precompute pairwise similarity matrix
    if param_type == "circular":
        # cos(z_i - z_j): 1 when aligned, -1 when opposite
        sim_matrix = np.cos(z_vals[:, None] - z_vals[None, :])  # (N, N)
    else:
        # Normalised: (z_i - mean)(z_j - mean) / var
        z_norm = (z_vals - np.mean(z_vals)) / (np.std(z_vals) + 1e-12)
        sim_matrix = z_norm[:, None] * z_norm[None, :]           # (N, N)

    np.fill_diagonal(sim_matrix, np.nan)

    # Build (N, n_bins) correlation vector matrix
    ind_corr_matrix = np.full((N, n_actual_bins), np.nan)

    # Subsample
    if subsample is not None and subsample < N:
        rng   = np.random.default_rng()
        sel   = rng.choice(N, size=subsample, replace=False)
    else:
        sel = np.arange(N)

    for i in sel:
        for b in range(n_actual_bins):
            lo, hi = edges[b], edges[b + 1]
            partner_mask = (dists[i] >= lo) & (dists[i] <= hi) if b == n_actual_bins - 1 \
                           else (dists[i] >= lo) & (dists[i] < hi)

            partners = np.where(partner_mask)[0]
            if len(partners) < min_partners:
                continue

            # Mean pairwise similarity between i and all partners in this bin
            ind_corr_matrix[i, b] = np.mean(sim_matrix[i, partners])

    # Bin-level summaries, dropping bins below threshold
    bin_centers  = []
    C_list       = []
    C_se_list    = []
    n_pairs_list = []
    n_ind_list   = []
    kept_bins    = []

    for b in range(n_actual_bins):
        lo, hi = edges[b], edges[b + 1]
        bin_mask = (dists >= lo) & (dists <= hi) if b == n_actual_bins - 1 \
                   else (dists >= lo) & (dists < hi)
        n_pairs = int(np.sum(bin_mask)) // 2

        if n_pairs < min_pairs_per_bin:
            continue

        col   = ind_corr_matrix[:, b]
        valid = col[~np.isnan(col)]

        if len(valid) == 0:
            continue

        bin_centers.append(0.5 * (edges[b] + edges[b + 1]))
        C_list.append(np.mean(valid))
        C_se_list.append(np.std(valid) / np.sqrt(len(valid)))
        n_pairs_list.append(n_pairs)
        n_ind_list.append(len(valid))
        kept_bins.append(b)

    return dict(
        bin_centers      = np.array(bin_centers),
        C                = np.array(C_list),
        C_se             = np.array(C_se_list),
        n_pairs          = np.array(n_pairs_list, dtype=int),
        n_individuals    = np.array(n_ind_list,   dtype=int),
        individual_ids   = sel,
        individual_corrs = ind_corr_matrix[:, kept_bins],  # (N, M)
    )