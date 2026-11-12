'''_____________________________________________________IMPORTS____________________________________________________________'''

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from tqdm import tqdm

import data_handling as dh
from animate import animate_orientations_fast

'''_____________________________________________________FUNCTIONS____________________________________________________________'''


def _combine_relevant_files(a_dir: str, b_frames: Iterable[int]) -> xr.Dataset:
    """Load A frames present in B; the first sorted file wins on overlap."""
    a_files = sorted(Path(a_dir).glob("*.hdf5"))
    if not a_files:
        raise ValueError(f"No .hdf5 files found in {a_dir}.")

    wanted = set(np.asarray(list(b_frames)).tolist())
    seen: set = set()
    segments = []
    for path in a_files:
        ds = dh.load_preprocessed_data(str(path))
        if "frame" not in ds.dims or "id" not in ds.dims:
            ds.close()
            raise ValueError(f"{path} must have 'frame' and 'id' dimensions.")
        frames = np.asarray(ds.frame.values)
        keep = np.array([f in wanted and f not in seen for f in frames])
        seen.update(frames[keep].tolist())
        if keep.any():
            segments.append(ds.isel(frame=np.flatnonzero(keep)).transpose("frame", "id"))
        else:
            ds.close()

    if not segments:
        raise ValueError("No overlapping frames found between A files and B.")
    return xr.concat(segments, dim="frame", join="outer", data_vars="all").sortby("frame")


def _choose_positions(ds: xr.Dataset, x_name: str, y_name: str) -> tuple[str, str]:
    """Use requested position channels, or infer a common pair."""
    if x_name is not None or y_name is not None:
        if x_name is None or y_name is None:
            raise ValueError("Specify both x and y names, or neither.")
        if x_name not in ds or y_name not in ds:
            raise KeyError(f"Position variables {x_name!r}, {y_name!r} not found.")
        return x_name, y_name
    for pair in (("centroid_x", "centroid_y"), ("x", "y"), ("x_raw", "y_raw")):
        if all(name in ds for name in pair):
            return pair
    raise KeyError("Could not infer position channels; pass x and y names.")


def _match_frame(ap: np.ndarray, bp: np.ndarray, max_distance: float):
    """Solve the one-frame assignment on only spatially plausible pairs."""
    a_valid = np.isfinite(ap).all(axis=1)
    b_valid = np.isfinite(bp).all(axis=1)
    a_idx = np.flatnonzero(a_valid)
    b_idx = np.flatnonzero(b_valid)
    if not len(a_idx) or not len(b_idx):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    a_points, b_points = ap[a_idx], bp[b_idx]
    tree = cKDTree(a_points)
    candidates = tree.query_ball_point(b_points, r=max_distance)

    rows, cols, costs = [], [], []
    for row, neighbors in enumerate(candidates):
        if neighbors:
            cols_for_row = np.asarray(neighbors, dtype=int)
            distances = np.linalg.norm(a_points[cols_for_row] - b_points[row], axis=1)
            rows.extend([row] * len(cols_for_row))
            cols.extend(cols_for_row.tolist())
            # Sparse matching treats stored zeros specially; the tiny offset
            # preserves distance ordering while making exact overlaps valid.
            costs.extend((distances + 1e-12).tolist())

    # Each B row can take its own unmatched column. A penalty larger than the
    # maximum possible total distance makes the solver maximize valid matches
    # first, then minimize their distances.
    penalty = (min(len(a_idx), len(b_idx)) + 1) * (max_distance + 1.0)
    rows.extend(range(len(b_idx)))
    cols.extend(len(a_idx) + np.arange(len(b_idx)))
    costs.extend([penalty] * len(b_idx))
    graph = coo_matrix((costs, (rows, cols)), shape=(len(b_idx), len(a_idx) + len(b_idx))).tocsr()
    assigned_rows, assigned_cols = min_weight_full_bipartite_matching(graph)
    matched = assigned_cols < len(a_idx)
    return b_idx[assigned_rows[matched]], a_idx[assigned_cols[matched]]


def associate_detection_metrics(
    a_dir: str,
    b_file: str,
    output_name: str | Path,
    *,
    a_x: str | None = "centroid_x",
    a_y: str | None = "centroid_y",
    b_x: str | None = "x_raw",
    b_y: str | None = "y_raw",
    metric_vars: Sequence[str] | None = None,
    max_distance: float | None = None,
    params: dict | None = None,
    n_jobs: int = 1,
) -> xr.Dataset:
    """Add A's per-detection metrics to B using framewise spatial matching.

    ``max_distance`` bounds candidate pairs and is required when either input
    has more than 500 detections in a frame. This keeps the assignment sparse
    instead of allocating a multi-hundred-megabyte dense distance matrix.
    """
    if max_distance is not None and max_distance <= 0:
        raise ValueError("max_distance must be positive or None.")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer.")

    b = dh.load_preprocessed_data(str(b_file)).transpose("frame", "id")
    a = _combine_relevant_files(a_dir, b.frame.values)
    ax, ay = _choose_positions(a, a_x, a_y)
    bx, by = _choose_positions(b, b_x, b_y)

    if metric_vars is None:
        metric_vars = [name for name in a.data_vars if name not in {ax, ay}]
    missing = set(metric_vars) - set(a.data_vars)
    if missing:
        raise KeyError(f"A is missing metric variables: {sorted(missing)}")
    collisions = set(metric_vars) & set(b.data_vars)
    if collisions:
        raise ValueError(f"Metric variables already exist in B: {sorted(collisions)}")

    if max_distance is None and max(a.sizes["id"], b.sizes["id"]) > 500:
        raise ValueError("For >500 detections per frame, set max_distance to enable sparse matching.")
    if max_distance is None:
        # Small-input fallback: a finite radius covering the frame extent.
        all_xy = np.concatenate([
            np.column_stack((a[ax].values.ravel(), a[ay].values.ravel())),
            np.column_stack((b[bx].values.ravel(), b[by].values.ravel())),
        ])
        all_xy = all_xy[np.isfinite(all_xy).all(axis=1)]
        max_distance = float(np.linalg.norm(np.ptp(all_xy, axis=0))) if len(all_xy) else 1.0

    result = b.copy()
    for name in metric_vars:
        if set(a[name].dims) != {"frame", "id"}:
            raise ValueError(f"Metric {name!r} must have exactly 'frame' and 'id' dimensions.")
        result[name] = (("frame", "id"), np.full((b.sizes["frame"], b.sizes["id"]), np.nan))
        result[name].attrs = dict(a[name].attrs)

    a_frame_i = {f: i for i, f in enumerate(a.frame.values.tolist())}
    b_frame_i = {f: i for i, f in enumerate(b.frame.values.tolist())}
    common = [(a_frame_i[f], b_frame_i[f]) for f in b.frame.values.tolist() if f in a_frame_i]
    axv, ayv = a[ax].transpose("frame", "id").values, a[ay].transpose("frame", "id").values
    bxv, byv = b[bx].values, b[by].values

    def work(pair):
        ai, bi = pair
        targets, sources = _match_frame(
            np.column_stack((axv[ai], ayv[ai])),
            np.column_stack((bxv[bi], byv[bi])), max_distance,
        )
        return ai, bi, targets, sources

    workers = min(os.cpu_count() or 1, len(common)) if n_jobs == -1 else n_jobs
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(work, common) if pool else map(work, common)
        metric_arrays = {name: a[name].transpose("frame", "id").values for name in metric_vars}
        for ai, bi, targets, sources in tqdm(results, total=len(common), desc="Matching frames"):
            for name, values in metric_arrays.items():
                result[name].values[bi, targets] = values[ai, sources]
    finally:
        if pool:
            pool.shutdown()

    output_name = Path(output_name + f'{int(b.frame.values[0])}_{int(b.frame.values[-1])}_5.0Hz')
    output_name.parent.mkdir(parents=True, exist_ok=True)
    (output_name.parent / "params").mkdir(parents=True, exist_ok=True)

    # Switch coords to (id, frame) order for consistency with other datasets.
    for var in result.data_vars:
        result[var] = result[var].transpose('id', 'frame')

    dh.save_ds(result, str(output_name), params or {})
    a.close()
    b.close()
    return result



kp_dir = '/keypoints/20230329/dewarped/'

for i in range(3, 11):
    bb_file = f'/bb/20230329/dewarped/batch_{i}_5.0Hz.hdf5'
    output_dir = f'/bb_kp/20230329/'
    merged = associate_detection_metrics(a_dir=kp_dir, b_file=bb_file, output_name = output_dir, max_distance=0.02)

    ds = dh.load_preprocessed_data(str(output_dir + f'{int(merged.frame.values[0])}_{int(merged.frame.values[-1])}_5.0Hz.hdf5'))

    ds = ds.drop_vars([var for var in ds.data_vars if var not in ['x_raw', 'y_raw', 'theta']])

    dh.save_ds(ds, f'/bb_kp/20230329/raw_{int(merged.frame.values[0])}_{int(merged.frame.values[-1])}_5.0Hz', {})

# bb_file = '/bb/20230329/dewarped/batch_2_5.0Hz.hdf5'
# output_dir = '/bb_kp/20230329/'
# merged = associate_detection_metrics(a_dir=kp_dir, b_file=bb_file, output_name = output_dir, max_distance=0.02)

# plots_path = '/output/20230329/both/'
# ds = dh.load_preprocessed_data(str(output_dir + '5900_11900_5.0Hz.hdf5'))
# animate_orientations_fast(ds, 'spline', plots_path, 0, 300)