"""Interactive metric correlation plots around reflection events.

The public function below expects ``rel_reflections`` to contain absolute
frame numbers (one per event), as returned by ``find_reflections``.  It pools
events at the same event-relative frame and computes a population correlation
curve for absolute x and y separation. See the docstring for assumptions.
"""
from __future__ import annotations

from typing import Literal, Sequence
import numpy as np
import xarray as xr
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from helper_fns import get_frame_slice
from cluster_analysis import find_reflections
from data_handling import load_preprocessed_data

def _correlation_by_axis_distance(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    axis: int,
    n_bins: int,
    min_partners: int,
    param_type: Literal["scalar", "circular"],
    segment_sizes: Sequence[int] | None = None,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast, chunked version of the signed axis-distance correlation.

    Keeps the estimator in the supplied function: scalar similarities are
    products of globally standardized values; circular similarities are
    cos(value_i - value_j); partner similarities are averaged for each focal
    individual and then averaged over valid focal individuals.

    Pairs never cross ``segment_sizes`` boundaries (each segment is one event
    snapshot). Displacement uses coord[focal] - coord[partner], matching the
    axis branch of ``metric_corr_vectors`` in correlation_length.py.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (x) or 1 (y)")
    if n_bins < 1 or min_partners < 1 or chunk_size < 1:
        raise ValueError("n_bins, min_partners, and chunk_size must be positive")

    coord = np.asarray(x if axis == 0 else y, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    if coord.ndim != 1 or val.ndim != 1 or len(coord) != len(val):
        raise ValueError("x, y, and values must be parallel 1-D arrays")
    valid = np.isfinite(coord) & np.isfinite(val)

    # Keep segment boundaries correct when invalid points are removed.
    if segment_sizes is None:
        sizes = np.array([len(coord)], dtype=np.int64)
    else:
        sizes_in = np.asarray(segment_sizes, dtype=np.int64)
        if np.any(sizes_in < 0) or sizes_in.sum() != len(coord):
            raise ValueError("segment_sizes must be nonnegative and sum to len(x)")
        old_edges = np.r_[0, np.cumsum(sizes_in)]
        # Include beginning/end boundaries after filtering; searchsorted gives
        # the number of retained observations before each original boundary.
        kept_boundaries = np.searchsorted(np.flatnonzero(valid), old_edges[1:-1], side="left")
        sizes = np.diff(np.r_[0, kept_boundaries, int(valid.sum())])

    coord, val = coord[valid], val[valid]
    n = len(val)
    if n < 2:
        return np.array([]), np.array([]), np.array([])
    if sizes.sum() != n:
        raise RuntimeError("Internal segment-size mismatch after filtering")

    # Signed displacement edges span all within-event coordinate differences.
    lo_coord = min(float(coord[a:b].min()) for a, b in zip(np.r_[0, np.cumsum(sizes)[:-1]], np.cumsum(sizes)) if b > a)
    hi_coord = max(float(coord[a:b].max()) for a, b in zip(np.r_[0, np.cumsum(sizes)[:-1]], np.cumsum(sizes)) if b > a)
    lo, hi = lo_coord - hi_coord, hi_coord - lo_coord
    if not np.isfinite(lo + hi) or lo == hi:
        return np.array([]), np.array([]), np.array([])
    edges = np.linspace(lo, hi, n_bins + 1)
    width = (hi - lo) / n_bins

    if param_type == "scalar":
        sd = val.std()
        if sd == 0:
            nan = np.full(n_bins, np.nan)
            return (edges[:-1] + edges[1:]) / 2, nan, nan.copy()
        z = (val - val.mean()) / sd
    elif param_type == "circular":
        z = None
    else:
        raise ValueError("param_type must be 'scalar' or 'circular'")

    focal_sum = np.zeros(n_bins, dtype=np.float64)
    focal_sum_sq = np.zeros(n_bins, dtype=np.float64)
    focal_count = np.zeros(n_bins, dtype=np.int64)
    seg_start = 0

    for seg_n in sizes:
        seg_n = int(seg_n)
        seg_end = seg_start + seg_n
        c = coord[seg_start:seg_end]
        v = val[seg_start:seg_end]
        zs = z[seg_start:seg_end] if z is not None else None

        for start in range(0, seg_n, chunk_size):
            stop = min(start + chunk_size, seg_n)
            # Temporary arrays are only chunk_size x seg_n, never N x N.
            delta = c[start:stop, None] - c[None, :]
            bin_id = np.floor((delta - lo) / width).astype(np.int32)
            # Put the exact upper endpoint in the final bin; remove self-pairs
            # and any pair outside the shared range.
            bin_id[delta == hi] = n_bins - 1
            row_id = np.arange(start, stop)
            bin_id[np.arange(stop - start), row_id] = -1
            in_range = (bin_id >= 0) & (bin_id < n_bins)

            if not in_range.any():
                continue
            # Flatten (focal-row, distance-bin) into one bincount index. This
            # replaces n_bins separate masks and reductions for each chunk.
            focal_id = np.broadcast_to(np.arange(stop - start)[:, None], bin_id.shape)
            group = (focal_id * n_bins + bin_id)[in_range]
            if zs is None:
                similarity = np.cos(v[start:stop, None] - v[None, :])[in_range]
            else:
                similarity = (zs[start:stop, None] * zs[None, :])[in_range]

            group_count = np.bincount(group, minlength=(stop - start) * n_bins)
            group_sum = np.bincount(group, weights=similarity,
                                    minlength=(stop - start) * n_bins)
            group_count = group_count.reshape(stop - start, n_bins)
            group_sum = group_sum.reshape(stop - start, n_bins)
            good = group_count >= min_partners
            focal_means = np.divide(group_sum, group_count,
                                    out=np.zeros_like(group_sum), where=good)
            focal_sum += focal_means.sum(axis=0)
            focal_sum_sq += np.square(focal_means).sum(axis=0)
            focal_count += good.sum(axis=0)

        seg_start = seg_end

    corr = np.divide(focal_sum, focal_count, out=np.full(n_bins, np.nan), where=focal_count > 0)
    # Standard error across valid focal-level correlations in each bin.
    # Bins with fewer than two focal observations cannot have an estimated SE.
    variance_num = focal_sum_sq - np.divide(
        focal_sum * focal_sum, focal_count,
        out=np.zeros(n_bins, dtype=np.float64), where=focal_count > 0,
    )
    sample_var = np.divide(
        np.maximum(variance_num, 0.0), focal_count - 1,
        out=np.full(n_bins, np.nan), where=focal_count > 1,
    )
    corr_se = np.sqrt(np.divide(
        sample_var, focal_count, out=np.full(n_bins, np.nan), where=focal_count > 0,
    ))
    return (edges[:-1] + edges[1:]) / 2, corr, corr_se

def plot_reflection_metric_correlations(
    ds: xr.Dataset,
    output_dir:str,
    *,
    variable: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    density: str = "density_voronoi_None",
    x: str = "x_raw",
    y: str = "y_raw",
    density_quantile_bands: Sequence[tuple[float, float]] = ((0, 20), (40, 60), (80, 100)),
    n_bins: int = 20,
    min_partners: int = 3,
    mirror_side:str = 'left',
    param_type: Literal["scalar", "circular"] = "scalar",
    fps:float = 5,
    arena_center_m:np.ndarray = np.array([0.01, 0]),
    subsample:int = 1
) -> go.Figure:
    """Build a 4-row x 2-column Plotly figure with an event-relative slider.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain coordinates/data variables ``frame`` and ``id``, plus x,
        y, ``variable`` and ``density`` with dimensions frame x id.
    rel_reflections : sequence of int
        Absolute frame indices of reflection events, e.g. the first item
        returned by ``find_reflections``. Events are pooled together.
    reflection_sides : sequence, optional
        Side labels parallel to events. Used only with ``event_sides``.
    variable, density, x, y : str
        Dataset variable names. x/y are metric coordinates; density selects
        focal animals for quantile rows.
    relative_frames : sequence of int
        Event-relative frames available in the slider.
    density_quantile_bands : 3 percentile pairs
        Rows 2-4, calculated from finite density values at each sampled frame.
        Defaults to 0–20%, 40–60%, and 80–100%.
    param_type : {"scalar", "circular"}
        ``circular`` treats ``variable`` as radians, following the source
        module's cosine-difference definition.
    event_sides : sequence, optional
        If supplied, retain only events whose parallel ``reflection_sides``
        label occurs in this sequence.
    mirror_side : str
        Side label whose x coordinates are multiplied by -1 to reflect it
        into the opposite-side coordinate system. Defaults to ``"left"``.

    Notes
    -----
    Events whose target absolute frame is outside ``ds.frame`` are skipped.
    Density strata select focal individuals; their partners can be any valid
    individual, so each row answers how focal animals in that density range
    correlate with their spatial neighbours. Each event contributes each
    focal animal once. This function pools correlation curves and does not
    currently display standard errors.
    """
    # Check validity of inputs
    required = (variable, density, x, y)
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"Missing dataset variables: {missing}")
    if len(density_quantile_bands) != 3:
        raise ValueError("density_quantile_bands must contain exactly three percentile pairs")
    if n_bins < 1 or min_partners < 1:
        raise ValueError("n_bins and min_partners must be positive")

    # Get absolute and relative frames
    abs_frames, ds_idcs = get_frame_slice(ds, rel_start=start_frame, rel_end=end_frame, in_function_subsample=subsample)

    # Get reflection events and their sides, if any, within the relative frame range
    rel_reflections, _, marching_period, reflection_sides = find_reflections(ds, fps, start_frame, end_frame, arena_center_m, subsample)
    half_period = marching_period // 2
    frame_step = max(1, int(round(np.median(np.diff(abs_frames)))))
    relative_frames=np.arange(-half_period, half_period + 1, frame_step)

    # Keep only events with a complete relative-frame window
    event_idcs = np.array([i for i, event_frame in enumerate(rel_reflections) if all(int(event_frame + relative_frame) in ds_idcs for relative_frame in relative_frames)]).astype(int)

    # Put every input into frame x id order once, so the inner loop only uses numpy arrays.
    arrays = {name: np.asarray(ds[name].transpose("frame", "id").values) for name in required}

    def compute_curves_for_relative_frame(rel):
        frame_curves = {}

        for row in range(4):
            for axis in range(2):

                sample_x, sample_y, sample_v, sample_d = [], [], [], []
                sizes = []

                # Iterate over events
                for event_idx in event_idcs:
                    pos = int(rel_reflections[event_idx] + rel)

                    xx, yy = arrays[x][pos].copy(), arrays[y][pos].copy()

                    # Reflect one side such that all of them line up
                    if reflection_sides[event_idx] == mirror_side.lower():
                        xx = -1*(xx - arena_center_m[0]) + arena_center_m[0]

                    vv, dd = arrays[variable][pos], arrays[density][pos]

                    # Only compute points for which all variables are valid
                    valid = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(vv) & np.isfinite(dd)

                    if row:
                        # Limit correlation calculation to certain density quantile bands
                        lo_q, hi_q = density_quantile_bands[row - 1]
                        lo, hi = np.nanpercentile(dd[valid], [lo_q, hi_q]) if valid.any() else (np.nan, np.nan)

                        # Half-open percentile band except for the final endpoint.
                        selected = (dd >= lo) & ((dd <= hi) if hi_q == 100 else (dd < hi))
                        valid &= selected

                    if valid.sum() >= 2:
                        sample_x.append(xx[valid]); sample_y.append(yy[valid])
                        sample_v.append(vv[valid]); sample_d.append(dd[valid])
                        sizes.append(int(valid.sum()))

                if sample_v:
                    # Concatenate data and compute correlation by axis
                    xx = np.concatenate(sample_x); yy = np.concatenate(sample_y); vv = np.concatenate(sample_v)
                    centers, corr, corr_se = _correlation_by_axis_distance(xx, yy, vv, axis=axis, n_bins=n_bins, min_partners=min_partners, param_type=param_type, segment_sizes=sizes)
                    curves[(int(rel), row, axis)] = (centers, corr, corr_se)

                else:
                    centers, corr, corr_se = np.array([]), np.array([]), np.array([])

                frame_curves[(int(rel), row, axis)] = (centers, corr, corr_se)

        return frame_curves

    # Compute each (relative frame, row, axis) curve in parallel and retain it for slider updates.
    curves: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with tqdm_joblib(tqdm(desc='Compute correlation by relative frame: ', total=len(relative_frames))):
        results = Parallel(n_jobs=4, prefer="threads", batch_size=1)(delayed(compute_curves_for_relative_frame)(int(rel)) for rel in relative_frames)
    curves = {key: value for frame_curves in results for key, value in frame_curves.items()}
        
    fig = make_subplots(rows=4, cols=2, shared_xaxes=False,
                        subplot_titles=("All density", "All density",
                                        *[f"Density {a}–{b}th percentile" for a, b in density_quantile_bands for _ in (0, 1)]))
    
    row_labels = ["All data", *[f"Density {a}–{b}%" for a, b in density_quantile_bands]]
    for row in range(4):
        for axis in range(2):
            centers, corr, corr_se= curves[(int(relative_frames[0]), row, axis)] if len(relative_frames) else (np.array([]), np.array([]))
            fig.add_trace(go.Scatter(x=centers, y=corr, mode="lines+markers", 
                                     error_y=dict(type="data", array=corr_se, visible=True),name=row_labels[row],
                                     showlegend=False, connectgaps=False), row=row+1, col=axis+1)
            fig.update_xaxes(title_text="Δx (m)" if axis == 0 else "Δy (m)", row=row+1, col=axis+1)
            fig.update_yaxes(title_text="Correlation" if axis == 0 else None, range=[-1, 1], row=row+1, col=axis+1)

    # Each slider step updates the eight traces in place, preserving the subplot layout.
    steps = []
    for rel in relative_frames:
        xs, ys, errors = [], [], []
        for row in range(4):
            for axis in range(2):
                centers, corr, corr_se = curves[(int(rel), row, axis)]
                xs.append(centers); ys.append(corr); errors.append(corr_se)
        steps.append(dict(method="update", label=str(int(rel)),
                          args=[{"x": xs, "y": ys, "error_y.array": errors}, {"title": f"Metric correlations at relative frame {int(rel)}"}]))
    fig.update_layout(title=f"Metric correlations at relative frame {int(relative_frames[0])}" if len(relative_frames) else "Metric correlations",
                      sliders=[dict(active=0, currentvalue={"prefix": "Frames from reflection: "}, steps=steps)],
                      height=1000, width=1100, template="plotly_white", margin=dict(t=110, b=100))

    save_dir = os.path.join(output_dir, f"correlations/{variable}_{abs_frames[0]}_{abs_frames[-1]}_{fps:.1f}Hz")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{variable}_reflection_aligned_metric_correlations.html")
    fig.write_html(save_path)
    print(f"Reflection-aligned correlations saved to {save_path}")

if __name__ == "__main__":
    # Example usage with a preprocessed dataset and reflection events.
    ds_name = f'/bb_kp/20230329/0_6000_5.0Hz'
    ds = load_preprocessed_data(ds_name + '.hdf5')
    plot_reflection_metric_correlations(ds, output_dir='/output/20230329/both/', variable='ang_speed_spline',
                                              density_quantile_bands=((0, 20), (40, 60), (80, 100)),
                                              n_bins=20, min_partners=3, param_type='scalar')
