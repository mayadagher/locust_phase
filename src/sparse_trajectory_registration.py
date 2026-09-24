#!/usr/bin/env python3
"""Match selected MoCap trajectories to trajectories detected in video.

Video and MoCap samples are synchronized using their real timestamps. Candidate
pairs are then compared after removing translation, rotation, and isotropic
scale. Accepted matches are used to estimate the spatial transform from MoCap
coordinates into dewarped video coordinates.
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from data_handling import load_preprocessed_data
from helper_fns import get_frame_slice
from time_sync import (
    downsample_mocap_for_video,
    get_temporal_overlap,
    get_times_mocap,
    get_times_video,
)
from dewarping import dewarp_pts

import matplotlib.pyplot as plt
from tqdm import tqdm

Array = np.ndarray


@dataclass(frozen=True)
class SpatialTransform:
    """A fitted map using ``video_xy = mocap_xy @ matrix.T + offset``."""

    name: str
    matrix: Array
    offset: Array
    allow_reflection: bool | None = None

    def apply(self, points: Array) -> Array:
        points = np.asarray(points, dtype=float)
        return points @ self.matrix.T + self.offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "matrix": self.matrix.tolist(),
            "offset": self.offset.tolist(),
            "allow_reflection": self.allow_reflection,
            "determinant": float(np.linalg.det(self.matrix)),
        }


@dataclass
class RegistrationResult:
    """Outputs from trajectory assignment and spatial-model comparison."""

    matches: pd.DataFrame
    paired_points: pd.DataFrame
    model_metrics: pd.DataFrame
    transforms: dict[str, SpatialTransform]
    config: dict[str, Any]

    def save(self, output_dir: str | Path) -> None:
        """Write human-readable result tables and transform parameters."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.matches.to_csv(output_dir / "matches.csv", index=False)
        self.paired_points.to_csv(output_dir / "paired_points.csv", index=False)
        self.model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
        with (output_dir / "transforms.json").open("w", encoding="utf-8") as f:
            json.dump(
                {name: fit.to_dict() for name, fit in self.transforms.items()},
                f,
                indent=2,
            )
        summary = {
            "config": _json_safe(self.config),
            "n_assignments": int(len(self.matches)),
            "n_accepted": int(self.matches["accepted"].sum())
            if not self.matches.empty
            else 0,
            "recommended_model": _recommend_model(self.model_metrics),
        }
        with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


@dataclass(frozen=True)
class _Trajectory:
    track_id: Any
    xy: Array
    valid: Array


def _json_safe(value: Any) -> Any:
    """Convert result metadata into values accepted by ``json.dump``."""

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    return value


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Check the fixed MoCap table format used by this analysis."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"MoCap dataframe is missing columns: {missing}")


def find_sparse_mocap_ids(
    mocap_df: pd.DataFrame,
    n_ids: int = 25,
    *,
    k: int = 5,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_step: int = 25,
    min_frame_fraction: float = 0.5,
) -> pd.DataFrame:
    """Find MoCap animals that are consistently far from tagged neighbours.

    Sparsity is measured as the distance to the kth-nearest tagged animal in
    each sampled frame. The final score is the 25th percentile of that distance
    across time, so a high score requires an animal to be isolated consistently
    rather than in only a few frames.

    Returns a table of the ``n_ids`` sparsest animals, ordered from most to least
    sparse. Pass the ``particle_id`` column to ``mocap_candidate_ids``.
    """

    # STEP 1: Validate the fixed MoCap format and selection parameters
    _require_columns(mocap_df, ("frame", "particle_id", "x", "y"))
    if n_ids < 1:
        raise ValueError("n_ids must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if not 0 < min_frame_fraction <= 1:
        raise ValueError("min_frame_fraction must be in (0, 1]")

    # STEP 2: Select an evenly spaced set of frames from the requested interval
    frames = np.sort(mocap_df["frame"].unique())
    if frame_start is not None:
        frames = frames[frames >= frame_start]
    if frame_end is not None:
        frames = frames[frames <= frame_end]
    frames = frames[::frame_step]
    if len(frames) == 0:
        raise ValueError("No MoCap frames fall inside the requested interval")

    work = mocap_df.loc[
        mocap_df["frame"].isin(frames),
        ["frame", "particle_id", "x", "y"],
    ].dropna(subset=["x", "y"])

    # STEP 3: Measure kth-neighbour distance separately in each sampled frame
    frame_scores = []
    for frame, group in work.groupby("frame", sort=False):
        if len(group) <= k:
            continue

        points = group[["x", "y"]].to_numpy(dtype=float)
        distances, _ = cKDTree(points).query(points, k=k + 1)
        frame_scores.append(
            pd.DataFrame(
                {
                    "particle_id": group["particle_id"].to_numpy(),
                    "frame": frame,
                    "kth_neighbor_distance": distances[:, k],
                }
            )
        )

    if not frame_scores:
        raise ValueError(f"No sampled frame contains more than {k} animals")

    # STEP 4: Keep animals observed often enough and rank persistent isolation
    scores = pd.concat(frame_scores, ignore_index=True)
    summary = (
        scores.groupby("particle_id", as_index=False)
        .agg(
            sparse_score=("kth_neighbor_distance", lambda x: x.quantile(0.25)),
            median_neighbor_distance=("kth_neighbor_distance", "median"),
            n_scored_frames=("frame", "nunique"),
        )
    )
    min_frames = int(math.ceil(len(frames) * min_frame_fraction))
    summary = summary.loc[summary["n_scored_frames"] >= min_frames]
    if summary.empty:
        raise ValueError(
            "No animal was present in enough sampled frames; reduce "
            "min_frame_fraction"
        )

    return summary.sort_values(
        ["sparse_score", "median_neighbor_distance"],
        ascending=False,
        ignore_index=True,
    ).head(n_ids)


def _validate_video_dataset(video_ds: xr.Dataset) -> None:
    """Check the fixed video dataset format used by this analysis."""

    for coord in ("id", "frame"):
        if coord not in video_ds.coords:
            raise ValueError(f"Video dataset has no coordinate {coord!r}")

    for variable in ("x_high_ord", "y_high_ord"):
        if variable not in video_ds:
            raise ValueError(f"Video dataset has no variable {variable!r}")
        if video_ds[variable].dims != ("id", "frame"):
            raise ValueError(
                f"{variable!r} must have dimensions ('id', 'frame'); "
                f"got {video_ds[variable].dims}"
            )


def _synchronized_frames(
    video_ds: xr.Dataset,
    *,
    video_folder: str | Path,
    mocap_timestamp_path: str | Path,
    mocap_csv_path: str | Path,
    mocap_fps: float,
    analysis_start: datetime.datetime | None,
    analysis_end: datetime.datetime | None,
    sample_step: int,
) -> tuple[Array, Array, Array, Array]:
    """Find corresponding video and MoCap frames within their real overlap."""

    # STEP 1: Read the absolute video frames represented in the dataset
    abs_video_frames, ds_indices = get_frame_slice(video_ds)
    if not np.allclose(abs_video_frames, video_ds.frame.values):
        raise ValueError("Video frame coordinates must contain integer absolute frames")

    # STEP 2: Get the real timestamp belonging to each dataset frame
    first_frame = int(abs_video_frames.min())
    last_frame = int(abs_video_frames.max())
    consecutive_video_times = get_times_video(
        str(video_folder),
        abs_start_frame=first_frame,
        abs_end_frame=last_frame + 1,
    )
    video_times = consecutive_video_times[abs_video_frames - first_frame]
    mocap_times = get_times_mocap(
        str(mocap_timestamp_path),
        str(mocap_csv_path),
        mocap_fps=mocap_fps,
    )

    # STEP 3: Restrict the analysis to the true temporal overlap
    overlap_start, overlap_end = get_temporal_overlap(video_times, mocap_times)
    start = max(overlap_start, analysis_start) if analysis_start is not None else overlap_start
    end = min(overlap_end, analysis_end) if analysis_end is not None else overlap_end
    if start > end:
        raise ValueError("The requested analysis interval has no temporal overlap")

    in_overlap = (video_times >= start) & (video_times <= end)
    overlap_ds_indices = ds_indices[in_overlap][::sample_step]
    overlap_video_frames = abs_video_frames[in_overlap][::sample_step]
    overlap_video_times = video_times[in_overlap][::sample_step]
    if len(overlap_video_times) == 0:
        raise ValueError("No video frames fall within the temporal overlap")

    # STEP 4: Pair every selected video frame with its closest MoCap frame
    mocap_indices, valid = downsample_mocap_for_video(
        overlap_video_times,
        mocap_times,
        mocap_fps=mocap_fps,
    )
    if not valid.any():
        raise ValueError("No video frames have a sufficiently close MoCap timestamp")

    # MoCap array indices are zero-based, while CSV frame numbers are one-based
    return (
        overlap_ds_indices[valid],
        overlap_video_frames[valid],
        mocap_indices[valid] + 1,
        overlap_video_times[valid],
    )


def _extract_video_trajectories(
    video_ds: xr.Dataset,
    ds_indices: Array,
    candidate_ids: Iterable[Any] | None,
) -> list[_Trajectory]:
    """Extract synchronized video positions for each requested video ID."""

    ids = video_ds.id.to_numpy()
    candidate_set = set(candidate_ids) if candidate_ids is not None else None
    selected = np.asarray(
        [i for i, track_id in enumerate(ids) if candidate_set is None or track_id in candidate_set],
        dtype=int,
    )

    # STEP 1: Allocate one float32 position array. Loading x and y separately
    # into it avoids holding x, y, and a stacked copy in memory simultaneously.
    xy = np.empty((len(selected), len(ds_indices), 2), dtype=np.float32)
    xy[:, :, 0] = video_ds["x_high_ord"].isel(
        id=selected, frame=ds_indices
    ).to_numpy()
    xy[:, :, 1] = video_ds["y_high_ord"].isel(
        id=selected, frame=ds_indices
    ).to_numpy()
    valid = np.isfinite(xy).all(axis=2)

    # STEP 2: Each trajectory keeps a view into the shared position array
    trajectories = []
    for row, i in enumerate(selected):
        trajectories.append(_Trajectory(ids[i], xy[row], valid[row]))

    return trajectories


def _extract_mocap_trajectories(
    mocap_df: pd.DataFrame,
    mocap_frames: Array,
    candidate_ids: Iterable[Any] | None,
) -> list[_Trajectory]:
    """Extract MoCap positions at the frames paired to the video timestamps."""

    _require_columns(mocap_df, ("frame", "particle_id", "x", "y"))
    if mocap_df.duplicated(["particle_id", "frame"]).any():
        raise ValueError("MoCap contains duplicate rows for a particle_id/frame pair")

    candidate_set = set(candidate_ids) if candidate_ids is not None else None
    trajectories = []

    for track_id, group in mocap_df.groupby("particle_id", sort=False):
        if candidate_set is not None and track_id not in candidate_set:
            continue

        # Reindexing preserves the order of the video-paired MoCap frames
        sampled = group.set_index("frame")[["x", "y"]].reindex(mocap_frames)
        xy = sampled.to_numpy(dtype=float)
        trajectories.append(
            _Trajectory(track_id=track_id, xy=xy, valid=np.isfinite(xy).all(axis=1))
        )

    if candidate_set is not None:
        found = {track.track_id for track in trajectories}
        missing = candidate_set - found
        if missing:
            raise ValueError(f"MoCap candidate IDs were not found: {sorted(missing)}")

    return trajectories


def _center_trajectory(track: _Trajectory) -> tuple[Array, Array] | None:
    """Remove translation using the mean of a trajectory's valid positions."""

    if not track.valid.any():
        return None
    center = np.mean(track.xy[track.valid], axis=0)
    return track.xy - center, track.valid.copy()


def _orthogonal_row_map(x: Array, y: Array, *, allow_reflection: bool) -> Array:
    """Return R minimizing ``||x @ R - y||`` for centered/scaled arrays."""

    cross = x.T @ y
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    correction = np.eye(2)
    if not allow_reflection and np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    return u @ correction @ vt


def _pair_row_map(
    x: Array,
    y: Array,
    *,
    alignment_model: Literal["similarity", "affine"],
    allow_reflection: bool,
) -> Array:
    """Fit the pair-specific shape alignment used only during assignment."""

    if alignment_model == "similarity":
        return _orthogonal_row_map(x, y, allow_reflection=allow_reflection)
    if alignment_model == "affine":
        if np.linalg.matrix_rank(x) < 2:
            raise ValueError("Pairwise affine alignment is rank deficient")
        row_map, *_ = np.linalg.lstsq(x, y, rcond=None)
        return row_map
    raise ValueError(f"Unknown pair alignment model: {alignment_model!r}")


def _pair_shape_cost(
    video: _Trajectory,
    mocap: _Trajectory,
    *,
    min_common_samples: int,
    min_common_fraction: float,
    pair_alignment_model: Literal["similarity", "affine"],
    allow_reflection: bool,
    trim_fraction: float,
    coverage_penalty: float,
    min_motion: float,
) -> tuple[float, int, float, float, float]:
    """Compare one synchronized pair after removing translation and scale."""

    common = video.valid & mocap.valid
    n_common = int(common.sum())
    common_fraction = n_common / len(common)
    if n_common < min_common_samples or common_fraction < min_common_fraction:
        return np.inf, n_common, common_fraction, np.nan, np.nan

    # Center each pair on exactly the samples they have in common. This avoids
    # requiring every trajectory to be present at one arbitrary baseline frame.
    y = video.xy[common]
    x = mocap.xy[common]
    y = y - np.mean(y, axis=0)
    x = x - np.mean(x, axis=0)
    video_scale = float(np.sqrt(np.mean(np.sum(y * y, axis=1))))
    mocap_scale = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
    if video_scale <= min_motion or mocap_scale <= min_motion:
        return np.inf, n_common, common_fraction, video_scale, mocap_scale
    y = y / video_scale
    x = x / mocap_scale

    try:
        row_map = _pair_row_map(
            x,
            y,
            alignment_model=pair_alignment_model,
            allow_reflection=allow_reflection,
        )
    except ValueError:
        return np.inf, n_common, common_fraction, video_scale, mocap_scale
    squared_error = np.sum((x @ row_map - y) ** 2, axis=1)
    keep_count = max(min_common_samples, int(math.ceil(n_common * (1.0 - trim_fraction))))
    if keep_count < n_common:
        keep = np.argpartition(squared_error, keep_count - 1)[:keep_count]
        try:
            row_map = _pair_row_map(
                x[keep],
                y[keep],
                alignment_model=pair_alignment_model,
                allow_reflection=allow_reflection,
            )
        except ValueError:
            return np.inf, n_common, common_fraction, video_scale, mocap_scale
        squared_error = np.sum((x[keep] @ row_map - y[keep]) ** 2, axis=1)
    shape_rmse = float(np.sqrt(np.mean(squared_error)))
    cost = shape_rmse + coverage_penalty * (1.0 - common_fraction)
    return cost, n_common, common_fraction, video_scale, mocap_scale


def _cost_matrix(
    video_tracks: list[_Trajectory],
    mocap_tracks: list[_Trajectory],
    *,
    shortlist_size: int,
    shortlist_batch_size: int,
    exhaustive_pair_limit: int,
    **cost_kwargs: Any,
) -> tuple[Array, str]:
    """Score feasible video/MoCap pairs, using a shortlist when needed."""

    shape = (len(video_tracks), len(mocap_tracks))
    costs = np.full(shape, np.inf, dtype=np.float32)
    use_shortlist = (
        shortlist_size > 0
        and shortlist_size < len(video_tracks)
        and len(video_tracks) * len(mocap_tracks) > exhaustive_pair_limit
    )
    if use_shortlist:
        candidate_mask = _radial_shortlist(
            video_tracks,
            mocap_tracks,
            shortlist_size=shortlist_size,
            batch_size=shortlist_batch_size,
            min_common_samples=cost_kwargs["min_common_samples"],
            min_common_fraction=cost_kwargs["min_common_fraction"],
            coverage_penalty=cost_kwargs["coverage_penalty"],
            min_motion=cost_kwargs["min_motion"],
        )
        mode = f"radial shortlist of {shortlist_size} video candidates per MoCap track"
    else:
        candidate_mask = np.ones(shape, dtype=bool)
        mode = "exhaustive"

    for j, mocap in enumerate(mocap_tracks):
        for i in np.flatnonzero(candidate_mask[:, j]):
            video = video_tracks[i]
            costs[i, j] = _pair_shape_cost(video, mocap, **cost_kwargs)[0]
    return costs, mode


def _estimated_assignment_mb(n_video: int, n_mocap: int) -> float:
    """Estimate dense assignment memory, including SciPy's likely work copy."""

    n_real_pairs = n_video * n_mocap
    n_augmented_pairs = (n_video + n_mocap) * n_mocap
    estimated_bytes = 4 * n_real_pairs + 12 * n_augmented_pairs
    return estimated_bytes / 1024**2


def _radial_shortlist(
    video_tracks: list[_Trajectory],
    mocap_tracks: list[_Trajectory],
    *,
    shortlist_size: int,
    batch_size: int,
    min_common_samples: int,
    min_common_fraction: float,
    coverage_penalty: float,
    min_motion: float,
) -> Array:
    """Cheap rotation/reflection/scale-free screening before pairwise SVDs."""

    n_video = len(video_tracks)
    n_mocap = len(mocap_tracks)
    shortlist = np.zeros((n_video, n_mocap), dtype=bool)
    if n_video == 0 or n_mocap == 0:
        return shortlist

    # STEP 1: Precompute radial trajectories in small batches. This keeps the
    # centered coordinate temporary bounded instead of duplicating the complete
    # video position array.
    n_samples = len(video_tracks[0].valid)
    video_valid = np.empty((n_video, n_samples), dtype=bool)
    video_radius = np.empty((n_video, n_samples), dtype=np.float32)
    for start in range(0, n_video, batch_size):
        stop = min(start + batch_size, n_video)
        batch_tracks = video_tracks[start:stop]
        batch_xy = np.stack([track.xy for track in batch_tracks]).astype(
            np.float32, copy=False
        )
        batch_valid = np.stack([track.valid for track in batch_tracks])
        centers = np.nanmean(batch_xy, axis=1, dtype=np.float64).astype(np.float32)
        batch_xy -= centers[:, None, :]
        batch_radius = np.linalg.norm(batch_xy, axis=2)
        batch_radius[~batch_valid] = 0
        video_valid[start:stop] = batch_valid
        video_radius[start:stop] = batch_radius

    # STEP 2: Compare each MoCap radial trajectory with bounded video batches
    for j, mocap in enumerate(mocap_tracks):
        centered = _center_trajectory(mocap)
        if centered is None:
            continue
        mocap_xy, mocap_valid = centered
        mocap_radius = np.linalg.norm(mocap_xy, axis=1)
        screen_cost = np.full(n_video, np.inf, dtype=np.float32)
        for start in range(0, n_video, batch_size):
            stop = min(start + batch_size, n_video)
            batch_valid = video_valid[start:stop]
            batch_radius = video_radius[start:stop]
            common = batch_valid & mocap_valid[None, :]
            n_common = common.sum(axis=1)
            fraction = n_common / n_samples
            denominator = np.maximum(n_common, 1)
            video_rms = np.sqrt(
                np.sum(np.where(common, batch_radius**2, 0.0), axis=1)
                / denominator
            )
            mocap_rms = np.sqrt(
                np.sum(
                    np.where(common, mocap_radius[None, :] ** 2, 0.0),
                    axis=1,
                )
                / denominator
            )
            usable = (
                (n_common >= min_common_samples)
                & (fraction >= min_common_fraction)
                & (video_rms > min_motion)
                & (mocap_rms > min_motion)
            )
            if not usable.any():
                continue

            normalized_video = batch_radius / np.where(
                video_rms > 0, video_rms, 1.0
            )[:, None]
            normalized_mocap = mocap_radius[None, :] / np.where(
                mocap_rms > 0, mocap_rms, 1.0
            )[:, None]
            squared = np.where(
                common,
                (normalized_video - normalized_mocap) ** 2,
                0.0,
            )
            batch_cost = np.full(stop - start, np.inf, dtype=np.float32)
            batch_cost[usable] = (
                np.sqrt(np.sum(squared[usable], axis=1) / n_common[usable])
                + coverage_penalty * (1.0 - fraction[usable])
            )
            screen_cost[start:stop] = batch_cost
        finite_indices = np.flatnonzero(np.isfinite(screen_cost))
        if finite_indices.size:
            keep_count = min(shortlist_size, finite_indices.size)
            local = np.argpartition(screen_cost[finite_indices], keep_count - 1)[:keep_count]
            shortlist[finite_indices[local], j] = True
    return shortlist


def _filter_video_candidates(
    tracks: list[_Trajectory],
    *,
    min_samples: int,
    min_fraction: float,
    min_motion: float,
) -> list[_Trajectory]:
    """Discard tracks that are too short or nearly stationary to identify."""

    selected = []
    for track in tracks:
        centered = _center_trajectory(track)
        if centered is None:
            continue
        xy, valid = centered
        if valid.sum() < min_samples or valid.mean() < min_fraction:
            continue
        rms_motion = float(np.sqrt(np.mean(np.sum(xy[valid] ** 2, axis=1))))
        if rms_motion > min_motion:
            selected.append(track)
    return selected


def _competitor_margin(costs: Array, row: int, col: int) -> tuple[float, float, float]:
    """Compare a match with its closest video and MoCap alternatives."""

    chosen = costs[row, col]
    row_others = np.delete(costs[row], col)
    col_others = np.delete(costs[:, col], row)
    row_second = float(np.min(row_others)) if row_others.size else np.inf
    col_second = float(np.min(col_others)) if col_others.size else np.inf
    row_margin = row_second - chosen
    col_margin = col_second - chosen
    return row_margin, col_margin, min(row_margin, col_margin)


def _trajectory_jump_ratio(track: _Trajectory) -> float:
    """Measure the largest step relative to the median step in a video track."""

    consecutive = track.valid[:-1] & track.valid[1:]
    if not consecutive.any():
        return np.nan
    steps = np.linalg.norm(np.diff(track.xy, axis=0)[consecutive], axis=1)
    typical_step = float(np.median(steps))
    if typical_step <= np.finfo(float).eps:
        return np.inf if np.max(steps) > 0 else np.nan
    return float(np.max(steps) / typical_step)


def _assign(
    video_tracks: list[_Trajectory],
    mocap_tracks: list[_Trajectory],
    costs: Array,
    *,
    max_shape_cost: float,
    min_margin: float,
    cost_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """Solve a partial assignment and reject costly or ambiguous matches."""

    columns = [
        "video_id", "mocap_id", "shape_cost", "n_common_samples",
        "common_fraction", "video_motion_scale", "mocap_motion_scale",
        "video_max_step_ratio",
        "row_margin", "column_margin", "assignment_margin", "accepted",
        "rejection_reason",
    ]
    n_video, n_mocap = costs.shape
    if n_mocap == 0:
        return pd.DataFrame(columns=columns)
    finite = np.isfinite(costs)
    finite_max = float(np.max(costs[finite])) if finite.any() else max_shape_cost
    large = max(1e6, finite_max * 1e6, max_shape_cost * 1e6)

    # Video rows are optional because the video population contains many
    # untagged animals.  Each MoCap column gets its own dummy row, so it can be
    # left unmatched without consuming a real video trajectory.
    augmented = np.full(
        (n_video + n_mocap, n_mocap), large, dtype=np.float32
    )
    if n_video:
        augmented[:n_video] = np.where(finite, costs, large)
    augmented[n_video:, :] = np.where(
        np.eye(n_mocap, dtype=bool), max_shape_cost, large
    )
    rows, cols = linear_sum_assignment(augmented)
    records = []
    for row, col in sorted(zip(rows, cols), key=lambda pair: pair[1]):
        if row >= n_video:
            records.append(
                {
                    "video_id": None,
                    "mocap_id": mocap_tracks[col].track_id,
                    "shape_cost": np.nan,
                    "n_common_samples": 0,
                    "common_fraction": 0.0,
                    "video_motion_scale": np.nan,
                    "mocap_motion_scale": np.nan,
                    "video_max_step_ratio": np.nan,
                    "row_margin": np.nan,
                    "column_margin": np.nan,
                    "assignment_margin": np.nan,
                    "accepted": False,
                    "rejection_reason": "no video assignment below unmatched cost",
                }
            )
            continue
        cost = float(costs[row, col])
        _, pair_n, pair_fraction, pair_video_scale, pair_mocap_scale = (
            _pair_shape_cost(
                video_tracks[row],
                mocap_tracks[col],
                **cost_kwargs,
            )
        )
        row_margin, column_margin, margin = _competitor_margin(costs, row, col)
        reasons = []
        if not np.isfinite(cost):
            reasons.append("insufficient overlap or motion")
        if cost > max_shape_cost:
            reasons.append("shape cost above threshold")
        if margin < min_margin:
            reasons.append("assignment is ambiguous")
        records.append(
            {
                "video_id": video_tracks[row].track_id,
                "mocap_id": mocap_tracks[col].track_id,
                "shape_cost": cost,
                "n_common_samples": pair_n,
                "common_fraction": pair_fraction,
                "video_motion_scale": pair_video_scale,
                "mocap_motion_scale": pair_mocap_scale,
                "video_max_step_ratio": _trajectory_jump_ratio(video_tracks[row]),
                "row_margin": row_margin,
                "column_margin": column_margin,
                "assignment_margin": margin,
                "accepted": len(reasons) == 0,
                "rejection_reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["accepted", "shape_cost"], ascending=[False, True], ignore_index=True
    )


def _paired_points(
    matches: pd.DataFrame,
    video_tracks: list[_Trajectory],
    mocap_tracks: list[_Trajectory],
    video_frames: Array,
    mocap_frames: Array,
    sample_times: Array,
    max_points_per_match: int,
) -> pd.DataFrame:
    """Collect synchronized point pairs from the accepted assignments."""

    if matches.empty or not matches["accepted"].any():
        return pd.DataFrame(
            columns=[
                "pair_index", "video_id", "mocap_id", "time",
                "video_frame", "mocap_frame",
                "mocap_x", "mocap_y", "video_x", "video_y",
            ]
        )
    video_by_id = {track.track_id: track for track in video_tracks}
    mocap_by_id = {track.track_id: track for track in mocap_tracks}
    pair_tables = []
    accepted = matches.loc[matches["accepted"]].reset_index(drop=True)
    for pair_index, match in accepted.iterrows():
        video = video_by_id[match["video_id"]]
        mocap = mocap_by_id[match["mocap_id"]]
        common = video.valid & mocap.valid
        indices = np.flatnonzero(common)

        # Evenly spaced samples preserve the full time range without creating
        # millions of Python dictionaries or overweighting long trajectories.
        if len(indices) > max_points_per_match:
            keep = np.linspace(
                0,
                len(indices) - 1,
                max_points_per_match,
                dtype=int,
            )
            indices = indices[keep]

        pair_tables.append(
            pd.DataFrame(
                {
                    "pair_index": pair_index,
                    "video_id": video.track_id,
                    "mocap_id": mocap.track_id,
                    "time": sample_times[indices],
                    "video_frame": video_frames[indices].astype(int),
                    "mocap_frame": mocap_frames[indices].astype(int),
                    "mocap_x": mocap.xy[indices, 0],
                    "mocap_y": mocap.xy[indices, 1],
                    "video_x": video.xy[indices, 0],
                    "video_y": video.xy[indices, 1],
                }
            )
        )
    return pd.concat(pair_tables, ignore_index=True)


def fit_spatial_transform(
    mocap_xy: Array,
    video_xy: Array,
    model: Literal["linear", "similarity", "affine"],
    *,
    allow_reflection: bool = True,
) -> SpatialTransform:
    """Fit a 2D transform from MoCap to video coordinates.

    ``linear`` has no offset, ``similarity`` has one isotropic scale plus an
    orthogonal matrix and offset, and ``affine`` has a free 2 by 2 matrix plus
    an offset.
    """

    x = np.asarray(mocap_xy, dtype=float)
    y = np.asarray(video_xy, dtype=float)
    if x.ndim != 2 or y.shape != x.shape or x.shape[1] != 2:
        raise ValueError("mocap_xy and video_xy must both have shape (n, 2)")
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y = x[finite], y[finite]
    if len(x) < 2:
        raise ValueError("At least two finite point pairs are required")

    if model == "linear":
        if np.linalg.matrix_rank(x) < 2:
            raise ValueError("Linear fit is rank deficient")
        row_matrix, *_ = np.linalg.lstsq(x, y, rcond=None)
        return SpatialTransform("linear", row_matrix.T, np.zeros(2))

    if model == "affine":
        design = np.column_stack((x, np.ones(len(x))))
        if np.linalg.matrix_rank(design) < 3:
            raise ValueError("Affine fit is rank deficient")
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        return SpatialTransform("affine", coefficients[:2].T, coefficients[2])

    if model == "similarity":
        x_mean = x.mean(axis=0)
        y_mean = y.mean(axis=0)
        xc = x - x_mean
        yc = y - y_mean
        denominator = float(np.sum(xc * xc))
        if denominator <= np.finfo(float).eps:
            raise ValueError("Similarity fit has no MoCap spatial variation")
        cross = xc.T @ yc
        u, singular_values, vt = np.linalg.svd(cross, full_matrices=False)
        correction = np.eye(2)
        if not allow_reflection and np.linalg.det(u @ vt) < 0:
            correction[-1, -1] = -1.0
        row_rotation = u @ correction @ vt
        scale = float(np.sum(singular_values * np.diag(correction)) / denominator)
        matrix = (scale * row_rotation).T
        offset = y_mean - x_mean @ matrix.T
        return SpatialTransform("similarity", matrix, offset, allow_reflection)

    raise ValueError(f"Unknown model: {model!r}")


def _errors(fit: SpatialTransform, x: Array, y: Array) -> tuple[float, float]:
    """Return RMSE and median Euclidean error for one spatial transform."""

    distance = np.linalg.norm(fit.apply(x) - y, axis=1)
    return float(np.sqrt(np.mean(distance**2))), float(np.median(distance))


def _video_spread(video_xy: Array) -> float:
    """Measure video spread for a scale-independent validation error."""

    center = np.nanmedian(video_xy, axis=0)
    return float(np.sqrt(np.nanmean(np.sum((video_xy - center) ** 2, axis=1))))


def _compare_models(
    points: pd.DataFrame,
    *,
    cv_folds: int,
    random_seed: int,
    allow_reflection: bool,
) -> tuple[pd.DataFrame, dict[str, SpatialTransform]]:
    """Fit spatial models and cross-validate by holding out whole matches."""

    columns = [
        "model", "status", "n_parameters", "n_points", "n_pairs",
        "train_rmse", "train_median_error", "cv_rmse", "cv_median_error",
        "cv_nrmse", "cv_group_rmse_mean", "cv_group_rmse_se", "cv_folds_used",
    ]
    if points.empty:
        return pd.DataFrame(columns=columns), {}
    x = points[["mocap_x", "mocap_y"]].to_numpy(dtype=float)
    y = points[["video_x", "video_y"]].to_numpy(dtype=float)
    groups = points["pair_index"].to_numpy()
    unique_groups = np.unique(groups)
    video_spread = _video_spread(y)
    parameter_count = {"linear": 4, "similarity": 4, "affine": 6}
    rng = np.random.default_rng(random_seed)
    shuffled_groups = rng.permutation(unique_groups)
    n_splits = min(max(int(cv_folds), 2), len(unique_groups))
    folds = np.array_split(shuffled_groups, n_splits) if n_splits >= 2 else []

    records = []
    transforms: dict[str, SpatialTransform] = {}
    for model in ("linear", "similarity", "affine"):
        try:
            fit = fit_spatial_transform(x, y, model, allow_reflection=allow_reflection)
            train_rmse, train_median = _errors(fit, x, y)
            transforms[model] = fit
            status = "ok"
        except ValueError as exc:
            records.append(
                {
                    "model": model,
                    "status": str(exc),
                    "n_parameters": parameter_count[model],
                    "n_points": len(points),
                    "n_pairs": len(unique_groups),
                    "train_rmse": np.nan,
                    "train_median_error": np.nan,
                    "cv_rmse": np.nan,
                    "cv_median_error": np.nan,
                    "cv_nrmse": np.nan,
                    "cv_group_rmse_mean": np.nan,
                    "cv_group_rmse_se": np.nan,
                    "cv_folds_used": 0,
                }
            )
            continue

        held_out_squared: list[Array] = []
        held_out_groups: list[Array] = []
        folds_used = 0
        for test_groups in folds:
            test = np.isin(groups, test_groups)
            train = ~test
            if not test.any() or not train.any():
                continue
            try:
                fold_fit = fit_spatial_transform(
                    x[train], y[train], model, allow_reflection=allow_reflection
                )
            except ValueError:
                continue
            residual = np.linalg.norm(fold_fit.apply(x[test]) - y[test], axis=1)
            held_out_squared.append(residual**2)
            held_out_groups.append(groups[test])
            folds_used += 1
        if held_out_squared:
            all_squared = np.concatenate(held_out_squared)
            all_groups = np.concatenate(held_out_groups)
            cv_rmse = float(np.sqrt(np.mean(all_squared)))
            cv_median = float(np.median(np.sqrt(all_squared)))
            cv_nrmse = cv_rmse / video_spread if video_spread > 0 else np.nan
            group_rmse = np.asarray(
                [
                    np.sqrt(np.mean(all_squared[all_groups == group]))
                    for group in np.unique(all_groups)
                ]
            )
            cv_group_mean = float(np.mean(group_rmse))
            cv_group_se = (
                float(np.std(group_rmse, ddof=1) / np.sqrt(len(group_rmse)))
                if len(group_rmse) > 1
                else np.nan
            )
        else:
            cv_rmse = cv_median = cv_nrmse = np.nan
            cv_group_mean = cv_group_se = np.nan
        records.append(
            {
                "model": model,
                "status": status,
                "n_parameters": parameter_count[model],
                "n_points": len(points),
                "n_pairs": len(unique_groups),
                "train_rmse": train_rmse,
                "train_median_error": train_median,
                "cv_rmse": cv_rmse,
                "cv_median_error": cv_median,
                "cv_nrmse": cv_nrmse,
                "cv_group_rmse_mean": cv_group_mean,
                "cv_group_rmse_se": cv_group_se,
                "cv_folds_used": folds_used,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns), transforms


def _add_global_transform_errors(
    matches: pd.DataFrame,
    points: pd.DataFrame,
    transforms: dict[str, SpatialTransform],
) -> pd.DataFrame:
    """Add per-match errors under each transform fitted to all accepted pairs."""

    matches = matches.copy()
    accepted_rows = matches.index[matches["accepted"]].to_numpy()
    x = points[["mocap_x", "mocap_y"]].to_numpy(dtype=float)
    y = points[["video_x", "video_y"]].to_numpy(dtype=float)

    for model, fit in transforms.items():
        rmse_col = f"global_{model}_rmse"
        max_col = f"global_{model}_max_error"
        matches[rmse_col] = np.nan
        matches[max_col] = np.nan

        residual = np.linalg.norm(fit.apply(x) - y, axis=1)
        for pair_index, match_row in enumerate(accepted_rows):
            pair_residual = residual[points["pair_index"].to_numpy() == pair_index]
            matches.loc[match_row, rmse_col] = np.sqrt(np.mean(pair_residual**2))
            matches.loc[match_row, max_col] = np.max(pair_residual)

    return matches


def _recommend_model(metrics: pd.DataFrame) -> str | None:
    """Choose the simplest model within one standard error of the best fit."""

    if metrics.empty:
        return None
    usable = metrics.loc[(metrics["status"] == "ok") & metrics["cv_rmse"].notna()].copy()
    if usable.empty:
        return None
    if "cv_group_rmse_mean" not in usable or usable["cv_group_rmse_mean"].isna().all():
        best = usable.sort_values(["cv_rmse", "n_parameters"]).iloc[0]
        return str(best["model"])

    # One-standard-error rule: prefer fewer parameters when its mean held-out
    # trajectory error is statistically indistinguishable from the best model.
    ranked = usable.dropna(subset=["cv_group_rmse_mean"]).copy()
    best = ranked.sort_values("cv_group_rmse_mean").iloc[0]
    best_se = float(best["cv_group_rmse_se"])
    tolerance = best_se if np.isfinite(best_se) else 0.0
    candidates = ranked.loc[
        ranked["cv_group_rmse_mean"] <= float(best["cv_group_rmse_mean"]) + tolerance
    ]
    chosen = candidates.sort_values(["n_parameters", "cv_group_rmse_mean"]).iloc[0]
    return str(chosen["model"])


def match_and_compare_transforms(
    video_ds: xr.Dataset,
    mocap_df: pd.DataFrame,
    *,
    video_folder: str | Path,
    mocap_timestamp_path: str | Path,
    mocap_csv_path: str | Path,
    mocap_fps: float = 25,
    analysis_start: datetime.datetime | None = None,
    analysis_end: datetime.datetime | None = None,
    sample_step: int = 1,
    min_common_samples: int = 7,
    min_common_fraction: float = 0.6,
    min_video_candidate_fraction: float = 0.6,
    pair_alignment_model: Literal["similarity", "affine"] = "similarity",
    shortlist_size: int = 100,
    shortlist_batch_size: int = 128,
    exhaustive_pair_limit: int = 500_000,
    max_assignment_memory_mb: float = 256,
    max_shape_cost: float = 0.75,
    min_assignment_margin: float = 0.05,
    allow_reflection: bool = True,
    trim_fraction: float = 0.1,
    coverage_penalty: float = 0.5,
    min_motion: float = 1e-8,
    max_points_per_match: int = 500,
    cv_folds: int = 5,
    random_seed: int = 0,
    video_candidate_ids: Iterable[Any] | None = None,
    mocap_candidate_ids: Iterable[Any] | None = None,
) -> RegistrationResult:
    """Match selected MoCap tracks to video tracks and compare spatial models.

    Parameters
    ----------
    video_ds
        Dataset with ``(id, frame)`` coordinates. ``frame`` contains absolute
        video frame numbers; positions are ``x_high_ord`` and ``y_high_ord``.
    mocap_df
        Table with exactly the relevant columns ``frame``, ``particle_id``,
        ``x``, and ``y``. MoCap frame numbers are one-based within the batch.
    video_folder, mocap_timestamp_path, mocap_csv_path
        Inputs expected by ``get_times_video`` and ``get_times_mocap``.
    analysis_start, analysis_end
        Optional datetime limits applied inside the automatically detected
        overlap. Leave both as ``None`` to use the complete overlap.
    sample_step
        Use every nth synchronized video sample. Increase this first if the
        synchronized position array is too large.
    mocap_candidate_ids
        IDs selected from sparse parts of the MoCap data. Restricting this list
        is strongly recommended because only tagged animals can have matches.
    shortlist_batch_size
        Number of video trajectories processed together during shortlisting.
        Smaller values reduce temporary memory without changing the result.
    max_points_per_match
        Maximum synchronized point pairs retained per accepted match for fitting
        and saving. Samples are spread evenly across the analysis interval.
    max_shape_cost, min_assignment_margin
        Screening thresholds applied after the one-to-one Hungarian assignment.
        Costs are dimensionless; margins compare an assignment with its nearest
        row and column competitors. ``pair_alignment_model='affine'`` makes the
        assignment insensitive to shear and anisotropic scale, but is easier to
        overfit and requires genuinely two-dimensional motion.

    Notes
    -----
    Pair scoring removes translation, isotropic scale, and an independently
    fitted rotation/reflection. This is useful before the spatial transform is
    known, but it also makes similar trajectories difficult to distinguish.
    Results should therefore be judged using assignment margins and held-out
    global-transform error, not the assigned IDs alone.
    """

    # STEP 1: Validate the small set of formats supported by this analysis
    _validate_video_dataset(video_ds)
    _require_columns(mocap_df, ("frame", "particle_id", "x", "y"))
    if mocap_fps <= 0:
        raise ValueError("mocap_fps must be positive")
    if sample_step < 1:
        raise ValueError("sample_step must be at least 1")
    if shortlist_batch_size < 1:
        raise ValueError("shortlist_batch_size must be at least 1")
    if max_assignment_memory_mb <= 0:
        raise ValueError("max_assignment_memory_mb must be positive")
    if max_points_per_match < 2:
        raise ValueError("max_points_per_match must be at least 2")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    if not 0 < min_common_fraction <= 1:
        raise ValueError("min_common_fraction must be in (0, 1]")
    if not 0 < min_video_candidate_fraction <= 1:
        raise ValueError("min_video_candidate_fraction must be in (0, 1]")
    if pair_alignment_model not in {"similarity", "affine"}:
        raise ValueError("pair_alignment_model must be 'similarity' or 'affine'")

    # STEP 2: Pair video frames with MoCap frames using their real timestamps
    ds_indices, video_frames, mocap_frames, sample_times = _synchronized_frames(
        video_ds,
        video_folder=video_folder,
        mocap_timestamp_path=mocap_timestamp_path,
        mocap_csv_path=mocap_csv_path,
        mocap_fps=mocap_fps,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        sample_step=sample_step,
    )

    # STEP 3: Extract only the synchronized samples for each trajectory
    n_video_tracks_in_dataset = video_ds.sizes["id"]
    all_video_tracks = _extract_video_trajectories(
        video_ds,
        ds_indices,
        video_candidate_ids,
    )
    video_tracks = _filter_video_candidates(
        all_video_tracks,
        min_samples=min_common_samples,
        min_fraction=min_video_candidate_fraction,
        min_motion=min_motion,
    )
    mocap_tracks = _extract_mocap_trajectories(
        mocap_df,
        mocap_frames,
        mocap_candidate_ids,
    )
    if not all_video_tracks:
        raise ValueError("No video trajectories were found")
    if not video_tracks:
        raise ValueError("No video trajectories passed the coverage and motion filters")
    if not mocap_tracks:
        raise ValueError("No MoCap trajectories were found")

    # Fail before allocating the dense assignment matrices. In practice this
    # usually means that mocap_candidate_ids was accidentally left unrestricted.
    assignment_mb = _estimated_assignment_mb(len(video_tracks), len(mocap_tracks))
    if assignment_mb > max_assignment_memory_mb:
        raise MemoryError(
            f"Assignment needs approximately {assignment_mb:.0f} MB for "
            f"{len(video_tracks)} video x {len(mocap_tracks)} MoCap tracks. "
            "Pass a smaller mocap_candidate_ids list, restrict video_candidate_ids, "
            "or deliberately increase max_assignment_memory_mb."
        )

    # STEP 4: Score candidate shapes, then solve a partial one-to-one assignment
    cost_kwargs = {
        "min_common_samples": min_common_samples,
        "min_common_fraction": min_common_fraction,
        "pair_alignment_model": pair_alignment_model,
        "allow_reflection": allow_reflection,
        "trim_fraction": trim_fraction,
        "coverage_penalty": coverage_penalty,
        "min_motion": min_motion,
    }
    costs, pair_cost_mode = _cost_matrix(
        video_tracks,
        mocap_tracks,
        shortlist_size=shortlist_size,
        shortlist_batch_size=shortlist_batch_size,
        exhaustive_pair_limit=exhaustive_pair_limit,
        **cost_kwargs,
    )
    matches = _assign(
        video_tracks,
        mocap_tracks,
        costs,
        max_shape_cost=max_shape_cost,
        min_margin=min_assignment_margin,
        cost_kwargs=cost_kwargs,
    )

    # STEP 5: Fit global spatial models using only confident assignments
    points = _paired_points(
        matches,
        video_tracks,
        mocap_tracks,
        video_frames,
        mocap_frames,
        sample_times,
        max_points_per_match,
    )
    metrics, transforms = _compare_models(
        points,
        cv_folds=cv_folds,
        random_seed=random_seed,
        allow_reflection=allow_reflection,
    )
    matches = _add_global_transform_errors(matches, points, transforms)

    config = {
        "mocap_fps": mocap_fps,
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "overlap_first_sample": sample_times[0],
        "overlap_last_sample": sample_times[-1],
        "sample_step": sample_step,
        "n_synchronized_samples": len(sample_times),
        "min_common_samples": min_common_samples,
        "min_common_fraction": min_common_fraction,
        "min_video_candidate_fraction": min_video_candidate_fraction,
        "pair_alignment_model": pair_alignment_model,
        "shortlist_size": shortlist_size,
        "shortlist_batch_size": shortlist_batch_size,
        "exhaustive_pair_limit": exhaustive_pair_limit,
        "estimated_assignment_memory_mb": assignment_mb,
        "max_assignment_memory_mb": max_assignment_memory_mb,
        "pair_cost_mode": pair_cost_mode,
        "n_video_tracks_in_dataset": n_video_tracks_in_dataset,
        "n_video_tracks_considered": len(all_video_tracks),
        "n_video_candidates": len(video_tracks),
        "n_mocap_tracks": len(mocap_tracks),
        "video_candidate_ids_supplied": video_candidate_ids is not None,
        "mocap_candidate_ids_supplied": mocap_candidate_ids is not None,
        "max_shape_cost": max_shape_cost,
        "min_assignment_margin": min_assignment_margin,
        "allow_reflection": allow_reflection,
        "trim_fraction": trim_fraction,
        "coverage_penalty": coverage_penalty,
        "max_points_per_match": max_points_per_match,
        "recommended_model": _recommend_model(metrics),
    }
    return RegistrationResult(matches, points, metrics, transforms, config)

def find_sparse_mocap_ids(
    mocap_df: pd.DataFrame,
    n_ids: int = 25,
    *,
    k: int = 5,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_step: int = 25,
    min_frame_fraction: float = 0.5,
) -> pd.DataFrame:
    """Find MoCap animals that are consistently far from tagged neighbours.

    Sparsity is measured as the distance to the kth-nearest tagged animal in
    each sampled frame. The final score is the 25th percentile of that distance
    across time, so a high score requires an animal to be isolated consistently
    rather than in only a few frames.

    Returns a table of the ``n_ids`` sparsest animals, ordered from most to least
    sparse. Pass the ``particle_id`` column to ``mocap_candidate_ids``.
    """

    # STEP 1: Validate the fixed MoCap format and selection parameters
    _require_columns(mocap_df, ("frame", "particle_id", "x", "y"))
    if n_ids < 1:
        raise ValueError("n_ids must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if not 0 < min_frame_fraction <= 1:
        raise ValueError("min_frame_fraction must be in (0, 1]")

    # STEP 2: Select an evenly spaced set of frames from the requested interval
    frames = np.sort(mocap_df["frame"].unique())
    if frame_start is not None:
        frames = frames[frames >= frame_start]
    if frame_end is not None:
        frames = frames[frames <= frame_end]
    frames = frames[::frame_step]
    if len(frames) == 0:
        raise ValueError("No MoCap frames fall inside the requested interval")

    work = mocap_df.loc[
        mocap_df["frame"].isin(frames),
        ["frame", "particle_id", "x", "y"],
    ].dropna(subset=["x", "y"])

    # STEP 3: Measure kth-neighbour distance separately in each sampled frame
    frame_scores = []
    for frame, group in work.groupby("frame", sort=False):
        if len(group) <= k:
            continue

        points = group[["x", "y"]].to_numpy(dtype=float)
        distances, _ = cKDTree(points).query(points, k=k + 1)
        frame_scores.append(
            pd.DataFrame(
                {
                    "particle_id": group["particle_id"].to_numpy(),
                    "frame": frame,
                    "kth_neighbor_distance": distances[:, k],
                }
            )
        )

    if not frame_scores:
        raise ValueError(f"No sampled frame contains more than {k} animals")

    # STEP 4: Keep animals observed often enough and rank persistent isolation
    scores = pd.concat(frame_scores, ignore_index=True)
    summary = (
        scores.groupby("particle_id", as_index=False)
        .agg(
            sparse_score=("kth_neighbor_distance", lambda x: x.quantile(0.25)),
            median_neighbor_distance=("kth_neighbor_distance", "median"),
            n_scored_frames=("frame", "nunique"),
        )
    )
    min_frames = int(math.ceil(len(frames) * min_frame_fraction))
    summary = summary.loc[summary["n_scored_frames"] >= min_frames]
    if summary.empty:
        raise ValueError(
            "No animal was present in enough sampled frames; reduce "
            "min_frame_fraction"
        )

    return summary.sort_values(
        ["sparse_score", "median_neighbor_distance"],
        ascending=False,
        ignore_index=True,
    ).head(n_ids)

def estimate_transform_from_matched_segments(
    matched_segments: pd.DataFrame,
    *,
    model: Literal["similarity", "affine"] = "similarity",
    allow_reflection: bool = True,
    max_points_per_segment: int = 200,
) -> SpatialTransform:
    """Estimate one spatial transform from accepted matched trajectory segments.

    ``matched_segments`` should normally be ``result.paired_points`` or the
    saved ``paired_points.csv``. Each segment contributes at most
    ``max_points_per_segment`` evenly spaced samples so long segments do not
    dominate the fit.
    """

    # STEP 1: Validate the accepted-segment table
    required = (
        "pair_index",
        "mocap_x",
        "mocap_y",
        "video_x",
        "video_y",
    )
    _require_columns(matched_segments, required)
    if matched_segments.empty:
        raise ValueError("matched_segments is empty")
    if max_points_per_segment < 2:
        raise ValueError("max_points_per_segment must be at least 2")

    # STEP 2: Give each accepted segment comparable weight in the transform
    balanced_segments = []
    for _, segment in matched_segments.groupby("pair_index", sort=False):
        finite = np.isfinite(
            segment[["mocap_x", "mocap_y", "video_x", "video_y"]]
        ).all(axis=1)
        segment = segment.loc[finite]
        if len(segment) > max_points_per_segment:
            keep = np.linspace(
                0,
                len(segment) - 1,
                max_points_per_segment,
                dtype=int,
            )
            segment = segment.iloc[keep]
        if len(segment):
            balanced_segments.append(segment)

    if not balanced_segments:
        raise ValueError("matched_segments contains no finite point pairs")
    points = pd.concat(balanced_segments, ignore_index=True)

    # STEP 3: Fit the requested MoCap-to-video transform
    return fit_spatial_transform(
        points[["mocap_x", "mocap_y"]].to_numpy(dtype=float),
        points[["video_x", "video_y"]].to_numpy(dtype=float),
        model,
        allow_reflection=allow_reflection,
    )

def check_match(mocap_df, vid_ds, mocap_id, vid_id, video_folder, mocap_timestamp_path, mocap_csv_path, plots_path:str, mocap_batch:int,mocap_fps=25, analysis_start=None, analysis_end=None, sample_step=1, transform_matrix=None, transform_offset=None):
    '''Plot a single match to visually check the quality of the registration.'''

    ds_indices, video_frames, mocap_frames, sample_times = _synchronized_frames(
    video_ds,
    video_folder=video_folder,
    mocap_timestamp_path=mocap_timestamp_path,
    mocap_csv_path=mocap_csv_path,
    mocap_fps=mocap_fps,
    analysis_start=analysis_start,
    analysis_end=analysis_end,
    sample_step=sample_step,
    )


    mocap_track = _extract_mocap_trajectories(mocap_df, mocap_frames, [mocap_id])[0]
    vid_track = _extract_video_trajectories(vid_ds, ds_indices, [vid_id])[0]

    # Check the rows for THIS particle, rather than the complete dataframe
    particle_rows = mocap_df.loc[
        mocap_df["particle_id"] == mocap_id,
        ["frame", "x", "y"],
    ].copy()

    finite = np.isfinite(
        particle_rows[["x", "y"]].to_numpy(dtype=float)
    ).all(axis=1)

    finite_particle_rows = particle_rows.loc[finite]

    matching_frames = np.intersect1d(
        mocap_frames,
        finite_particle_rows["frame"].to_numpy(),
)

    print("MoCap ID:", mocap_id)
    print("Number of rows for ID:", len(particle_rows))
    print("Finite rows for ID:", finite.sum())

    if len(particle_rows):
        print(
            "Frames available for ID:",
            particle_rows["frame"].min(),
            particle_rows["frame"].max(),
        )

    print("Requested frame range:", mocap_frames.min(), mocap_frames.max())
    print("Requested frames:", len(mocap_frames))
    print("Finite matching frames for ID:", len(matching_frames))
    print("First matching frames:", matching_frames[:10])

    # Apply the transform to the MoCap track if provided
    if transform_matrix is not None and transform_offset is not None:
        mocap_xy_transformed = (transform_matrix @ mocap_track.xy.T).T + transform_offset
        mocap_track = _Trajectory(mocap_track.track_id, mocap_xy_transformed, mocap_track.valid)

    print(np.nanmean(np.isnan(mocap_track.xy[:,0])))
    print(np.nanmean(np.isnan(mocap_track.xy[:,1])))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mocap_track.xy[:, 0], mocap_track.xy[:, 1], 'o-', label=f'MoCap {mocap_id}', alpha=0.7)
    ax.plot(vid_track.xy[:, 0], vid_track.xy[:, 1], 'o-', label=f'Video {vid_id}', alpha=0.7)
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.set_title(f'Match check: transformed MoCap {mocap_id} vs. video {vid_id}')
    ax.legend()
    plt.savefig(f'{plots_path}match_check_mocap_{mocap_id}_vid_{vid_id}_mocbatch_00{mocap_batch}.png')

def _spatial_assign(video_tracks, mocap_tracks, transform, max_distance_px,
                min_margin_px, min_common_samples, min_common_fraction):
    """Assign whole tracks under one shared transform; never align pairs separately."""

    # STEP 1: Score one pair at a time to bound temporary memory
    costs = np.full((len(video_tracks), len(mocap_tracks)), np.inf)
    for j, mocap in enumerate(mocap_tracks):
        predicted = transform.apply(mocap.xy)
        for i, video in enumerate(video_tracks):
            common = video.valid & mocap.valid
            if common.sum() < min_common_samples or common.mean() < min_common_fraction:
                continue
            errors = np.linalg.norm(predicted[common] - video.xy[common], axis=1)
            # Require 90% of shared positions to lie within the spatial gate.
            # This tolerates occasional outliers, but does not repair ID swaps.
            if np.quantile(errors, 0.9) <= max_distance_px:
                costs[i, j] = np.sqrt(np.mean(errors**2))

    # STEP 2: Reject ambiguity before assignment so rejected edges cannot block others
    eligible = np.zeros_like(costs, dtype=bool)
    margins = {}
    for i, j in zip(*np.where(np.isfinite(costs))):
        margin = _competitor_margin(costs, i, j)[2]
        if margin >= min_margin_px and costs[i, j] < max_distance_px:
            eligible[i, j] = True
            margins[i, j] = margin
    n_video, n_mocap = costs.shape
    augmented = np.full((n_video + n_mocap, n_mocap), max_distance_px * 1e6)
    augmented[:n_video] = np.where(eligible, costs, max_distance_px * 1e6)
    for j in range(n_mocap):
        augmented[n_video + j, j] = max_distance_px
    rows, cols = linear_sum_assignment(augmented)
    records = []
    for i, j in zip(rows, cols):
        accepted = i < n_video
        records.append(dict(
            video_id=video_tracks[i].track_id if accepted else None,
            mocap_id=mocap_tracks[j].track_id,
            accepted=accepted,
            spatial_rmse=costs[i, j] if accepted else np.nan,
            spatial_margin=margins[i, j] if accepted else np.nan,
            rejection_reason="" if accepted else "coverage, distance or ambiguity",
        ))
    return pd.DataFrame(records, columns=["video_id", "mocap_id", "accepted",
                                         "spatial_rmse", "spatial_margin", "rejection_reason"])

def iterative_spatial_registration(
    video_ds: xr.Dataset, mocap_df: pd.DataFrame, seed_points: pd.DataFrame, *,
    video_folder: str | Path, mocap_timestamp_path: str | Path,
    mocap_csv_path: str | Path, mocap_candidate_ids: Iterable[Any],
    mocap_fps: float = 25, analysis_start=None, analysis_end=None,
    sample_step: int = 5, max_distance_px: float = 70,
    min_margin_px: float = 10, seed_max_deterioration_px: float = 2,
    min_common_samples: int = 10, min_common_fraction: float = 0.8,
    max_iterations: int = 5, max_points_per_match: int = 200,
) -> RegistrationResult:
    """Refine a similarity transform from manually verified paired_points rows.

    Seed identities are reserved and ONLY their supplied rows are trusted.
    Other tracks are reassigned on each iteration under the shared transform.
    max_distance_px gates the 90th percentile of positional error; min_margin_px
    separates a match from competing eligible tracks. Neither is a probability.
    Seed RMSE and 90th-percentile error may increase by at most
    seed_max_deterioration_px relative to the ORIGINAL seed fit, per seed pair.
    This prevents cumulative drift. Rejected updates leave the transform intact.

    No track splitting is performed: use short intervals if identities switch.
    Iteration diagnostics are in config['iteration_history']; model_metrics is
    empty because this refinement does not perform independent validation.
    """
    # STEP 1: Validate and fit only the manually verified segments
    if (max_distance_px <= 0 or min_margin_px <= 0 or seed_max_deterioration_px < 0
            or sample_step < 1 or max_iterations < 1 or max_points_per_match < 2
            or min_common_samples < 2 or not 0 < min_common_fraction <= 1):
        raise ValueError("Invalid spatial registration thresholds or sample counts")
    _validate_video_dataset(video_ds)
    columns = ["mocap_x", "mocap_y", "video_x", "video_y"]
    seed = seed_points.copy()
    if seed.empty or not np.isfinite(seed[columns].to_numpy()).all():
        raise ValueError("Supply nonempty, finite, manually verified seed points")
    identities = seed[["video_id", "mocap_id"]].drop_duplicates()
    if identities.video_id.duplicated().any() or identities.mocap_id.duplicated().any():
        raise ValueError("Seed identities must be one-to-one")
    transform = estimate_transform_from_matched_segments(seed)
    def seed_errors(fit):
        values = []
        for _, group in seed.groupby("pair_index", sort=False):
            errors = np.linalg.norm(fit.apply(group[["mocap_x", "mocap_y"]].to_numpy())
                                    - group[["video_x", "video_y"]].to_numpy(), axis=1)
            values.append([np.sqrt(np.mean(errors**2)), np.quantile(errors, .9)])
        return np.asarray(values)
    baseline_errors = seed_errors(transform)

    # STEP 2: Synchronize with the existing timestamp helpers
    indices, vf, mf, times = _synchronized_frames(
        video_ds, video_folder=video_folder, mocap_timestamp_path=mocap_timestamp_path,
        mocap_csv_path=mocap_csv_path, mocap_fps=mocap_fps,
        analysis_start=analysis_start, analysis_end=analysis_end, sample_step=sample_step)
    video = _extract_video_trajectories(video_ds, indices,
        [i for i in video_ds.id.values if i not in set(identities.video_id)])
    mocap = _extract_mocap_trajectories(mocap_df, mf,
        [i for i in mocap_candidate_ids if i not in set(identities.mocap_id)])
    if 2 * _estimated_assignment_mb(len(video), len(mocap)) > 256:
        raise MemoryError("Reduce mocap_candidate_ids: dense assignment exceeds 256 MB budget")
    def assign(fit):
        return _spatial_assign(video, mocap, fit, max_distance_px, min_margin_px,
                               min_common_samples, min_common_fraction)
    def collect(matches):
        new = _paired_points(matches, video, mocap, vf, mf, times, max_points_per_match)
        # Avoid pair-index collisions with manually selected seed rows
        new["pair_index"] += int(seed.pair_index.max()) + 1
        return pd.concat([seed, new], ignore_index=True)

    # STEP 3: Propose updates, always testing against the original seed alignment
    history = []
    for iteration in range(max_iterations):
        matches = assign(transform)
        if matches.empty or not matches.accepted.any():
            break
        points = collect(matches)
        proposed = estimate_transform_from_matched_segments(points,
            max_points_per_segment=max_points_per_match)
        safe = bool(np.all(seed_errors(proposed) <= baseline_errors + seed_max_deterioration_px))
        x = points[["mocap_x", "mocap_y"]].to_numpy()
        y = points[["video_x", "video_y"]].to_numpy()
        old_rmse = _errors(transform, x, y)[0]
        new_rmse = _errors(proposed, x, y)[0]
        accepted = safe and new_rmse <= old_rmse
        history.append(dict(iteration=iteration + 1, n_new_matches=int(matches.accepted.sum()),
                            before_rmse=old_rmse, proposed_rmse=new_rmse,
                            seed_safe=safe, update_accepted=accepted))
        if not accepted:
            break
        change = np.max(np.linalg.norm(proposed.apply(x) - transform.apply(x), axis=1))
        transform = proposed
        if change < 0.1:
            break

    # STEP 4: Reassign under the FINAL transform; never return stale assignments
    matches = assign(transform)
    points = collect(matches)
    matches["is_seed"] = False
    trusted = identities.assign(accepted=True, is_seed=True, rejection_reason="verified seed")
    matches = pd.concat([trusted, matches], ignore_index=True)
    config = dict(method="seeded_spatial_iteration", recommended_model="similarity",
                  max_distance_px=max_distance_px, min_margin_px=min_margin_px,
                  seed_max_deterioration_px=seed_max_deterioration_px,
                  sample_step=sample_step, n_synchronized_samples=len(times),
                  iteration_history=history)
    # Leave model_metrics empty: training on the seed is not independent validation
    return RegistrationResult(matches, points, pd.DataFrame(),
                              {"similarity": transform}, config)

def dewarp_high_ord(
    ds: xr.Dataset,
    calibration_path: str,
    frames_per_chunk: int = 100,
) -> xr.Dataset:
    """Dewarp x_high_ord and y_high_ord while preserving their original shape."""

    # STEP 1: Create output arrays with NaNs in the original positions
    shape = (ds.sizes["id"], ds.sizes["frame"])
    x_dewarped = np.full(shape, np.nan, dtype=np.float32)
    y_dewarped = np.full(shape, np.nan, dtype=np.float32)

    # STEP 2: Dewarp finite points in manageable frame chunks
    for start in tqdm(range(0, ds.sizes["frame"], frames_per_chunk), 'Dewarping frames', unit='chunk'):
        stop = min(start + frames_per_chunk, ds.sizes["frame"])

        x = ds["x_high_ord"].isel(frame=slice(start, stop)).values
        y = ds["y_high_ord"].isel(frame=slice(start, stop)).values

        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue

        points = np.column_stack((x[valid], y[valid]))
        dewarped = np.asarray(
            dewarp_pts(points, calibration_path),
            dtype=np.float32,
        )

        if dewarped.shape != points.shape:
            raise ValueError(
                f"dewarp_pts returned shape {dewarped.shape}; "
                f"expected {points.shape}"
            )

        x_chunk = x_dewarped[:, start:stop]
        y_chunk = y_dewarped[:, start:stop]
        x_chunk[valid] = dewarped[:, 0]
        y_chunk[valid] = dewarped[:, 1]

    # STEP 3: Add the dewarped positions without modifying the original columns
    ds = ds.copy(deep=False)
    ds["x_high_ord"] = (("id", "frame"), x_dewarped)
    ds["y_high_ord"] = (("id", "frame"), y_dewarped)

    return ds

def find_informative_mocap_ids(
    mocap_df: pd.DataFrame, n_ids: int = 25, *,
    min_step_distance: float, max_speed: float,
    frame_start: int, frame_end: int, mocap_fps: float = 25,
    frame_step: int = 5, k: int = 5, min_frame_fraction: float = 0.8,
    min_moving_steps: int = 10, min_turns: int = 3,
    turn_angle_deg: float = 30, min_axis_ratio: float = 0.1,
) -> pd.DataFrame:
    """Rank sparse, moving, turning MoCap tracks within ONE analysis interval.

    min_step_distance is displacement in native MoCap units per sampled step;
    max_speed is in native MoCap units/second. Set these from localization noise
    and plausible motion, not from the video pixel tolerance. No units are guessed.
    Frame bounds are inclusive batch-relative CSV frame numbers (one-based).
    Run on short intervals, then match using the SAME interval. Returned rows
    include all scoring components; particle_id supplies mocap_candidate_ids.

    This ranks candidate IDs, not confirmed identities. It does not remove video
    identity swaps or ensure isolation among untagged animals.
    """
    # STEP 1: Validate explicit sampling and motion thresholds
    _require_columns(mocap_df, ("frame", "particle_id", "x", "y"))
    if (n_ids < 1 or frame_start > frame_end or frame_step < 1 or k < 1
            or min_step_distance <= 0 or max_speed <= 0 or mocap_fps <= 0
            or not 0 < min_frame_fraction <= 1 or min_moving_steps < 2
            or min_turns < 1 or not 0 < turn_angle_deg < 180
            or not 0 <= min_axis_ratio <= 1):
        raise ValueError("Invalid interval, sampling or motion thresholds")
    frames = np.arange(frame_start, frame_end + 1, frame_step)
    work = mocap_df.loc[mocap_df.frame.isin(frames),
                        ["frame", "particle_id", "x", "y"]].copy()
    if work.duplicated(["frame", "particle_id"]).any():
        raise ValueError("Duplicate particle_id/frame rows in selection interval")
    work = work.loc[np.isfinite(work[["x", "y"]]).all(axis=1)].copy()
    work["neighbor_distance"] = np.nan

    # STEP 2: Use every visible tagged animal when measuring local crowding
    for _, group in work.groupby("frame", sort=False):
        if len(group) > k:
            distances, _ = cKDTree(group[["x", "y"]].to_numpy()).query(
                group[["x", "y"]].to_numpy(), k=[k + 1])
            work.loc[group.index, "neighbor_distance"] = distances[:, 0]

    # STEP 3: Reject insufficient coverage, implausible steps and weak geometry
    records = []
    for pid, group in work.groupby("particle_id", sort=False):
        group = group.sort_values("frame")
        coverage = len(group) / len(frames)
        if coverage < min_frame_fraction:
            continue
        xy = group[["x", "y"]].to_numpy()
        df = np.diff(group.frame.to_numpy())
        delta = np.diff(xy, axis=0)
        distance = np.linalg.norm(delta, axis=1)
        consecutive = df == frame_step
        speed = distance / (df / mocap_fps)
        # Never count turns across missing samples or accept an obvious jump.
        if np.any(speed[consecutive] > max_speed):
            continue
        moving = consecutive & (distance >= min_step_distance)
        if moving.sum() < min_moving_steps:
            continue
        singular = np.linalg.svd(xy - xy.mean(axis=0), compute_uv=False)
        axis_ratio = singular[1] / singular[0] if singular[0] > 0 else 0
        if axis_ratio < min_axis_ratio:
            continue

        # STEP 4: Count turning episodes, not every sample of one long bend
        usable = moving[:-1] & moving[1:]
        angles = np.zeros(len(usable))
        dots = np.sum(delta[:-1] * delta[1:], axis=1)
        denom = distance[:-1] * distance[1:]
        angles[usable] = np.degrees(np.arccos(np.clip(
            dots[usable] / denom[usable], -1, 1)))
        turns = usable & (angles >= turn_angle_deg)
        n_turns = int(np.sum(turns & ~np.r_[False, turns[:-1]]))
        if n_turns < min_turns:
            continue
        neighbors = group.neighbor_distance.dropna()
        if len(neighbors) / len(frames) < min_frame_fraction:
            continue
        records.append(dict(particle_id=pid, sparse_score=neighbors.quantile(.25),
            axis_ratio=axis_ratio, n_turns=n_turns, coverage=coverage,
            moving_fraction=float(moving.sum() / max(1, len(frames)-1)),
            path_length=float(distance[consecutive].sum()),
            frame_start=frame_start, frame_end=frame_end))

    # STEP 5: Combine ranks so large numerical distance units cannot dominate
    columns = ["particle_id", "sparse_score", "axis_ratio", "n_turns", "coverage",
               "moving_fraction", "path_length", "frame_start", "frame_end"]
    scores = pd.DataFrame(records, columns=columns)
    components = ["sparse_score", "axis_ratio", "n_turns", "moving_fraction"]
    scores["candidate_score"] = scores[components].rank(pct=True).mean(axis=1)
    return scores.sort_values("candidate_score", ascending=False,
                              kind="stable", ignore_index=True).head(n_ids)


if __name__ == "__main__":
    batch_num = 0
    ds_load_name = f'/output/{20230329}/bb_plots/preprocessed_h5s/batch_{batch_num}/traj_data.h5'
    calibration_path = '/intrinsics/arena_board_calibration/calibration_official.yaml'
    video_ds = dewarp_high_ord(load_preprocessed_data(ds_load_name), calibration_path=calibration_path, frames_per_chunk = 100)

    plots_path = '/output/20230329/kp_plots/calibration/'
    path_to_vid_dir = '/original/20230329/video/'
    # h5_prep = f'/keypoints/dewarped/20230329_preprocessed_complete_dewarped_batch_0_5.0Hz.hdf5'
    mocap_batch = 49
    path_to_mocap_batch = f'/mocap/20230329/csvs/10K_Marching_00{mocap_batch}.csv'
    path_to_mocap_ts = '/mocap/20230329/qtm_capture_times.csv'
    mocap_df = pd.read_csv(path_to_mocap_batch)

    # candidates = find_sparse_mocap_ids(
    # mocap_df,
    # n_ids=25,
    # k=5,
    # frame_start=None,
    # frame_end=None,
    # frame_step=25,
    # min_frame_fraction=0.5)

    motion_threshold = 1 # mm
    speed_limit = 150 # Three whole body lengths in one frame is not likely (mm/s)
    candidates = find_informative_mocap_ids(
    mocap_df,
    n_ids=25,
    frame_start=1,
    frame_end=len(np.unique(mocap_df['frame'])),
    min_step_distance=motion_threshold,
    max_speed=speed_limit,
    frame_step=5,
    k=5,
    )

    print(candidates)
    sparse_mocap_ids = candidates["particle_id"].to_numpy()

    result = match_and_compare_transforms(
    video_ds,
    mocap_df,
    video_folder=path_to_vid_dir,
    mocap_timestamp_path=path_to_mocap_ts,
    mocap_csv_path=path_to_mocap_batch,
    mocap_candidate_ids=sparse_mocap_ids,
    sample_step=5,
    shortlist_size=30,
    shortlist_batch_size=64,
    max_points_per_match=200,
    mocap_fps = 25)

    result.save(f"trajectory_registration_output_dewarped_mocap_batch_00{mocap_batch}_informative")

    # paired_points = pd.read_csv(f"trajectory_registration_output_dewarped_mocap_batch_00{mocap_batch}/paired_points.csv")
    # transform_json = f'trajectory_registration_output_dewarped_mocap_batch_00{mocap_batch}/transforms.json'
    # with open(transform_json, 'r') as f:
    #     transform_data = json.load(f)
    # matrix = transform_data['similarity']['matrix']
    # offset = transform_data['similarity']['offset']

    # check_match(mocap_df, video_ds, mocap_id=12, vid_id=7191, video_folder=path_to_vid_dir, mocap_timestamp_path=path_to_mocap_ts, mocap_csv_path=path_to_mocap_batch, plots_path=plots_path, mocap_batch = mocap_batch,mocap_fps=25, analysis_start=None, analysis_end=None, sample_step=1, transform_matrix = matrix, transform_offset = offset)
    
    # seed = paired_points.loc[
    # (paired_points["mocap_id"] == 52875) &
    # (paired_points["video_id"] == 6516)
    # ].copy()

    # result = iterative_spatial_registration(
    #     video_ds,
    #     mocap_df,
    #     seed,
    #     video_folder=path_to_vid_dir,
    #     mocap_timestamp_path=path_to_mocap_ts,
    #     mocap_csv_path=path_to_mocap_batch,
    #     mocap_candidate_ids=sparse_mocap_ids,
    #     max_distance_px=70,             # Approximately 5 cm (70 px)
    #     min_margin_px=10,               # Separation from competing matches
    #     seed_max_deterioration_px=2,    # Protect the verified alignment
    # )

    # # # transform = result.transforms["similarity"]
    # result.save("iterative_registration_output_dewarped")