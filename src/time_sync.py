'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
from pathlib import Path
import xarray as xr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

from space_sync import find_mocap_to_video_matrices, apply_transform
from data_handling import load_preprocessed_data

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def get_times_video(video_folder: str, abs_start_frame: int = 0, abs_end_frame: int = None):
    ''' Retrieve numpy array of timestamps for a slice of frames from a video folder.'''

    # STEP 0: Validate inputs
    assert abs_start_frame >= 0, "abs_start_frame must be non-negative."
    assert abs_end_frame is None or abs_end_frame > abs_start_frame, "abs_end_frame must be greater than abs_start_frame."

    # STEP 1: Order file name in the video_folder and get the splice specified by the absolute start and end frame indices
    video_folder = Path(video_folder)
    frame_titles = sorted(list(video_folder.glob('*.jpg')))[abs_start_frame:abs_end_frame]

    # STEP 2: Extract timestamps from the file names and convert to datetime objects
    timestamps = []
    for frame_title in frame_titles:
        # Extract the timestamp from the file name (assuming the format is consistent)
        timestamp_str = '_'.join(frame_title.stem.split('_')[-2:])  # Get the last part of the stem

        # Create a datetime object from the timestamp string
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S-%f")  # Convert to datetime object
        timestamps.append(timestamp)

    return np.array(timestamps)

def get_times_mocap(ts_path:str, csv_path:str, rel_start_frame:int = 0, rel_end_frame:int = None, mocap_fps:int = 25):
    ''' Retrieve numpy array of timestamps for a slice of frames within a single MoCap batch.'''

    # STEP 0: Validate inputs
    assert rel_start_frame >= 0, "rel_start_frame must be non-negative."
    assert rel_end_frame is None or rel_end_frame > rel_start_frame, "rel_end_frame must be greater than rel_start_frame."

    # STEP 1: Find the row containing information for the specified batch_name in the ts_path CSV file
    ts_df = pd.read_csv(ts_path)
    batch_name = csv_path.split('/')[-1].split('.')[0] + '.qtm' # Extract batch name from the csv_path
    batch_row = ts_df[ts_df['relative_path'] == batch_name]

    if not len(batch_row):
        raise ValueError(f"Batch name '{batch_name}' not found in the timestamp CSV file.")

    # STEP 2: Extract the initial timestamp from the batch_row
    timestamp = datetime.datetime.strptime(batch_row['capture_start_local'].values[0], "%Y-%m-%dT%H:%M:%S.%f")  # Convert to datetime object

    # STEP 3: Determine the length of the batch to assure the requested slice is within bounds
    with open(csv_path, mode = 'r') as mocap:
        mocap_df = pd.read_csv(mocap)
        mocap_frame_idcs = np.unique(mocap_df['frame'])
        mocap_length = len(mocap_frame_idcs)

    assert rel_end_frame is None or rel_end_frame <= mocap_length, f"rel_end_frame must be less than or equal to the length of the MoCap batch ({mocap_length})."
    assert rel_start_frame < mocap_length, f"rel_start_frame must be less than the length of the MoCap batch ({mocap_length})."
    assert mocap_frame_idcs[0] == 1, "MoCap batch does not start with frame index 1, contrary to what is assumed."
    assert not np.sum(np.diff(mocap_frame_idcs) != 1), "MoCap frames are not strictly consecutive, contrary to what is assumed."

    # STEP 4: Generate timestamps for the specified slice of frames
    if rel_end_frame is None:
        rel_end_frame = mocap_length

    timestamps = [timestamp + datetime.timedelta(seconds=i/mocap_fps) for i in range(rel_start_frame, rel_end_frame)]

    return np.array(timestamps)

def _datetime_to_seconds(times):
    """Convert datetime-like sequence to floating-point Unix seconds."""
    return np.asarray([t.timestamp() for t in times], dtype=float)

def downsample_mocap_for_video(vid_times:np.ndarray, mocap_times:np.ndarray, mocap_fps:int = 25):
    ''' Finds indices of MoCap timestamps that are closest to each video timestamp. Returns a list of indices of MoCap timestamps for downsampling the MoCap data.'''

    # STEP 0: Validate inputs
    assert len(vid_times) > 0, "vid_times must not be empty."
    assert len(mocap_times) > 0, "mocap_times must not be empty."

    # STEP 1: Convert datetime arrays to numpy arrays of seconds since epoch for easier comparison
    vid_seconds = _datetime_to_seconds(vid_times)
    mocap_seconds = _datetime_to_seconds(mocap_times)

    # STEP 2: Find the closest MoCap timestamp for each video timestamp
    matched_indices = []
    dists = []
    for vid_time in vid_seconds:
        closest_index = np.argmin(np.abs(mocap_seconds - vid_time))
        matched_indices.append(closest_index)
        dists.append(np.abs(mocap_seconds[closest_index] - vid_time))

    # Check that MoCap timestamps are within 1/mocap_fps seconds of the video timestamps
    matched_indices = np.array(matched_indices).astype(int)
    valid_mask = np.array(dists) <= 0.5/mocap_fps  # Create a boolean mask for valid matches

    return matched_indices, valid_mask

def get_temporal_overlap(vid_times:np.ndarray, mocap_times:np.ndarray):
    ''' Return the real local time range of overlap in video and MoCap data.'''

    # Get start and end times of both datasets
    start_video, end_video = vid_times[0], vid_times[-1]
    start_mocap, end_mocap = mocap_times[0], mocap_times[-1]

    # Determine the later start and the earlier end
    if type(start_video) == datetime.datetime:
        start = [start_video, start_mocap][int(start_video.timestamp() - start_mocap.timestamp() < 0)]
        end = [end_video, end_mocap][int(end_video.timestamp() - end_mocap.timestamp() > 0)]
    elif type(start_video) == float or type(start_video) == np.float64:
        start = max(start_video, start_mocap)
        end = min(end_video, end_mocap)
    else:
        raise TypeError(f"Inappropriate data type (must be datetime.datetime or float): {type(start_video)}.")

    return start, end

def _robust_zscore(x):
    """
    Robustly standardize a score across candidate offsets.

    Returns approximately comparable values even when the underlying
    metrics have very different units.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)

    valid = np.isfinite(x)
    if valid.sum() < 2:
        return out

    values = x[valid]

    med = np.median(values)
    mad = np.median(np.abs(values - med))

    if mad == 0:
        std = np.std(values)

        if std == 0:
            out[valid] = 0
            return out

        out[valid] = (values - med) / std

    else:
        # Make comparable to Gaussian std
        out[valid] = (values - med) / (1.4826 * mad)

    return out

def finite_zscore(x):
    ''' A normal z-score function.'''
    x = np.asarray(x, dtype=float)

    mean = np.nanmean(x)
    std = np.nanstd(x)

    if not np.isfinite(std) or std == 0:
        return np.full_like(x, np.nan)

    return (x - mean) / std

def get_video_points(ds:xr.Dataset, abs_video_frame:int):
    """Return finite video detections for one absolute video frame."""

    frame_matches = np.where(ds.frame.values == abs_video_frame)[0]

    if len(frame_matches) != 1:
        return np.empty((0, 2), dtype=float)

    i = frame_matches[0]

    points = np.column_stack([ds.centroid_x.values[i], ds.centroid_y.values[i]])

    return points[np.isfinite(points).all(axis=1)]

def get_mocap_points(mocap_df:pd.DataFrame, mocap_frame_idx:int):
    """
    mocap_frame_idx is zero-based.
    CSV frame numbers are one-based.
    """

    frame_number = int(mocap_frame_idx) + 1

    points = mocap_df.loc[mocap_df["frame"] == frame_number, ["x", "y"]].to_numpy(dtype=float)

    return points[np.isfinite(points).all(axis=1)]

def pointset_alignment_error(video_points:np.ndarray, mocap_points:np.ndarray, R:np.ndarray, s:np.ndarray, t:np.ndarray, max_match_distance=None, close_distance_m:float = 0.02):
    """
    Robust one-way point-set distance.

    Every transformed MoCap point is matched to its nearest video detection.
    This is deliberately asymmetric because video contains many animals that
    are not tagged in MoCap.

    Lower is better.
    """

    if len(video_points) == 0 or len(mocap_points) == 0:
        return np.nan

    # Transform MoCap points from original coordinates into world coordinates
    mocap_world = apply_transform(mocap_points, R, s, t)

    # Compute the distance between each MoCap point and their nearest video point
    tree = cKDTree(video_points)
    distances, video_idx = tree.query(mocap_world, k=1)
    distances = distances[np.isfinite(distances)]

    # Check whether those matches are mutual: matched video point -> nearest MoCap point
    mocap_tree = cKDTree(mocap_world)
    _, mocap_idx_back = mocap_tree.query(video_points[video_idx], k=1)
    mutual = (mocap_idx_back == np.arange(len(mocap_world)))
    mutual = mutual[np.isfinite(distances)]

    # Only include distances up to some value
    if max_match_distance is not None:
        distances = distances[distances <= max_match_distance]

    # Compute median distance
    med_dist = float(np.median(distances))

    # Compute qth quantile
    q10 = float(np.quantile(distances, 0.1))
    q25 = float(np.quantile(distances, 0.25))

    # Compute fraction of close matches
    close = float(np.mean(distances <= close_distance_m))

    # Compute mutual close fraction
    mutual_close = float(np.mean(mutual & (distances < close_distance_m)))

    return med_dist, q10, q25, close, mutual_close

def knn_density_quantiles(pts:np.ndarray, k:int=5, quantiles:list=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    pts: shape (n_detections, 2)

    Returns:
        shape (n_quantiles,)
    """

    # Only keep points where x and y are valid
    pts = pts[np.isfinite(pts).all(axis=1), :]

    if len(pts) <= k:
        print(f'Insufficient number of points ({len(pts)}) for {k}-NN density estimates.')
        return np.full(len(quantiles), np.nan)

    # Find the distance to the kth neighbour
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    r_k = dists[:, k]

    valid = np.isfinite(r_k) & (r_k > 0)

    if not valid.any():
        print(f'No valid {k}-NN distance found.')
        return np.full(len(quantiles), np.nan)

    # Compute the density using the kth distance
    density = k / (np.pi * r_k[valid] ** 2)

    # Multiplicative density differences become additive.
    log_density = np.log(density)

    # Take the quantiles of the log-densities in this frame
    return np.quantile(log_density, quantiles)

def score_offset(offset_s:float, anchor_video_frames:np.ndarray, vid_times:np.ndarray[datetime.datetime], mocap_times:np.ndarray[datetime.datetime], 
                    vid_ds:xr.Dataset, mocap_df:pd.DataFrame, R:np.ndarray, s:np.ndarray, t:np.ndarray, vid_blinks:np.ndarray[bool], mocap_blinks:np.ndarray[bool], mocap_fps:float=25, k:int = 5, quantiles:np.ndarray = np.array([0.1, 0.3, 0.5, 0.7, 0.9])):
    """
    Use spatial scores, density dynamics, and IR light blinking to score a single offset."""

    # Add offset to MoCap times
    mocap_times = [m_time + datetime.timedelta(seconds = offset_s) for m_time in mocap_times]

    # Assign closest MoCap frame (WITH offset) to anchor video frames
    mocap_idcs, valid = downsample_mocap_for_video(vid_times[anchor_video_frames], mocap_times, mocap_fps)

    # Compute spatial score, log densities, and blink scores for each anchor video frame
    spatial_distribution = np.full((5, len(anchor_video_frames)), np.nan) # median distance/q10/q25/close fraction/mutual close fraction, frame
    log_den_video = np.full((len(quantiles), len(anchor_video_frames)), np.nan) # quantile, frame
    log_den_mocap = np.full((len(quantiles), len(anchor_video_frames)), np.nan) # quantile, frame

    for i, video_frame in enumerate(anchor_video_frames):

        # Check that an appropriate corresponding MoCap frame was selected
        if not valid[i]:
            continue

        # Get frame-specific video and MoCap points
        video_points = get_video_points(vid_ds, video_frame)
        mocap_points = get_mocap_points(mocap_df, mocap_idcs[i])

        # Compute error matching MoCap points to nearest video points
        spatial_distribution[:,i] = pointset_alignment_error(video_points, mocap_points, R, s, t)

        # Compute knn-density and take logarithm
        log_den_video[:,i] = knn_density_quantiles(video_points, k, quantiles)
        log_den_mocap[:,i] = knn_density_quantiles(mocap_points, k, quantiles)

    # Take median of spatial distribution metrics across different frames to make score more robust
    spatial_metrics = np.nanmedian(spatial_distribution, axis = 1)
    
    # Finalize density scores
    if len(quantiles) > 1:
        quant_corr = np.full(len(quantiles) + 1, np.nan)
    else:
        quant_corr = np.full(len(quantiles), np.nan)

    for q in range(len(quantiles)):

        # Take z-score of log-density quantiles to make video and MoCap data comparable
        x = finite_zscore(log_den_video[q,:])
        y = finite_zscore(log_den_mocap[q,:])

        # Get Pearson-product moment correlation between video and MoCap at this quantile
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        quant_corr[q] = np.corrcoef(x, y)[0, 1]

    # Include 'spread' term for density
    if len(quantiles) > 1:

        # Take z-scores
        x = finite_zscore(log_den_video[-1,:] - log_den_video[0,:])
        y = finite_zscore(log_den_mocap[-1,:] - log_den_mocap[0,:])

        # Get correlation coefficient
        if not np.std(x) == 0 and not np.std(y) == 0:
            quant_corr[-1] = np.corrcoef(x, y)[0, 1]
    
    # Take mean across quantiles for final density score
    med_density_score = np.nanmedian(quant_corr)

    # # Compute blinking alignment
    # mocap_idcs, valid = downsample_mocap_for_video(vid_times, mocap_times, mocap_fps)
    # blink_score = np.mean((vid_blinks[valid]*mocap_blinks[mocap_idcs[valid]]).astype(int)) # Fraction of blink states that are the same in the video and MoCap
    # blink_score = np.nan

    # return med_frame_error, med_density_score, blink_score
    return spatial_metrics, quant_corr

def estimate_timestamp_correction(vid_ds:xr.Dataset, mocap_df:pd.DataFrame, vid_times:np.ndarray[datetime.datetime], mocap_times:np.ndarray[datetime.datetime], R:np.ndarray[float], s:np.ndarray[float], t:np.ndarray[float], 
                                  vid_blinks:np.ndarray[bool], mocap_blinks:np.ndarray[bool], search_seconds:float = 120.0, mocap_fps:float = 25, n_anchor_frames:int = 30, k:int = 5, quantiles:np.ndarray = np.array([0.1, 0.5, 0.9]), 
                                  spatial_weight:float = 1.0, density_weight:float = 0.2, blink_weight:float = 0.25, timestamp_weight:float = 0.15):
    """
    Estimate the correction to nominal MoCap timestamps.

    Convention
    ----------
    corrected_mocap_time = nominal_mocap_time + offset_seconds

    Therefore:
      positive offset => MoCap timestamps must be shifted later
      negative offset => MoCap timestamps must be shifted earlier
    """

    # Define offsets to be compared
    # step = 1.0 / mocap_fps
    step = 1
    offsets = np.arange(-search_seconds, search_seconds + step / 2, step)

    # Find overlap of video and MoCap batch
    vid_times_s = _datetime_to_seconds(vid_times)
    mocap_times_s = _datetime_to_seconds(mocap_times)
    start, _ = get_temporal_overlap(vid_times_s, mocap_times_s + search_seconds) # Be conservative with overlap so that different offsets can be explored
    _, end = get_temporal_overlap(vid_times_s, mocap_times_s - search_seconds)
    candidate_video_frames = np.where((vid_times_s >= start) & (vid_times_s <= end))[0]

    if len(candidate_video_frames) == 0:
        raise ValueError("No candidate temporal overlap.")

    if len(candidate_video_frames) > n_anchor_frames:
        anchor_idx = np.linspace(0, len(candidate_video_frames) - 1, n_anchor_frames).round().astype(int) # Pick uniformly

        anchor_video_frames = candidate_video_frames[anchor_idx]

    else:
        anchor_video_frames = candidate_video_frames

    # ------------------------------------------------------------
    # Evaluate every candidate offset.
    # ------------------------------------------------------------

    # blink_scores = np.full(len(offsets), np.nan)
    spatial_dists = np.full((5, len(offsets)), np.nan)
    density_dists = np.full((len(quantiles) + 1, len(offsets)), np.nan)

    # Iterate over offsets
    for i, offset in enumerate(tqdm(offsets, "Computing scores for offsets.")):

        # Get scores for this offset
        # spatial_err, d_score, b_score = score_offset(offset, anchor_video_frames, vid_times, mocap_times, vid_ds, mocap_df, R, s, t, vid_blinks, mocap_blinks, mocap_fps, k, quantiles)
        spatial_dist, d_dist = score_offset(offset, anchor_video_frames, vid_times, mocap_times, vid_ds, mocap_df, R, s, t, vid_blinks, mocap_blinks, mocap_fps, k, quantiles)

        # Store scores
        # spatial_errors[i] = spatial_err
        # density_scores[i] = d_score
        # blink_scores[i] = b_score

        spatial_dists[:,i] = spatial_dist
        density_dists[:,i] = d_dist

    

    # ------------------------------------------------------------
    # Convert metrics onto comparable scales.
    #
    # spatial: LOWER is better, hence minus sign.
    # density: HIGHER is better.
    # blink:   HIGHER is better.
    # ------------------------------------------------------------

    # Instantiate final results
    combined = np.zeros(len(offsets), dtype=float)
    weight_sum = np.zeros(len(offsets), dtype=float,)
    
    def add_component(values, weight):
        valid = np.isfinite(values)
        combined[valid] += (weight * values[valid])
        weight_sum[valid] += weight

    for i, row in enumerate(spatial_dists):
        if i < 3:
            add_component(_robust_zscore(-row), spatial_weight)
        else:
            add_component(_robust_zscore(row), spatial_weight)

    for i, row in enumerate(density_dists):
        add_component(_robust_zscore(row), density_weight)

    # blink_z = _robust_zscore(blink_scores)

    # Weak timestamp prior: prefer corrections near zero unless the data disagree
    timestamp_prior = -(offsets / search_seconds) ** 2

    # add_component(blink_z, blink_weight)
    add_component(timestamp_prior,timestamp_weight)

    valid = weight_sum > 0
    combined[valid] /= weight_sum[valid]
    combined[~valid] = np.nan

    best_idx = np.nanargmax(combined)

    # return {
    #     "offset_seconds": offsets,
    #     "combined_score": combined,
    #     "spatial_error": spatial_errors,
    #     "density_score": density_scores,
    #     "blink_score": blink_scores,
    #     "best_offset_seconds": float(
    #         offsets[best_idx]
    #     ),
    #     "best_index": int(best_idx),
    #     "anchor_video_frames":
    #         anchor_video_frames,
    # }

    return {"offset_seconds": offsets, "spatial_distribution": spatial_dists, "density_distribution": density_dists, "density_quantiles": quantiles, 
            "combined_score": combined, "best_offset_seconds": float(offsets[best_idx]), "best_index": int(best_idx)}

def plot_timestamp_correction(result:dict, plots_dir=str):

    offsets = result["offset_seconds"]
    quantiles = result["density_quantiles"]
    spatial_dists = result["spatial_distribution"]
    d_dists = result["density_distribution"]
    
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    ax[0].plot(offsets, spatial_dists[0,:], label = 'Median NN')
    ax[0].plot(offsets, spatial_dists[1,:], label = 'q10')
    ax[0].plot(offsets, spatial_dists[2,:], label = 'q25')
    ax[0].set_ylabel('Distance (m)')
    ax[0].legend()

    ax[1].plot(offsets, spatial_dists[3,:], label = 'Close matches')
    ax[1].plot(offsets, spatial_dists[4,:], label = 'Mutual close matches')
    ax[1].set_ylabel('Fraction')
    ax[1].legend()

    for i, q in enumerate(quantiles):
        ax[2].plot(offsets, d_dists[i,:], label = f'q{int(100*q)}')
    ax[2].plot(offsets, d_dists[-1,:], label = f'q{int(100*quantiles[-1])}-{int(100*quantiles[0])}')
    ax[2].set_ylabel('Density correlation')
    ax[2].legend()

    ax[3].plot(offsets, result["combined_score"])
    ax[3].set_ylabel("Combined score")
    ax[3].set_xlabel("MoCap timestamp correction (s)")
    
    best = result["best_offset_seconds"]
    
    for ax in ax:
        ax.axvline(best, linestyle="--")
    
    fig.suptitle(f"Best correction = {best:+.3f} s")

    fig.tight_layout()

    fig.savefig(f'{plots_dir}multimodal_temporal_calibration_distributions.png', dpi=200)

    return

if __name__ == '__main__':

    plots_path = '/output/20230329/kp_plots/calibration/'
    path_to_vid_dir = '/original/20230329/video/'
    calibration_path = '/intrinsic_calibration/arena_board_calibration/calibration_official.yaml'
    path_to_mocap_batch = '/mocap/20230329/csvs/10K_Marching_0049.csv'
    path_to_mocap_ts = '/mocap/20230329/qtm_capture_times.csv'
    h5_prep = f'/keypoints/dewarped/20230329_preprocessed_complete_dewarped_batch_0_5.0Hz.hdf5'
    vid_ds = load_preprocessed_data(h5_prep)
    mocap_df = pd.read_csv(path_to_mocap_batch)
    # R, s, t, _, _, vid_blinks, mocap_blinks = find_mocap_to_video_matrices(path_to_vid_dir, calibration_path, path_to_mocap_batch, (500, 800, 6400, 6700), plot = True, plots_path = plots_path)

    vid_times = get_times_video(path_to_vid_dir)
    mocap_times = get_times_mocap(path_to_mocap_ts, path_to_mocap_batch)

    R = np.array([[ 0.01645639, -0.99986458], [ 0.99986458,  0.01645639]])
    s = 0.0009699278288084318
    t = np.array([ 0.02765381, -0.19051078])

    result = estimate_timestamp_correction(vid_ds, mocap_df, vid_times, mocap_times, R, s, t, np.empty(2), np.empty(2))
    plot_timestamp_correction(result, plots_path)

