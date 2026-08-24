'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import cv2
from pathlib import Path
import pandas as pd
from image_analysis import load_image_sequence
from scipy.spatial.distance import pdist
from scipy.optimize import least_squares
from scipy.ndimage import binary_dilation, label
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from helper_fns import fft_timeseries
from data_handling import load_preprocessed_data
from tqdm import tqdm
from dewarping import dewarp_img_sequence
import pickle as pkl

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def check_batch_overlap(mocap_folder:str, mocap_batch_idcs:list[int], vid_ds:xr.Dataset, mocap_fps:float = 25, vid_fps:float = 5, batch_overlap_pct:float =  0.9):
    ''' Ensure MoCap batch and the video ds batch have sufficient overlap (important for temporal synchronization).'''

    mocap_files = sorted(Path(mocap_folder).glob("*.csv"))

    # Open first file to find normal MoCap length
    with open(mocap_files[0], mode = 'r') as mocap:
        mocap_df = pd.read_csv(mocap)
        mocap_length = len(np.unique(mocap_df['frame']))

    # Find frames in all considered MoCap files for calibration
    # NOTE: Consecutive batches are assumed to immediately follow each other
    all_mocap_frames = []
    for mocap_batch_idx in mocap_batch_idcs:
        with open(mocap_files[mocap_batch_idx], mode='r') as mocap:
            mocap_df = pd.read_csv(mocap)
            mocap_frames = np.unique(mocap_df['frame'])

            # Convert frames to ds values (assuming lag of 0)
            mocap_frames = (mocap_frames + mocap_length * mocap_batch_idx) * vid_fps / mocap_fps

        all_mocap_frames.append(mocap_frames)

    all_mocap_frames = np.concatenate(all_mocap_frames, axis = 0)

    # Determine percentage of overlap with video data
    ds_frames = np.unique(vid_ds.frame.values)
    overlap = np.mean((all_mocap_frames >= ds_frames[0]) & (all_mocap_frames <= ds_frames[-1]))

    assert overlap > batch_overlap_pct, f"MoCap batch ({mocap_batch_idx}) and ds batch do not sufficiently overlap for calibration."

    return mocap_length

def sort_pts(pts):
    ''' Match lights across frames by sorting consistently (by x then y) so that averaging is done over the same physical light each time.'''
    pts = np.asarray(pts)
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[idx]

def is_square_corners(pts:np.ndarray, square_tol:float, diag_tol:float = 0.3, max_side_length:float = np.inf) -> bool:
    '''Checks whether 3 or 4 input points form corners of a square (two equal sides, correct diagonal).
    max_side_length defines the maximum allowable side length of the square.'''
    
    n = len(pts)

    assert n in [3, 4], f"Number of points ({n}) invalid; must be 3 or 4."

    # Compute and order distances by size
    dists = pdist(pts)
    d = np.sort(dists)

    # Check if first n distances are equal (shortest distances)
    sides_idx = 2*(n//2)
    sides_equal = np.sum(np.abs(d[:sides_idx] - d[:sides_idx,np.newaxis]) < square_tol) == sides_idx**2

    if d[:sides_idx].mean() > max_side_length:
        return False
    
    # Check if last distances are proper diagonals
    diags_ok = np.sum(np.abs(d[sides_idx:] - np.sqrt(2)*d[:sides_idx].mean()) < diag_tol) == n//2

    return sides_equal and diags_ok

def find_lights_video(images:list[np.ndarray], world_bounds:dict, px_per_m:float, warp_matrix:np.ndarray, video_box_region:tuple, pixel_thresh:int = 75, circle_thresh = 0.3, 
                      square_tol:float = 5, diag_tol:float = 0.3, max_side_length:float = 55, mocap_fps:float = 25, vid_fps:float = 5, plot:bool = False, plots_path:str = None):
    ''' Find positions of IR lights in dewarped video coordinates, as well as the frames in which they are on.'''

    # STEP 1: Transform video_box_region from original image pixel coordinates to dewarped image pixel coordinates using the warp matrix.
    y0, y1, x0, x1 = video_box_region
    corners = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x0, y1, 1.0], [x1, y1, 1.0]], dtype=float).T # Added 1 row at the bottom because warp_matrix is 3x3
    mapped = warp_matrix @ corners
    mapped = (mapped[:2, :] / mapped[2, :]).T  # (4,2) as (x', y') in rectified image pixels -> bottom row is a scaling factor
    x_coords = mapped[:, 0]
    y_coords = mapped[:, 1]

    # Turn coordinates into integer pixel indices
    rx0 = int(np.floor(np.min(x_coords)))
    rx1 = int(np.ceil(np.max(x_coords)))
    ry0 = int(np.floor(np.min(y_coords)))
    ry1 = int(np.ceil(np.max(y_coords)))

    # Clip to rectified image bounds (use first dewarped image as reference)
    h_ref, w_ref = images[0].shape[:2]
    rx0 = max(0, min(rx0, w_ref - 1))
    rx1 = max(0, min(rx1, w_ref))
    ry0 = max(0, min(ry0, h_ref - 1))
    ry1 = max(0, min(ry1, h_ref))

    # STEP 2: Threshold cropped image to find dark, circular-ish regions, and check if 3/4 of them form a square (the calibration box)
    all_lights = []
    ir_lights = []
    vis_on_frames = np.full(len(images), False)

    # Iterate over all images
    for i, img in enumerate(images):

        # Crop and threshold image to only show dark regions (IR lights are always dark, visual light is sometimes dark)
        img = img[ry0:ry1, rx0:rx1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, pixel_thresh, 255, cv2.THRESH_BINARY_INV)

        # Find contours and keep positions of circular-ish ones
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dot_list = [] # Store dots that are potentially the lights

        # Check if contours are circular enough to be considered as potential light positions
        for cnt in contours:
            if cv2.arcLength(cnt, True) > 0:

                circularity = 4 * np.pi * (cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2))
                if circularity > circle_thresh:
                    cnt = cnt.squeeze()
                    M = cv2.moments(cnt)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dot_list.append(np.array([cx, cy])) # These are in cropped image pixel coordinates

        # Check every combination of 4 points
        found4 = False
        if len(dot_list) >= 4:
            for combo in itertools.combinations(dot_list, 4):
                if is_square_corners(combo, square_tol, diag_tol, max_side_length):
                    all_lights.append(sort_pts(combo)) # Sort points so that they are consistently ordered across frames for averaging
                    found4 = True
                    break

        if not found4 and len(dot_list) >= 3:
            # Check every combination of 3 points (should be the case when the visual light is ON)
            for combo in itertools.combinations(dot_list, 3):
                if is_square_corners(combo, square_tol, diag_tol, max_side_length):
                    ir_lights.append(sort_pts(combo)) # Sort points so that they are consistently ordered across frames for averaging
                    vis_on_frames[i] = True
                    break

    # STEP 3: Determine the positions of the IR lights and the visual-spectrum light in absolute rectified image pixel coordinates
    if len(all_lights) == 0 or len(ir_lights) == 0:
        raise ValueError("No valid calibration light detections found in video images.")

    # Compute median light positions across frames
    all_lights = np.median(np.asarray(all_lights), axis=0)
    ir_lights = np.median(np.asarray(ir_lights), axis=0)

    # Determine the visual-spectrum light as the point farthest from the 3 IR lights.
    if all_lights.ndim != 2 or all_lights.shape[1] != 2:
        raise ValueError(f"Unexpected all_lights shape: {all_lights.shape}")
    if ir_lights.ndim != 2 or ir_lights.shape[1] != 2:
        raise ValueError(f"Unexpected ir_lights shape: {ir_lights.shape}")

    # Find index of visual light as the one farthest from the 3 IR lights
    distances = np.linalg.norm(all_lights[:, None, :] - ir_lights[None, :, :], axis=2)
    mean_dist = np.mean(distances, axis=1)
    vis_idx = int(np.argmax(mean_dist))

    # Convert positions from cropped image pixel coordinates to absolute rectified image pixel coordinates by adding the offsets of the cropped region
    vis_light = all_lights[vis_idx] + np.array([rx0, ry0], dtype=float)
    ir_lights += np.array([rx0, ry0], dtype=float)

    # STEP 4: Rescale the points to arena coordinates using the calibration parameters
    xmin = world_bounds["xmin"]
    ymax = world_bounds["ymax"]

    ir_lights_arena = np.column_stack([ir_lights[:, 0] / px_per_m + xmin, ymax - ir_lights[:, 1] / px_per_m])
    vis_light_arena = np.column_stack([vis_light[0] / px_per_m + xmin, ymax - vis_light[1] / px_per_m])

    # STEP 5: Convert the vis_on_frames array to match the MoCap frame rate by repeating each video frame's boolean value
    vis_on_frames = np.repeat(vis_on_frames, round(mocap_fps/vid_fps)) # Put frames in MoCap fps

    # STEP 6: Optional -> visualize the detected lights on the first image for verification
    if plot:

        if plots_path is None:
            print('Cannot save plot because no plot directory was given.')
        
        else:
            fig = plt.figure()
            plt.imshow(images[0]) # Plot IR coordinates on first image

            # Plot IR lights in blue and visual light in orange
            ir_x = ir_lights[:, 0]
            ir_y = ir_lights[:, 1]
            plt.scatter(ir_x, ir_y, s = 3, c = 'blue', label = 'IR')
            plt.scatter(vis_light[0], vis_light[1], s = 3, c = 'orange', label = 'Visual')
            plt.xlim([np.min(ir_lights, axis = 0)[0] - 100, np.max(ir_lights, axis = 0)[0] + 100])
            plt.ylim([np.min(ir_lights, axis = 0)[1] - 100, np.max(ir_lights, axis = 0)[1] + 100])
            plt.legend(loc="upper right")
            fig.savefig(f'{plots_path}video_light_detections.png')
            plt.close(fig)

    print('IR and visual-spectrum light found in video.')
    return ir_lights_arena, vis_light_arena[0], vis_on_frames

def fit_circle_to_binary_heatmap(binary_map: np.ndarray):
    """
    Fit a circle to a binarized heatmap and return results in bin coordinate space.
    """

    # Extract boundary contour in pixel/index space
    contours = measure.find_contours(binary_map.astype(float), level=0.5)
    if not contours:
        raise ValueError("No contours found — check that binary_map has a clear filled region.")
    contour = max(contours, key=len)  # longest = arena boundary
    # find_contours returns (row, col), rows = x axis, cols = y axis
    row_pts, col_pts = contour[:, 0], contour[:, 1]

    # Algebraic initializer
    def algebraic_fit(x, y):
        A = np.column_stack([x, y, np.ones_like(x)])
        b = x**2 + y**2
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx = result[0] / 2
        cy = result[1] / 2
        r = np.sqrt(result[2] + cx**2 + cy**2)
        return cx, cy, r

    cx0, cy0, r0 = algebraic_fit(col_pts, row_pts)  # (x=col, y=row) in px

    # Geometric fit: minimize radial residuals
    def residuals(params, x, y):
        cx, cy, r = params
        return np.sqrt((x - cx)**2 + (y - cy)**2) - r

    result = least_squares(residuals, x0=[cx0, cy0, r0], args=(col_pts, row_pts), method="lm")
    cx, cy, r = result.x
    return cx, cy, r

def find_regions_of_interest_mocap(mocap_df:pd.DataFrame, n_hist_bins:int = 300, occupancy_pct:float = 0.02, n_buffer: int = 3, plot:bool = False, plots_path:str = None):
    '''Find regions that contain high occupancy outside of arena in MoCap coordinates.'''

    # STEP 1: Make an occupancy heatmap and then threshold it
    heatmap, x_edges, y_edges = np.histogram2d(mocap_df['x'], mocap_df['y'], n_hist_bins)
    binary_map = (heatmap > np.max(heatmap)*occupancy_pct).T # Transpose to have axis 0 be columns and axis 1 be rows (image convention)

    # STEP 2: Fit circle to the heatmap to exclude regions in the arena
    cx, cy, r = fit_circle_to_binary_heatmap(binary_map) # Definition of the arena circle in bin coordinates

    # Set the value of all bins within the arena to 0
    x_idx = np.arange(binary_map.shape[1])  # col indices
    y_idx = np.arange(binary_map.shape[0])  # row indices
    xs, ys = np.meshgrid(x_idx, y_idx)      # xs/ys both shape (n_rows, n_cols)
    invalid = (np.square(xs - cx) + np.square(ys - cy)) < (1.1*r)**2 # Give a bit of allowance to make sure all of (imperfectly round) arena is excluded
    binary_map[invalid] = 0

    # STEP 3: Find candidate regions

    # Dilate by one bin in each direction (2n + 1 x 2n + 1 structuring element = n-bin-thick buffer)
    dilated = binary_dilation(binary_map, structure=np.ones((2*n_buffer + 1, 2*n_buffer + 1)))

    # Label connected components
    labeled, n_clusters = label(dilated)
    print(f"Found {n_clusters} region(s) of interest.")

    regions = []
    for i in range(1, n_clusters + 1):
        rows, cols = np.where(labeled == i)

        # MoCap coord bounds with half-bin padding to cover full bin area
        x_min = x_edges[cols.min()]
        x_max = x_edges[cols.max() + 1]
        y_min = y_edges[rows.min()]
        y_max = y_edges[rows.max() + 1]

        regions.append({"bounds": (x_min, x_max, y_min, y_max), "n_pixels": len(rows)})

    # STEP 4: Convert fit arena circle parameters to MoCap coordinates (from bin coordinates)

    # Find cx, cy, and r in the MoCap coordinates
    def index_to_mocap(idx, centers):
        # Linearly interpolate: idx 0 → centers[0], idx N-1 → centers[-1]
        return np.interp(idx, np.arange(len(centers)), centers)
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2  # shape: (n_x_bins,)
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2  # shape: (n_y_bins,)

    cx_mocap = index_to_mocap(cx, x_centers)
    cy_mocap = index_to_mocap(cy, y_centers)

    # Radius: average pixel size across both axes
    dx = (x_edges[-1] - x_edges[0]) / (len(x_edges) - 1)  # MoCap units per x cell
    dy = (y_edges[-1] - y_edges[0]) / (len(y_edges) - 1)  # MoCap units per y cell
    r_mocap = r * (dx + dy) / 2

    # STEP 5: Optional -> visualize the detected regions on the heatmap for verification
    if plot:

        if plots_path is None:
            print('Cannot save plot because no plot directory was given.')

        else:
            fig, ax = plt.subplots()
            ax.imshow(binary_map, 'viridis', extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]])
            for i, region in enumerate(regions):
                x_min, x_max, y_min, y_max = region['bounds']
                ax.add_patch(Rectangle((x_min, y_min), width = x_max - x_min, height = y_max - y_min, fill = False, color = 'white'))
            fig.savefig(f'{plots_path}mocap_candidate_regions.png')
            plt.close(fig)

    print('Found candidate regions for calibration box in MoCap data.')
    return regions, np.array([cx_mocap, cy_mocap]), r_mocap

def validate_region_of_interest_mocap(mocap_df: pd.DataFrame, regions: list[dict], square_tol:float = 5, diag_tol:float = 0.3, max_side_length:float = np.inf, plot:bool = False,
                                      plots_path:str = None):
    '''Uses single MoCap file and regions of interest to determine which regions contain the calibration block.'''

    # STEP 1: Check for square formations in each region and give them points accordingly

    # Score each region over all frames (one point for every frame in which there are exactly 3 points in a square-ish formation) and collect these points
    region_scores = [0] * len(regions)
    light_points = {i: [] for i in range(len(regions))}
    blink_frames = {i: np.full(np.unique(mocap_df['frame']).shape, False) for i in range(len(regions))}
    
    # Iterate over each frame
    for frame_idx, frame_df in mocap_df.groupby('frame'):

        # Iterate over each region
        for r_idx, region in enumerate(regions):

            # Find all points in the region
            x_min, x_max, y_min, y_max = region['bounds']
            in_region = ((frame_df['x'] >= x_min) & (frame_df['x'] <= x_max) & (frame_df['y'] >= y_min) & (frame_df['y'] <= y_max))
            pts = frame_df.loc[in_region, ['x', 'y', 'z']].values

            # Check that there are exactly 3 points and that they are in a square-ish formation
            if len(pts) == 3 and is_square_corners(pts[:,:2], square_tol, diag_tol, max_side_length):
                region_scores[r_idx] += 1

                # Store points
                light_points[r_idx].append(pts)
                blink_frames[r_idx][frame_idx - 1] = True # Frame indices in MoCap file start at 1

    # STEP 2: Choose region with best score, sort and average points to find IR positions

    best_idx = int(np.argmax(region_scores))
    print(f"Best region: index {best_idx} with {region_scores[best_idx]} square-corner frames.")
    # print(f"  Scores: {region_scores}")

    if region_scores[best_idx] == 0:
        raise ValueError("No region produced valid square-corner detections. Check tolerance or region bounds.")

    # Keep points in the best region
    light_points = np.array(light_points[best_idx])  # shape: (n_valid_frames, 3 points, x/y/z)
    sorted_lights = np.array([sort_pts(f) for f in light_points])  # (n_frames, 3, 3)
    centroids = sorted_lights.mean(axis=0)  # (3, 3) — one centroid per light

    # STEP 3: Optional -> scatter IR positions in MoCap coordinates to check they are square-ish
    if plot:

        if plots_path is None:
            print('Cannot save plot because no plot directory was given.')

        else:
            fig, ax = plt.subplots()
            ax.scatter(centroids[:,0], centroids[:,1])
            ax.set_aspect('equal')
            fig.savefig(f'{plots_path}mocap_ir_positions.png')
            plt.close(fig)

    return centroids, blink_frames[best_idx]

class MocapFileSequence:
    """A lightweight, re-iterable view over frame-offset MoCap CSV files."""

    def __init__(self, file_specs, n_frames):
        self.file_specs = list(file_specs)
        self.n_frames = int(n_frames)

    def iter_dataframes(self, usecols=None):
        for spec in self.file_specs:
            frame = pd.read_csv(spec["path"], usecols=usecols)
            if "frame" in frame.columns:
                frame["frame"] = frame["frame"].to_numpy() + spec["offset"]
            yield frame

    def points_for_frame(self, frame_number):
        for spec in self.file_specs:
            if spec["first_frame"] <= frame_number <= spec["last_frame"]:
                frame = pd.read_csv(spec["path"], usecols=["frame", "x", "y"])
                local_frame = frame_number - spec["offset"]
                points = frame.loc[frame["frame"] == local_frame, ["x", "y"]].to_numpy(dtype=float)
                return points[np.isfinite(points).all(axis=1)]
        return np.empty((0, 2), dtype=float)

    @property
    def frame_bounds(self):
        return (
            min(spec["first_frame"] for spec in self.file_specs),
            max(spec["last_frame"] for spec in self.file_specs),
        )

def find_lights_mocap(mocap_folder:str, mocap_batch_idcs:list[int], mocap_batch_length: int, n_hist_bins:int = 300, occupancy_pct:float = 0.02, n_buffer:int = 3, square_tol:float = 5, diag_tol:float = 0.3,
                      plot:bool = False, plots_path:str = None, file_backed:bool = False):
    '''Find MoCap IR positions and blinks while loading only one CSV at a time.'''

    mocap_files = sorted(Path(mocap_folder).glob("*.csv"))
    if not mocap_batch_idcs:
        raise ValueError("mocap_batch_idcs must contain at least one batch")

    # A single file is sufficient to establish the static arena and candidate
    # calibration-box regions. All selected files are then scanned to score
    # those regions and assemble the blink trace.
    first_batch_idx = mocap_batch_idcs[0]
    first_df = pd.read_csv(mocap_files[first_batch_idx])
    regions, _, _ = find_regions_of_interest_mocap(
        first_df, n_hist_bins, occupancy_pct, n_buffer, plot, plots_path
    )

    region_scores = np.zeros(len(regions), dtype=int)
    light_sums = [np.zeros((3, 3), dtype=float) for _ in regions]
    light_counts = np.zeros(len(regions), dtype=int)
    blink_indices = [[] for _ in regions]
    file_specs = []
    max_global_frame = 0

    for mocap_batch_idx in mocap_batch_idcs:
        path = mocap_files[mocap_batch_idx]
        frame = first_df if mocap_batch_idx == first_batch_idx else pd.read_csv(path)
        offset = mocap_batch_idx * mocap_batch_length
        local_min = int(frame["frame"].min())
        local_max = int(frame["frame"].max())
        first_global = local_min + offset
        last_global = local_max + offset
        max_global_frame = max(max_global_frame, last_global)
        file_specs.append({
            "path": path,
            "offset": offset,
            "first_frame": first_global,
            "last_frame": last_global,
        })

        for local_frame, frame_df in frame.groupby("frame", sort=False):
            global_frame = int(local_frame) + offset
            for region_idx, region in enumerate(regions):
                x_min, x_max, y_min, y_max = region["bounds"]
                in_region = (
                    frame_df["x"].between(x_min, x_max)
                    & frame_df["y"].between(y_min, y_max)
                )
                points = frame_df.loc[in_region, ["x", "y", "z"]].to_numpy()
                if len(points) == 3 and is_square_corners(
                    points[:, :2], square_tol, diag_tol, np.inf
                ):
                    sorted_points = sort_pts(points)
                    region_scores[region_idx] += 1
                    light_sums[region_idx] += sorted_points
                    light_counts[region_idx] += 1
                    # Preserve the existing frame-1 indexing convention.
                    blink_indices[region_idx].append(max(0, global_frame - 1))

        if frame is not first_df:
            del frame

    best_idx = int(np.argmax(region_scores))
    if region_scores[best_idx] == 0:
        raise ValueError("No region produced valid square-corner detections")
    mocap_irs = light_sums[best_idx] / light_counts[best_idx]
    mocap_blinks = np.zeros(max_global_frame, dtype=bool)
    mocap_blinks[np.asarray(blink_indices[best_idx], dtype=int)] = True
    print(f"Best region: index {best_idx} with {region_scores[best_idx]} square-corner frames.")

    if plot and plots_path is not None:
        fig, axis = plt.subplots()
        axis.scatter(mocap_irs[:, 0], mocap_irs[:, 1])
        axis.set_aspect("equal")
        fig.savefig(f"{plots_path}mocap_ir_positions.png")
        plt.close(fig)

    source = MocapFileSequence(file_specs, max_global_frame)
    if file_backed:
        return mocap_irs, mocap_blinks, source

    # Compatibility option for callers that explicitly need one DataFrame.
    mocap_df = pd.concat(source.iter_dataframes(), ignore_index=True)
    return mocap_irs, mocap_blinks, mocap_df

def _rolling_nanmean(x: np.ndarray, window: int) -> np.ndarray:
    ''' Takes a rolling average, accounting for nans.'''

    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    half = window // 2

    for i in range(len(x)):
        start = max(0, i - half)
        stop = min(len(x), i + half + 1)
        if np.isfinite(x[start:stop]).any():
            out[i] = np.nanmean(x[start:stop])
    return out

def median_knn_density_timeseries(positions: np.ndarray, k: int = 5, log_density: bool = True, smooth_window: int = 1) -> np.ndarray:
    """
    Estimate a median local (topological) density signal from untracked point detections.

    For each frame, local density for each point is estimated from its kth
    nearest-neighbour radius: k / (pi r_k^2). The returned frame-level signal
    is the median across points.
    """

    if k < 1:
        raise ValueError("k must be >= 1.")
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1.")

    density = np.full(positions.shape[1], np.nan, dtype=float)

    for frame_idx in tqdm(range(positions.shape[1]), desc="Computing median kNN density"):

        # Extract only finite points
        pts = positions[:,frame_idx,:] # (x/y, n_ids)
        pts = pts[:, np.isfinite(pts).all(axis=0)].T

        if len(pts) <= k:
            print('Insufficient points for KNN.')
            continue

        # Find k nearest neighbours
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=k + 1)
        kth_dist = dists[:, k]
        valid = np.isfinite(kth_dist) & (kth_dist > 0)

        if not np.any(valid):
            continue

        # Compute the local density and optionally take the logarithm
        local_density = k / (np.pi * kth_dist[valid]**2)
        if log_density:
            local_density = np.log(local_density)

        density[frame_idx] = np.nanmedian(local_density)

    if smooth_window > 1:
        density = _rolling_nanmean(density, smooth_window)

    return density

def median_knn_density_from_mocap(source, k=5, log_density=True):
    """Compute frame density directly from a DataFrame or file-backed source.

    Unlike ``df_to_padded_array``, memory use is proportional to one frame (or
    one CSV while reading), not ``frames * maximum_detections_per_frame``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    if isinstance(source, MocapFileSequence):
        density = np.full(source.n_frames, np.nan, dtype=float)
        dataframes = source.iter_dataframes(usecols=["frame", "x", "y"])
    elif isinstance(source, pd.DataFrame):
        if not {"frame", "x", "y"}.issubset(source.columns):
            raise ValueError("MoCap DataFrame must contain frame, x, and y")
        density = np.full(int(source["frame"].max()), np.nan, dtype=float)
        dataframes = (source,)
    else:
        raise TypeError("source must be a pandas DataFrame or MocapFileSequence")

    for dataframe in dataframes:
        for frame_number, frame_df in dataframe.groupby("frame", sort=False):
            density_index = int(frame_number) - 1
            if density_index < 0 or density_index >= len(density):
                continue
            points = frame_df[["x", "y"]].to_numpy(dtype=float)
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) <= k:
                continue
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=k + 1)
            kth_distance = distances[:, k]
            valid = np.isfinite(kth_distance) & (kth_distance > 0)
            if not valid.any():
                continue
            local_density = k / (np.pi * kth_distance[valid] ** 2)
            if log_density:
                local_density = np.log(local_density)
            density[density_index] = np.nanmedian(local_density)

    return density

def df_to_padded_array(df: pd.DataFrame, transform:bool = False, R:np.ndarray = None, s:np.ndarray = None, t:np.ndarray = None):

    # How many detections in each frame -> max gives you the 3rd dimension
    counts = df.groupby('frame').size()
    max_detections = counts.max()

    n_frames = int(df['frame'].max())
    out = np.full((2, n_frames, max_detections), np.nan)
    frame_idx = df['frame'].to_numpy() - 1  # Use raw frame number directly as index (but correct for it starting at 1)

    # Within each frame, assign each row a slot index 0..count-1
    slot = df.groupby('frame').cumcount()
    slot_idx = slot.to_numpy()
    
    if transform:
        out[:, frame_idx, slot_idx] = apply_transform(np.array([df['x'].to_numpy(), df['y'].to_numpy()]), R, s, t).T

    else:
        out[:, frame_idx, slot_idx] = np.array([df['x'].to_numpy(), df['y'].to_numpy()])

    return out

def normalized_cross_correlation(x:np.ndarray, y:np.ndarray, min_overlap:int = 2):
    """Pearson correlation at each lag, using only finite overlapping samples.
    Positive lag means y starts after x."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initialize basic arrays
    min_overlap = max(2, int(min_overlap))
    lags = np.arange(-(len(y) - 1), len(x))
    scores = np.full(len(lags), np.nan, dtype=float)

    for i, lag in enumerate(lags):
        x_start = max(0, lag)
        y_start = max(0, -lag)
        overlap = min(len(x) - x_start, len(y) - y_start)

        # Ensure there's sufficient overlap to avoid tapering
        if overlap < min_overlap:
            continue

        x_overlap = x[x_start:x_start + overlap]
        y_overlap = y[y_start:y_start + overlap]
        valid = np.isfinite(x_overlap) & np.isfinite(y_overlap)
        if valid.sum() < min_overlap:
            continue

        # Exclude nans
        x_valid = x_overlap[valid]
        y_valid = y_overlap[valid]

        # Avoid divide by 0
        x_std = np.std(x_valid)
        y_std = np.std(y_valid)
        if x_std == 0 or y_std == 0:
            continue

        # Compute Pearson coeff.
        scores[i] = np.corrcoef(x_valid, y_valid)[0, 1]

    return lags, scores

def compute_lag(vid_ds:xr.Dataset, mocap_df, buffer_scale:float = 0.5, mocap_fps:float = 25, vid_fps:float = 5, plot:bool = False, 
                plots_path:str = None):
    '''Determine lag of initial MoCap data relative to the start of the video (in MoCap frames).
    A positive lag means that the MoCap recording starts after the video.
    Median densities of the video and MoCap data are used as an aperiodic signal to inform relative lag.'''

    # STEP 1: Compute median density of video and MoCap data

    # Collect points for each frame
    vid_positions = np.array([vid_ds['centroid_x'].values, vid_ds['centroid_y'].values]) # (x/y, frames, n_ids), 5 Hz

    # Compute topological densities
    vid_density = median_knn_density_timeseries(vid_positions, k = 7)
    mocap_density = median_knn_density_from_mocap(mocap_df, k=7)

    # Use z-scores to account for different number of tracked animals in video and MoCap
    def finite_zscore(values):
        values = np.asarray(values, dtype=float)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if not np.isfinite(std) or std == 0:
            return np.zeros_like(values)
        return (values - mean) / std

    # Put both density traces on the MoCap frame rate before correlating them.
    vid_z = finite_zscore(vid_density)
    mocap_z = finite_zscore(mocap_density)
    fps_ratio = mocap_fps / vid_fps
    if not np.isclose(fps_ratio, round(fps_ratio)):
        raise ValueError("mocap_fps must be an integer multiple of vid_fps")
    vid_z_rep = np.repeat(vid_z, int(round(fps_ratio)))

    # STEP 2: Cross-correlate median densities to find peaks (plausible lags)

    # Compute minimum overlap and cross-correlate
    min_overlap = max(2, int(np.ceil(buffer_scale * min(len(vid_z_rep), len(mocap_z)))))
    density_lags, density_xcorr = normalized_cross_correlation(vid_z_rep, mocap_z, min_overlap=min_overlap)
    if not np.isfinite(density_xcorr).any():
        raise ValueError("No density lag has enough finite, non-constant overlapping samples "
                        f"(minimum overlap: {min_overlap})")
    initial_lag = int(density_lags[np.nanargmax(density_xcorr)])
    print(f'Best lag: {round(initial_lag/mocap_fps, 2)} seconds.')

    if plot:

        if plots_path is None:
            print('Cannot save plot because no plot directory was given.')

        else:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.14,
                subplot_titles=("Density correlations", "Density traces"),
            )
            fig.add_trace(
                go.Scatter(x=density_lags, y=density_xcorr, name="Pearson r"),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(x0=0, dx=1 / vid_fps, y=vid_z, name="Video"),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x0=initial_lag / mocap_fps,
                    dx=1 / mocap_fps,
                    y=mocap_z,
                    name="MoCap",
                ),
                row=2,
                col=1,
            )

            # The time axes are regular, so traces use scalar x0/dx values rather
            # than explicit x arrays. Each slider step updates one scalar x0;
            # it does not retain another full shifted MoCap time array.
            valid_lags = density_lags[np.isfinite(density_xcorr)]
            stride = max(1, int(np.ceil(len(valid_lags) / 201)))
            slider_lags = valid_lags[::stride]
            slider_lags = np.unique(np.append(slider_lags, initial_lag)).astype(int)
            steps = []
            for lag_value in slider_lags:
                marker = dict(
                    type="line", x0=lag_value, x1=lag_value, y0=0, y1=1,
                    xref="x", yref="y domain", line=dict(color="crimson", width=2),
                )
                steps.append(dict(
                    method="update",
                    label=str(lag_value),
                    args=[{"x0": [lag_value / mocap_fps]}, {"shapes": [marker]}],
                ))
                # The third Plotly.update argument applies the data update only to the
                # MoCap trace (trace 2); the layout update still moves the marker.
                steps[-1]["args"].append([2])

            fig.update_layout(
                template="plotly_white",
                height=750,
                hovermode="x unified",
                shapes=[steps[np.where(slider_lags == initial_lag)[0][0]]["args"][1]["shapes"][0]],
                sliders=[dict(
                    active=int(np.where(slider_lags == initial_lag)[0][0]),
                    currentvalue=dict(prefix="MoCap lag (frames): "),
                    pad=dict(t=45),
                    steps=steps,
                )],
            )
            fig.update_xaxes(title_text="Lag (MoCap frames)", row=1, col=1)
            fig.update_yaxes(title_text="Pearson correlation (r)", row=1, col=1)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="Standardized density", row=2, col=1)
            fig.add_annotation(
                text="Positive lag = MoCap starts after video",
                x=0.5,
                y=-0.30,
                xref="paper",
                yref="paper",
                showarrow=False,
            )


            output_path = Path(plots_path)
            if output_path.suffix.lower() != ".html":
                output_path = output_path / "mocap_density_lag.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_path)

    return initial_lag

def estimate_similarity_transform(points_a, points_b, allow_reflection=True):
    """
    Estimate scale s, orthogonal matrix R, and translation t such that:

        points_b ≈ s * R @ points_a + t

    If allow_reflection=True, R may have det(R) = -1, which permits y-axis flips.
    This is required when mapping physical/MoCap coordinates into video pixel
    coordinates with origin at top left.
    """
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)

    if points_a.shape != points_b.shape:
        raise ValueError(f"Shape mismatch: {points_a.shape=} vs {points_b.shape=}")
    if points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError("points_a and points_b must both have shape (N, 2).")
    if points_a.shape[0] < 2:
        raise ValueError("At least 2 points are required.")

    centroid_a = points_a.mean(axis=0)
    centroid_b = points_b.mean(axis=0)

    a_centered = points_a - centroid_a
    b_centered = points_b - centroid_b

    H = a_centered.T @ b_centered
    U, S, Vt = np.linalg.svd(H)

    if allow_reflection:
        # Unconstrained orthogonal fit: det(R) may be +1 or -1.
        R = Vt.T @ U.T
        s = float(np.sum(S) / np.sum(a_centered ** 2))
    else:
        # Proper rotation only: det(R) forced to +1.
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, d]) @ U.T
        s = float(np.sum(S * np.array([1.0, d])) / np.sum(a_centered ** 2))

    t = centroid_b - s * (R @ centroid_a)

    return R, s, t

def apply_transform(points, R, s, t):
    """Apply a 2D similarity transform to one point or an array of points."""
    points = np.asarray(points, dtype=np.float64)
    single_point = points.ndim == 1
    points_2d = points.reshape(-1, 2)
    transformed = s * (points_2d @ R.T) + t
    return transformed[0] if single_point else transformed

def estimate_best_similarity_transform(points_a, points_b, allow_reflection=True):
    """Fit a similarity transform while resolving point correspondence.

    The returned transform maps ``points_a`` onto
    ``points_b[result["permutation"]]``.
    """
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    if points_a.shape != points_b.shape:
        raise ValueError(f"Shape mismatch: {points_a.shape=} vs {points_b.shape=}")

    best = None
    for permutation in itertools.permutations(range(len(points_b))):
        target = points_b[list(permutation)]
        R, s, t = estimate_similarity_transform(
            points_a,
            target,
            allow_reflection=allow_reflection,
        )
        predicted = apply_transform(points_a, R, s, t)
        residuals = predicted - target
        rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
        candidate = {
            "R": R,
            "s": s,
            "t": t,
            "rmse": rmse,
            "permutation": permutation,
            "det": float(np.linalg.det(R)),
            "predicted": predicted,
            "target": target,
            "residuals": residuals,
        }
        if best is None or candidate["rmse"] < best["rmse"]:
            best = candidate

    return best

def save_untransformed_mocap_heatmap(mocap_source, plots_path, n_bins=180,
                                     occupancy_fraction=0.02, chunk_size=250_000):
    """Save raw MoCap occupancy and fit its arena circle before transformation.

    ``mocap_source`` may be either a DataFrame or ``MocapFileSequence``. The
    returned center and radius are expressed in the original MoCap units.
    """
    n_bins = int(n_bins)
    chunk_size = int(chunk_size)
    occupancy_fraction = float(occupancy_fraction)
    if n_bins < 10:
        raise ValueError("n_bins must be at least 10")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not 0 < occupancy_fraction <= 1:
        raise ValueError("occupancy_fraction must be in the interval (0, 1]")
    if not isinstance(mocap_source, MocapFileSequence):
        missing = {"x", "y"}.difference(mocap_source.columns)
        if missing:
            raise ValueError(f"MoCap DataFrame is missing columns: {sorted(missing)}")

    def iter_point_chunks():
        if isinstance(mocap_source, MocapFileSequence):
            dataframes = mocap_source.iter_dataframes(usecols=["x", "y"])
        else:
            dataframes = (mocap_source,)
        for dataframe in dataframes:
            x = dataframe["x"].to_numpy(dtype=float, copy=False)
            y = dataframe["y"].to_numpy(dtype=float, copy=False)
            for start in range(0, len(x), chunk_size):
                stop = min(start + chunk_size, len(x))
                valid = np.isfinite(x[start:stop]) & np.isfinite(y[start:stop])
                if valid.any():
                    yield x[start:stop][valid], y[start:stop][valid]

    x_min = y_min = np.inf
    x_max = y_max = -np.inf
    point_count = 0
    for x_chunk, y_chunk in iter_point_chunks():
        x_min = min(x_min, np.min(x_chunk))
        x_max = max(x_max, np.max(x_chunk))
        y_min = min(y_min, np.min(y_chunk))
        y_max = max(y_max, np.max(y_chunk))
        point_count += len(x_chunk)
    if point_count == 0:
        raise ValueError("MoCap source contains no finite x/y positions")

    # Square bins ensure a fitted radius in bin units converts equally along x/y.
    span = max(x_max - x_min, y_max - y_min)
    if not np.isfinite(span) or span <= 0:
        raise ValueError("MoCap position extent must be finite and non-zero")
    padding = 0.03 * span
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_extent = span / 2 + padding
    x_edges = np.linspace(center_x - half_extent, center_x + half_extent, n_bins + 1)
    y_edges = np.linspace(center_y - half_extent, center_y + half_extent, n_bins + 1)

    histogram = np.zeros((n_bins, n_bins), dtype=float)
    for x_chunk, y_chunk in iter_point_chunks():
        chunk_histogram, _, _ = np.histogram2d(
            x_chunk, y_chunk, bins=[x_edges, y_edges]
        )
        histogram += chunk_histogram

    threshold = max(1.0, occupancy_fraction * np.max(histogram))
    binary_occupancy = histogram.T >= threshold
    cx_bin, cy_bin, radius_bins = fit_circle_to_binary_heatmap(
        np.pad(binary_occupancy, 1, mode="constant", constant_values=False)
    )
    cx_bin -= 1
    cy_bin -= 1
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    circle_center = np.array([
        np.interp(cx_bin, np.arange(n_bins), x_centers),
        np.interp(cy_bin, np.arange(n_bins), y_centers),
    ])
    circle_radius = float(radius_bins * (x_edges[1] - x_edges[0]))

    output_path = Path(plots_path)
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        output_path = output_path / "untransformed_mocap_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7, 7))
    image = axis.imshow(
        np.log1p(histogram.T),
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="viridis",
        interpolation="nearest",
    )
    axis.add_patch(plt.Circle(
        circle_center, circle_radius, fill=False, color="white", linewidth=2
    ))
    axis.scatter(*circle_center, color="white", marker="+", s=60)
    axis.set_title(
        "Untransformed MoCap occupancy\n"
        f"center=({circle_center[0]:.4g}, {circle_center[1]:.4g}), "
        f"radius={circle_radius:.4g}"
    )
    axis.set_xlabel("Raw MoCap x")
    axis.set_ylabel("Raw MoCap y")
    axis.set_aspect("equal", adjustable="box")
    fig.colorbar(image, ax=axis, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(
        "Untransformed MoCap circle: "
        f"center={circle_center}, radius={circle_radius:.6g} raw MoCap units"
    )
    return {
        "output_path": output_path,
        "center": circle_center,
        "radius": circle_radius,
        "point_count": point_count,
        "histogram": histogram,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


def validate_transform(ds, mocap_df, vid_irs, mocap_irs, plots_path, n_bins=180, occupancy_fraction=0.02, chunk_size=250_000):
    """Compare video and transformed-MoCap occupancy in world coordinates.

    The MoCap-to-video similarity transform is estimated from the IR reference
    points and applied to every MoCap position *before* its heatmap is built.
    Both heatmaps use identical square bins, and a circle is fitted to each
    thresholded occupancy boundary to make translation/scale errors visible.

    Parameters
    ----------
    ds : xarray.Dataset
        Dewarped video detections with ``centroid_x`` and ``centroid_y`` in
        arena/world units.
    mocap_df : pandas.DataFrame
        Raw MoCap detections with ``x`` and ``y`` columns.
    vid_irs, mocap_irs : array-like
        Matching calibration-light sets in video-world and raw MoCap
        coordinates. Point correspondence is resolved automatically.
    plots_path : str or Path
        Output directory, or an explicit image filename.
    """
    if int(n_bins) < 10:
        raise ValueError("n_bins must be at least 10")
    n_bins = int(n_bins)
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    occupancy_fraction = float(occupancy_fraction)
    if not 0 < occupancy_fraction <= 1:
        raise ValueError("occupancy_fraction must be in the interval (0, 1]")
    if not {"centroid_x", "centroid_y"}.issubset(ds):
        raise ValueError("ds must contain centroid_x and centroid_y")
    if not isinstance(mocap_df, MocapFileSequence):
        missing_columns = {"x", "y"}.difference(mocap_df.columns)
        if missing_columns:
            raise ValueError(f"mocap_df is missing columns: {sorted(missing_columns)}")

    vid_irs = np.asarray(vid_irs, dtype=float)
    mocap_irs = np.asarray(mocap_irs, dtype=float)
    if vid_irs.ndim != 2 or vid_irs.shape[1] != 2:
        raise ValueError("vid_irs must have shape (N, 2) in world coordinates")
    if mocap_irs.ndim != 2 or mocap_irs.shape[0] != len(vid_irs) or mocap_irs.shape[1] < 2:
        raise ValueError("mocap_irs must have shape (N, 2+) and match vid_irs")

    best_transform = estimate_best_similarity_transform(
        mocap_irs[:, :2],
        vid_irs,
        allow_reflection=True,
    )
    R = best_transform["R"]
    scale = best_transform["s"]
    translation = best_transform["t"]

    # Keep the source columns one-dimensional. The previous implementation
    # materialized video_points, raw_mocap_points, transformed_mocap_points,
    # and all_points simultaneously, requiring several full-size copies.
    video_x = np.asarray(ds["centroid_x"].values, dtype=float).ravel()
    video_y = np.asarray(ds["centroid_y"].values, dtype=float).ravel()
    video_valid = np.isfinite(video_x) & np.isfinite(video_y)
    video_point_count = int(np.count_nonzero(video_valid))

    def iter_raw_mocap_chunks():
        if isinstance(mocap_df, MocapFileSequence):
            dataframes = mocap_df.iter_dataframes(usecols=["x", "y"])
        else:
            dataframes = (mocap_df,)
        for dataframe in dataframes:
            mocap_x = dataframe["x"].to_numpy(dtype=float, copy=False)
            mocap_y = dataframe["y"].to_numpy(dtype=float, copy=False)
            for start in range(0, len(mocap_x), chunk_size):
                stop = min(start + chunk_size, len(mocap_x))
                x_chunk = mocap_x[start:stop]
                y_chunk = mocap_y[start:stop]
                valid = np.isfinite(x_chunk) & np.isfinite(y_chunk)
                if valid.any():
                    yield np.column_stack([x_chunk[valid], y_chunk[valid]])

    mocap_point_count = sum(len(chunk) for chunk in iter_raw_mocap_chunks())
    if video_point_count == 0 or mocap_point_count == 0:
        raise ValueError("Video and MoCap must both contain finite positions")

    def iter_transformed_mocap_chunks():
        """Transform bounded chunks before they enter the histogram."""
        for raw_chunk in iter_raw_mocap_chunks():
            yield apply_transform(raw_chunk, R, scale, translation)

    # Determine the shared extent without constructing combined or fully
    # transformed point arrays.
    video_min = np.array([
        np.min(video_x, where=video_valid, initial=np.inf),
        np.min(video_y, where=video_valid, initial=np.inf),
    ])
    video_max = np.array([
        np.max(video_x, where=video_valid, initial=-np.inf),
        np.max(video_y, where=video_valid, initial=-np.inf),
    ])
    mocap_min = np.full(2, np.inf)
    mocap_max = np.full(2, -np.inf)
    for transformed_chunk in iter_transformed_mocap_chunks():
        mocap_min = np.minimum(mocap_min, np.min(transformed_chunk, axis=0))
        mocap_max = np.maximum(mocap_max, np.max(transformed_chunk, axis=0))

    x_min, y_min = np.minimum(video_min, mocap_min)
    x_max, y_max = np.maximum(video_max, mocap_max)
    span = max(x_max - x_min, y_max - y_min)
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Position extent must be finite and non-zero")
    padding = 0.03 * span
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_extent = span / 2 + padding
    x_edges = np.linspace(center_x - half_extent, center_x + half_extent, n_bins + 1)
    y_edges = np.linspace(center_y - half_extent, center_y + half_extent, n_bins + 1)

    video_hist = np.zeros((n_bins, n_bins), dtype=float)
    for start in range(0, len(video_x), chunk_size):
        stop = min(start + chunk_size, len(video_x))
        valid = video_valid[start:stop]
        if valid.any():
            chunk_hist, _, _ = np.histogram2d(
                video_x[start:stop][valid],
                video_y[start:stop][valid],
                bins=[x_edges, y_edges],
            )
            video_hist += chunk_hist

    # Every MoCap chunk is transformed before histogramming. At no point is a
    # full transformed MoCap point cloud retained in memory.
    mocap_hist = np.zeros((n_bins, n_bins), dtype=float)
    for transformed_chunk in iter_transformed_mocap_chunks():
        chunk_hist, _, _ = np.histogram2d(
            transformed_chunk[:, 0],
            transformed_chunk[:, 1],
            bins=[x_edges, y_edges],
        )
        mocap_hist += chunk_hist

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    bin_size = x_edges[1] - x_edges[0]

    def fit_occupancy_circle(histogram, label):
        peak = np.max(histogram)
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError(f"{label} occupancy heatmap is empty")
        binary = histogram.T >= max(1.0, occupancy_fraction * peak)
        # Padding guarantees that contours touching a heatmap edge are closed.
        cx_bin, cy_bin, radius_bins = fit_circle_to_binary_heatmap(
            np.pad(binary, 1, mode="constant", constant_values=False)
        )
        cx_bin -= 1
        cy_bin -= 1
        center = np.array([
            np.interp(cx_bin, np.arange(n_bins), x_centers),
            np.interp(cy_bin, np.arange(n_bins), y_centers),
        ])
        return {
            "center": center,
            "radius": float(radius_bins * bin_size),
            "binary_occupancy": binary,
        }

    video_circle = fit_occupancy_circle(video_hist, "Video")
    mocap_circle = fit_occupancy_circle(mocap_hist, "MoCap")

    print(
        "Video occupancy circle (world units): "
        f"center={video_circle['center']}, radius={video_circle['radius']:.6g}"
    )
    print(
        "Transformed MoCap occupancy circle (world units): "
        f"center={mocap_circle['center']}, radius={mocap_circle['radius']:.6g}"
    )

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    plot_specs = (
        (axes[0], video_hist, video_circle, "Video occupancy"),
        (axes[1], mocap_hist, mocap_circle, "Transformed MoCap occupancy"),
    )
    for axis, histogram, circle, title in plot_specs:
        image = axis.imshow(
            np.log1p(histogram.T),
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        axis.add_patch(plt.Circle(
            circle["center"],
            circle["radius"],
            fill=False,
            color="white",
            linewidth=2,
        ))
        axis.scatter(*circle["center"], color="white", marker="+", s=50)
        axis.set_title(
            f"{title}\ncenter=({circle['center'][0]:.3f}, {circle['center'][1]:.3f}), "
            f"r={circle['radius']:.3f}"
        )
        axis.set_xlabel("Arena x (world units)")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="log(1 + count)")
    axes[0].set_ylabel("Arena y (world units)")
    fig.suptitle(
        f"Spatial calibration validation | IR fit RMSE={best_transform['rmse']:.4g}, "
        f"det(R)={best_transform['det']:.3f}"
    )
    fig.tight_layout()

    output_path = Path(plots_path)
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        output_path = output_path / "validate_transform.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot the calibration references in the same world-coordinate system.
    # best_transform maps the raw MoCap IR points onto the permuted video IR
    # targets, so this overlay directly exposes residual rotation/scale/offset.
    transformed_mocap_irs = apply_transform(
        mocap_irs[:, :2],
        R,
        scale,
        translation,
    )
    matched_video_irs = best_transform["target"]
    ir_fig, ir_axis = plt.subplots(figsize=(7, 7))
    ir_axis.scatter(
        vid_irs[:, 0],
        vid_irs[:, 1],
        s=80,
        marker="x",
        linewidths=2,
        color="tab:blue",
        label="Video IR",
        zorder=3,
    )
    ir_axis.scatter(
        transformed_mocap_irs[:, 0],
        transformed_mocap_irs[:, 1],
        s=55,
        facecolors="none",
        edgecolors="tab:orange",
        linewidths=2,
        label="Transformed MoCap IR",
        zorder=3,
    )
    for mocap_point, video_point in zip(transformed_mocap_irs, matched_video_irs):
        ir_axis.plot(
            [mocap_point[0], video_point[0]],
            [mocap_point[1], video_point[1]],
            color="0.4",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            zorder=2,
        )
    ir_axis.set_xlabel("Arena x (world units)")
    ir_axis.set_ylabel("Arena y (world units)")
    ir_axis.set_title(
        f"IR reference overlay in world coordinates\n"
        f"RMSE={best_transform['rmse']:.6g}, det(R)={best_transform['det']:.3f}"
    )
    ir_axis.set_aspect("equal", adjustable="datalim")
    ir_axis.grid(alpha=0.25)
    ir_axis.legend()
    ir_fig.tight_layout()

    plots_path_obj = Path(plots_path)
    if plots_path_obj.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        ir_output_path = plots_path_obj.with_name(
            f"{plots_path_obj.stem}_irs{plots_path_obj.suffix}"
        )
    else:
        ir_output_path = plots_path_obj / "validate_transform_irs.png"
    ir_output_path.parent.mkdir(parents=True, exist_ok=True)
    ir_fig.savefig(ir_output_path, dpi=200, bbox_inches="tight")
    plt.close(ir_fig)

    return {
        "output_path": output_path,
        "ir_output_path": ir_output_path,
        "R": R,
        "scale": scale,
        "translation": translation,
        "transform_rmse": best_transform["rmse"],
        "ir_permutation": best_transform["permutation"],
        "transformed_mocap_irs": transformed_mocap_irs,
        "video_circle": video_circle,
        "mocap_circle": mocap_circle,
        "video_point_count": video_point_count,
        "mocap_point_count": mocap_point_count,
    }
    
def validate_transform_and_lag(mocap_df, vid_irs, mocap_irs, mocap_lag, img_dir, calibration_path, frame_width, frame_height, arena_radius, mocap_fps, vid_fps, window, plots_path, crop_fraction=0.1):
    """Interactively validate the spatial transform and temporal lag.

    ``vid_irs`` must be in the arena/world coordinates returned by
    :func:`find_lights_video`, not in original or rectified-image pixels.
    ``window`` is the number of MoCap frames to inspect on either side of
    ``mocap_lag``. ``crop_fraction`` controls the side length of the centered
    square crop relative to the shorter image dimension. Positive lag means
    MoCap starts after video, so video frame ``i`` is paired with MoCap frame
    ``round(i * mocap_fps / vid_fps - lag)``.

    Returns a dictionary containing the Plotly figure, transform parameters,
    fit error, and selected video frame.
    """
    if not isinstance(mocap_df, MocapFileSequence):
        required_columns = {"frame", "x", "y"}
        missing_columns = required_columns.difference(mocap_df.columns)
        if missing_columns:
            raise ValueError(f"mocap_df is missing columns: {sorted(missing_columns)}")
    if mocap_fps <= 0 or vid_fps <= 0:
        raise ValueError("mocap_fps and vid_fps must be positive.")
    window = int(window)
    if window < 0:
        raise ValueError("window must be non-negative.")
    crop_fraction = float(crop_fraction)
    if not 0 < crop_fraction <= 1:
        raise ValueError("crop_fraction must be in the interval (0, 1].")

    vid_irs = np.asarray(vid_irs, dtype=float)
    mocap_irs = np.asarray(mocap_irs, dtype=float)
    if vid_irs.ndim != 2 or vid_irs.shape[1] != 2:
        raise ValueError("vid_irs must have shape (N, 2) in arena coordinates.")
    if mocap_irs.ndim != 2 or mocap_irs.shape[0] != vid_irs.shape[0] or mocap_irs.shape[1] < 2:
        raise ValueError("mocap_irs must have shape (N, 2+) and match vid_irs.")
    if not np.isfinite(vid_irs).all() or not np.isfinite(mocap_irs[:, :2]).all():
        raise ValueError("IR coordinates must all be finite.")

    # Sorting independently in find_lights_video/find_lights_mocap does not
    # guarantee the same physical-light ordering after a coordinate rotation.
    # Resolve that ambiguity while fitting a proper similarity transform. Both
    # inputs are Cartesian/world coordinates, so reflection is deliberately
    # disabled; the y-axis inversion happens only for plotting.

    mocap_ir_xy = mocap_irs[:, :2]
    best_transform = estimate_best_similarity_transform(
        mocap_ir_xy,
        vid_irs,
        allow_reflection=True,
    )
    R = best_transform["R"]
    scale = best_transform["s"]
    translation = best_transform["t"]
    transform_rmse = best_transform["rmse"]
    permutation = best_transform["permutation"]

    if isinstance(mocap_df, MocapFileSequence):
        mocap_frame_min, mocap_frame_max = mocap_df.frame_bounds
    else:
        mocap_frames = np.sort(mocap_df["frame"].dropna().unique().astype(int))
        if len(mocap_frames) == 0:
            raise ValueError("mocap_df contains no valid frame values")
        mocap_frame_min, mocap_frame_max = mocap_frames[0], mocap_frames[-1]

    image_files = sorted(
        path for path in Path(img_dir).iterdir()
        if path.is_file() and path.suffix in {".jpg", ".jpeg"}
    )
    if not image_files:
        raise ValueError(f"No JPEG images found in {img_dir}")

    lag_center = int(round(mocap_lag))
    lag_values = np.arange(lag_center - window, lag_center + window + 1, dtype=int)
    fps_ratio = mocap_fps / vid_fps

    # Choose a static video frame for which every slider lag maps inside the
    # available MoCap frame interval. This avoids empty edge frames merely due
    # to the validation-frame choice.
    min_video_idx = int(np.ceil((mocap_frame_min + lag_values[-1]) / fps_ratio))
    max_video_idx = int(np.floor((mocap_frame_max + lag_values[0]) / fps_ratio))
    min_video_idx = max(0, min_video_idx)
    max_video_idx = min(len(image_files) - 1, max_video_idx)
    if min_video_idx > max_video_idx:
        raise ValueError(
            "No video frame has MoCap coverage for the full lag window; "
            "reduce window or check the lag and recording ranges"
        )
    image_idx = (min_video_idx + max_video_idx) // 2

    images, world_bounds, px_per_m, _ = dewarp_img_sequence(
        img_dir,
        calibration_path,
        start=image_idx,
        end=image_idx + 1,
        frame_width=frame_width,
        frame_height=frame_height,
        arena_radius=arena_radius,
    )
    if len(images) != 1:
        raise ValueError(f"Could not load video frame {image_idx}")
    full_image_rgb = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    full_height, full_width = full_image_rgb.shape[:2]
    crop_side = max(2, int(round(min(full_height, full_width) * crop_fraction)))
    crop_side = min(crop_side, full_height, full_width)
    crop_x0 = (full_width - crop_side) // 2
    crop_y0 = (full_height - crop_side) // 2
    image_rgb = full_image_rgb[
        crop_y0:crop_y0 + crop_side,
        crop_x0:crop_x0 + crop_side,
    ]
    image_height, image_width = image_rgb.shape[:2]

    xmin = world_bounds["xmin"]
    ymax = world_bounds["ymax"]

    def arena_to_rectified_pixels(points):
        """Map arena coordinates into pixels relative to the central crop."""
        points = np.asarray(points, dtype=float)
        return np.column_stack([
            (points[:, 0] - xmin) * px_per_m - crop_x0,
            (ymax - points[:, 1]) * px_per_m - crop_y0,
        ])

    def transformed_frame_pixels(frame):
        if isinstance(mocap_df, MocapFileSequence):
            frame_points = mocap_df.points_for_frame(frame)
        else:
            frame_points = mocap_df.loc[mocap_df["frame"] == frame, ["x", "y"]].to_numpy(dtype=float)
            frame_points = frame_points[np.isfinite(frame_points).all(axis=1)]
        if len(frame_points) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        arena_points = apply_transform(frame_points, R, scale, translation)
        return arena_points, arena_to_rectified_pixels(arena_points)

    video_ir_pixels = arena_to_rectified_pixels(vid_irs)
    initial_mocap_frame = int(round(image_idx * fps_ratio - lag_center))
    initial_arena, initial_pixels = transformed_frame_pixels(initial_mocap_frame)

    fig = go.Figure()
    fig.add_trace(go.Image(z=image_rgb, name=f"video frame {image_idx}"))
    fig.add_trace(go.Scatter(
        x=video_ir_pixels[:, 0],
        y=video_ir_pixels[:, 1],
        mode="markers",
        name="video IR references",
        marker=dict(color="cyan", size=10, symbol="x"),
        hovertemplate="video IR<br>x=%{customdata[0]:.3f} m<br>y=%{customdata[1]:.3f} m<extra></extra>",
        customdata=vid_irs,
    ))
    fig.add_trace(go.Scatter(
        x=initial_pixels[:, 0],
        y=initial_pixels[:, 1],
        mode="markers",
        name="transformed MoCap",
        marker=dict(color="magenta", size=5, opacity=0.65),
        customdata=initial_arena,
        hovertemplate="MoCap transformed<br>x=%{customdata[0]:.3f} m<br>y=%{customdata[1]:.3f} m<extra></extra>",
    ))

    slider_steps = []
    for lag_value in lag_values:
        mocap_frame = int(round(image_idx * fps_ratio - lag_value))
        arena_points, pixel_points = transformed_frame_pixels(mocap_frame)
        slider_steps.append(dict(
            method="update",
            label=str(lag_value),
            args=[
                {
                    "x": [pixel_points[:, 0]],
                    "y": [pixel_points[:, 1]],
                    "customdata": [arena_points],
                },
                {"title": f"Video frame {image_idx} | MoCap frame {mocap_frame} | lag {lag_value}"},
                [2],
            ],
        ))

    fig.update_layout(
        title=f"Video frame {image_idx} | MoCap frame {initial_mocap_frame} | lag {lag_center}",
        template="plotly_white",
        width=image_width,
        height=image_height + 150,
        sliders=[dict(
            active=window,
            currentvalue=dict(prefix="MoCap lag (frames; + = MoCap starts after video): "),
            pad=dict(t=35),
            steps=slider_steps,
        )],
        margin=dict(l=20, r=20, t=60, b=100),
    )
    fig.update_xaxes(range=[0, image_width], visible=False, constrain="domain")
    fig.update_yaxes(range=[image_height, 0], visible=False, scaleanchor="x", scaleratio=1)

    output_path = plots_path + 'validate_calibration.html'
    fig.write_html(output_path)

    return {
        "R": R,
        "scale": scale,
        "translation": translation,
        "transform_rmse": transform_rmse,
        "ir_permutation": permutation,
        "video_frame": image_idx,
        "initial_mocap_frame": initial_mocap_frame,
        "crop_bounds_pixels": (crop_x0, crop_y0, crop_x0 + crop_side, crop_y0 + crop_side),
    }

from pathlib import Path
from collections.abc import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_manual_lag_overlays(
    mocap_df,
    vid_irs,
    mocap_irs,
    video_frames,
    lag_values,
    img_dir,
    calibration_path,
    frame_width,
    frame_height,
    arena_radius,
    mocap_fps,
    vid_fps,
    output_path,
    crop_fraction=0.20,
    crop_center_world=None,
    marker_size=5,
):
    """
    Save cropped video images with spatially transformed MoCap points overlaid.

    Parameters
    ----------
    mocap_df
        Either:

        1. A pandas DataFrame containing ``frame``, ``x``, and ``y`` columns.
        2. A MocapFileSequence-like object with a ``points_for_frame(frame)``
           method.

    vid_irs : array-like, shape (N, 2)
        Video IR-reference positions in arena/world coordinates.

    mocap_irs : array-like, shape (N, 2+)
        Corresponding MoCap IR-reference positions. Only x and y are used.

    video_frames : int or iterable of int
        Video-frame indices to inspect.

    lag_values : int or iterable of int
        Candidate lags in MoCap frames.

        The convention is the same as in ``validate_transform_and_lag``:

            mocap_frame = round(
                video_frame * mocap_fps / vid_fps - lag
            )

        Thus, positive lag means that MoCap starts after the video.

    output_path : str or Path
        A folder named ``manual_lag_overlays`` will be created here.

    crop_fraction : float
        Side length of the square crop as a fraction of the shorter full-image
        dimension.

    crop_center_world : tuple[float, float] or None
        Optional crop center in arena/world coordinates. When None, the crop is
        centered on the image.

    marker_size : float
        Matplotlib scatter-marker size.

    Returns
    -------
    dict
        Transform parameters, output directory, and saved file paths.
    """
    if mocap_fps <= 0 or vid_fps <= 0:
        raise ValueError("mocap_fps and vid_fps must be positive.")

    if not 0 < crop_fraction <= 1:
        raise ValueError("crop_fraction must be in the interval (0, 1].")

    def as_integer_list(values, name):
        if np.isscalar(values):
            values = [values]
        elif not isinstance(values, Iterable):
            raise TypeError(f"{name} must be an integer or an iterable.")

        values = [int(round(value)) for value in values]
        if not values:
            raise ValueError(f"{name} cannot be empty.")
        return values

    video_frames = as_integer_list(video_frames, "video_frames")
    lag_values = as_integer_list(lag_values, "lag_values")

    vid_irs = np.asarray(vid_irs, dtype=float)
    mocap_irs = np.asarray(mocap_irs, dtype=float)

    if vid_irs.ndim != 2 or vid_irs.shape[1] != 2:
        raise ValueError("vid_irs must have shape (N, 2).")

    if (
        mocap_irs.ndim != 2
        or mocap_irs.shape[0] != vid_irs.shape[0]
        or mocap_irs.shape[1] < 2
    ):
        raise ValueError(
            "mocap_irs must have shape (N, 2+) and match vid_irs."
        )

    if not np.isfinite(vid_irs).all():
        raise ValueError("vid_irs contains non-finite coordinates.")

    if not np.isfinite(mocap_irs[:, :2]).all():
        raise ValueError("mocap_irs contains non-finite x/y coordinates.")

    # Use exactly the same spatial-transform convention as the original
    # validation function.
    best_transform = estimate_best_similarity_transform(
        mocap_irs[:, :2],
        vid_irs,
        allow_reflection=True,
    )

    R = best_transform["R"]
    scale = best_transform["s"]
    translation = best_transform["t"]

    def points_for_frame(frame):
        """Extract finite MoCap x/y positions for one frame."""
        if hasattr(mocap_df, "points_for_frame"):
            points = mocap_df.points_for_frame(frame)
            points = np.asarray(points, dtype=float)

            if points.size == 0:
                return np.empty((0, 2), dtype=float)

            if points.ndim != 2 or points.shape[1] < 2:
                raise ValueError(
                    "points_for_frame() must return an array with shape "
                    "(N, 2+)."
                )

            points = points[:, :2]

        else:
            required_columns = {"frame", "x", "y"}
            missing = required_columns.difference(mocap_df.columns)

            if missing:
                raise ValueError(
                    f"mocap_df is missing columns: {sorted(missing)}"
                )

            points = mocap_df.loc[
                mocap_df["frame"] == frame,
                ["x", "y"],
            ].to_numpy(dtype=float)

        if len(points) == 0:
            return np.empty((0, 2), dtype=float)

        return points[np.isfinite(points).all(axis=1)]

    output_dir = Path(output_path) / "manual_lag_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    fps_ratio = mocap_fps / vid_fps
    saved_files = []

    for video_frame in video_frames:
        # Dewarp the selected video frame.
        images, world_bounds, px_per_m, _ = dewarp_img_sequence(
            img_dir,
            calibration_path,
            start=video_frame,
            end=video_frame + 1,
            frame_width=frame_width,
            frame_height=frame_height,
            arena_radius=arena_radius,
        )

        if len(images) != 1:
            raise ValueError(
                f"Could not load exactly one image for video frame "
                f"{video_frame}."
            )

        full_image_rgb = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
        full_height, full_width = full_image_rgb.shape[:2]

        xmin = float(world_bounds["xmin"])
        ymax = float(world_bounds["ymax"])

        def arena_to_full_pixels(points):
            """Convert arena coordinates to full rectified-image pixels."""
            points = np.asarray(points, dtype=float)

            if points.size == 0:
                return np.empty((0, 2), dtype=float)

            return np.column_stack(
                [
                    (points[:, 0] - xmin) * px_per_m,
                    (ymax - points[:, 1]) * px_per_m,
                ]
            )

        # Construct one fixed crop for this video frame. Keeping the crop fixed
        # across candidate lags makes visual comparison easier.
        crop_side = int(
            round(min(full_height, full_width) * crop_fraction)
        )
        crop_side = max(2, min(crop_side, full_height, full_width))

        if crop_center_world is None:
            center_x_px = full_width / 2
            center_y_px = full_height / 2
        else:
            crop_center_world_array = np.asarray(
                crop_center_world,
                dtype=float,
            )

            if crop_center_world_array.shape != (2,):
                raise ValueError(
                    "crop_center_world must be an (x, y) coordinate."
                )

            center_pixel = arena_to_full_pixels(
                crop_center_world_array[None, :]
            )[0]
            center_x_px, center_y_px = center_pixel

        crop_x0 = int(round(center_x_px - crop_side / 2))
        crop_y0 = int(round(center_y_px - crop_side / 2))

        # Shift the crop back inside the image if it reaches an edge.
        crop_x0 = int(np.clip(crop_x0, 0, full_width - crop_side))
        crop_y0 = int(np.clip(crop_y0, 0, full_height - crop_side))

        crop_x1 = crop_x0 + crop_side
        crop_y1 = crop_y0 + crop_side

        image_crop = full_image_rgb[
            crop_y0:crop_y1,
            crop_x0:crop_x1,
        ]

        for lag in lag_values:
            mocap_frame = int(
                round(video_frame * fps_ratio - lag)
            )

            mocap_points = points_for_frame(mocap_frame)

            if len(mocap_points):
                transformed_points = apply_transform(
                    mocap_points,
                    R,
                    scale,
                    translation,
                )

                full_pixels = arena_to_full_pixels(transformed_points)
                crop_pixels = full_pixels - np.array(
                    [crop_x0, crop_y0],
                    dtype=float,
                )

                # Only retain points visible in the crop.
                visible = (
                    (crop_pixels[:, 0] >= 0)
                    & (crop_pixels[:, 0] < crop_side)
                    & (crop_pixels[:, 1] >= 0)
                    & (crop_pixels[:, 1] < crop_side)
                )
                crop_pixels = crop_pixels[visible]

            else:
                crop_pixels = np.empty((0, 2), dtype=float)

            fig, ax = plt.subplots(
                figsize=(7, 7),
                dpi=150,
            )

            ax.imshow(image_crop)

            if len(crop_pixels):
                ax.scatter(
                    crop_pixels[:, 0],
                    crop_pixels[:, 1],
                    s=marker_size,
                    c="magenta",
                    alpha=0.65,
                    linewidths=0,
                    rasterized=True,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No MoCap points in crop",
                    transform=ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.75,
                        "edgecolor": "none",
                    },
                )

            ax.set_title(
                f"Video frame {video_frame} | "
                f"MoCap frame {mocap_frame} | "
                f"lag {lag:+d} | "
                f"{len(crop_pixels)} visible points"
            )

            ax.set_xlim(0, crop_side)
            ax.set_ylim(crop_side, 0)
            ax.set_aspect("equal")
            ax.axis("off")

            filename = (
                f"video_{video_frame:08d}"
                f"__lag_{lag:+07d}"
                f"__mocap_{mocap_frame:08d}.png"
            )
            save_path = output_dir / filename

            fig.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight",
                pad_inches=0.03,
            )
            plt.close(fig)

            saved_files.append(save_path)

    return {
        "R": R,
        "scale": scale,
        "translation": translation,
        "transform_rmse": best_transform["rmse"],
        "ir_permutation": best_transform["permutation"],
        "output_dir": output_dir,
        "saved_files": saved_files,
    }


def full_calibration(img_dir:str, calibration_path:str, mocap_folder:str, mocap_batch_idcs:list[int], vid_ds:xr.Dataset, video_box_region:tuple, mocap_fps:float = 25, vid_fps:float = 5, batch_overlap_pct:float =  0.9, frame_width:int = 7000, 
                     frame_height:int = 7000, arena_radius:int = 2, n_images:int = 150, pixel_thresh:int = 75, circle_thresh:float = 0.3, square_tol:float = 5, diag_tol:float = 0.3, max_side_length:float = 55, 
                     n_hist_bins:int = 300, occupancy_pct:float = 0.02, n_buffer:int = 3, buffer_scale:float = 0.2, plot:bool = False, plots_path:str = None):
    ''' Synchronize the MoCap data with the video data and calibrate the MoCap data to the arena coordinates. MoCap data will be spatially and temporally transformed.'''

    # STEP 0: Ensure MoCap batch and ds batch have sufficient overlap (important, assuming the lag is much smaller than the batch lengths)
    mocap_batch_length = check_batch_overlap(mocap_folder, mocap_batch_idcs, vid_ds, mocap_fps, vid_fps, batch_overlap_pct)

    # STEP 1: Load dewarped video frames and calibration parameters
    images, world_bounds, px_per_m, warp_matrix = dewarp_img_sequence(img_dir, calibration_path, end = n_images, frame_width=frame_width, frame_height=frame_height, arena_radius=arena_radius)

    # STEP 2: Find position of IR lights in the video frames and the frames in which they are "on"
    vid_irs, _, vid_blinks = find_lights_video(images, world_bounds, px_per_m, warp_matrix, video_box_region, pixel_thresh = pixel_thresh, circle_thresh = circle_thresh, square_tol = square_tol, diag_tol = diag_tol, max_side_length = max_side_length, mocap_fps = mocap_fps, vid_fps = vid_fps, plot = True, plots_path = plots_path)
    
    # STEP 3: Find position of IR lights in the MoCap data and the frames in which they are "on"
    mocap_irs, mocap_blinks, mocap_df = find_lights_mocap(mocap_folder, mocap_batch_idcs, mocap_batch_length, n_hist_bins, occupancy_pct, n_buffer, square_tol, diag_tol,plot, plots_path, file_backed=True)

    # pkl.dump([vid_blinks, mocap_blinks, mocap_df], open('calib_debug.pkl', 'wb'))
    # [vid_blinks, mocap_blinks, mocap_df] = pkl.load(open('calib_debug.pkl', 'rb'))

    # STEP 4: Compute time lag between video and MoCap data
    # mocap_lag = compute_lag(vid_ds, mocap_df, buffer_scale, mocap_fps, vid_fps, plot, plots_path)

    # save_untransformed_mocap_heatmap(mocap_df, plots_path, n_bins=180,
                                    #  occupancy_fraction=0.02, chunk_size=250_000)

    # save_manual_lag_overlays(
    # mocap_df=mocap_df,
    # vid_irs=vid_irs,
    # mocap_irs=mocap_irs,
    # video_frames=500,
    # lag_values=range(-250, 250, 1),
    # img_dir=img_dir,
    # calibration_path=calibration_path,
    # frame_width=frame_width,
    # frame_height=frame_height,
    # arena_radius=arena_radius,
    # mocap_fps=mocap_fps,
    # vid_fps=vid_fps,
    # output_path=plots_path,
    # crop_fraction=0.20,
# )

    # STEP 5: Validate calibration
    # validate_transform_and_lag(mocap_df, vid_irs, mocap_irs, mocap_lag, img_dir, calibration_path, frame_width, frame_height, arena_radius, mocap_fps, vid_fps, 50, plots_path)
    # print('validating')
    validate_transform(ds, mocap_df, vid_irs, mocap_irs, plots_path)

plots_path = '/output/20230329/kp_plots/calibration/'
calibration_path = '/intrinsics/arena_board_calibration/calibration_official.yaml'
h5_prep = f'/keypoints/dewarped/20230329_preprocessed_complete_dewarped_batch_0_5.0Hz.hdf5'
ds = load_preprocessed_data(h5_prep)
full_calibration('/original/20230329/video/', calibration_path, '/mocap/20230329/csvs/', [0, 1, 2, 3], ds, (600, 900, 6300, 6600), plot = True, plots_path = plots_path)
