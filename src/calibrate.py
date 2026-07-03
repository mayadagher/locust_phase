'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import cv2
from pathlib import Path
import pandas as pd
from image_analysis import load_image_sequence
from scipy.spatial.distance import pdist
from scipy.optimize import least_squares
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import itertools
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from helper_fns import fft_timeseries
from data_handling import load_preprocessed_data
from tqdm import tqdm
'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def calibrate_camera(vid_folder:str, plot:bool = False):

    # CHECKERBOARD = (7, 5)
    # VIDEO = "calibration.mp4"
    # SAMPLE_EVERY_N_FRAMES = 30   # grab one frame per second at 30fps

    # objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    # objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # obj_points, img_points = [], []
    # cap = cv2.VideoCapture(VIDEO)
    # frame_idx = 0

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
    #         if found:
    #             criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #             corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #             obj_points.append(objp)
    #             img_points.append(corners)
    #             print(f"Frame {frame_idx}: corners found ({len(obj_points)} total)")
    #     frame_idx += 1

    # cap.release()
    # print(f"Calibrating with {len(obj_points)} frames...")
    # ret, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    # np.savez("calibration.npz", K=K, dist=dist)   # save for later use
        pass

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


def find_regions_of_interest_mocap(binary_map: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, n_buffer: int = 3):
    '''Find regions that contain high occupancy outside of arena in MoCap coordinates.'''

    # Find the definition of the arena circle in bin coordinates
    cx, cy, r = fit_circle_to_binary_heatmap(binary_map)

    # Set the value of all bins within the arena to 0
    x_idx = np.arange(binary_map.shape[1])  # col indices
    y_idx = np.arange(binary_map.shape[0])  # row indices
    xs, ys = np.meshgrid(x_idx, y_idx)      # xs/ys both shape (n_rows, n_cols)
    invalid = (np.square(xs - cx) + np.square(ys - cy)) < (1.1*r)**2 # Give a bit of allowance to make sure all of (imperfectly round) arena is excluded
    binary_map[invalid] = 0

    # Dilate by one bin in each direction (2n + 1 x 2n + 1 structuring element = n-bin-thick buffer)
    dilated = binary_dilation(binary_map, structure=np.ones((2*n_buffer + 1, 2*n_buffer + 1)))

    # Label connected components
    labeled, n_clusters = label(dilated)
    print(f"Found {n_clusters} region(s) of interest.")

    regions = []
    for i in range(1, n_clusters + 1):
        rows, cols = np.where(labeled == i)
        # World coord bounds with half-bin padding to cover full bin area
        x_min = x_edges[cols.min()]
        x_max = x_edges[cols.max() + 1]
        y_min = y_edges[rows.min()]
        y_max = y_edges[rows.max() + 1]
        regions.append({"bounds": (x_min, x_max, y_min, y_max), "n_pixels": len(rows)})
        # print(f"Region {i}: bounds = x [{x_min:.4f}, {x_max:.4f}], y [{y_min:.4f}, {y_max:.4f}]")

    # Find cx, cy, and r in the MoCap coordinates
    def index_to_world(idx, centers):
        # Linearly interpolate: idx 0 → centers[0], idx N-1 → centers[-1]
        return np.interp(idx, np.arange(len(centers)), centers)
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2  # shape: (n_x_bins,)
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2  # shape: (n_y_bins,)

    cx_world = index_to_world(cx, x_centers)
    cy_world = index_to_world(cy, y_centers)

    # Radius: average pixel size across both axes
    dx = (x_edges[-1] - x_edges[0]) / (len(x_edges) - 1)  # world units per x cell
    dy = (y_edges[-1] - y_edges[0]) / (len(y_edges) - 1)  # world units per y cell
    r_world = r * (dx + dy) / 2

    # _, ax = plt.subplots()
    # ax.imshow(binary_map, 'viridis', extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]])
    # for i, region in enumerate(regions):
    #     x_min, x_max, y_min, y_max = region['bounds']
    #     ax.add_patch(Rectangle((x_min, y_min), width = x_max - x_min, height = y_max - y_min, fill = False, color = 'white'))
    # plt.savefig('mocap_calibration.png')

    return regions, np.array([cx_world, cy_world]), r_world

def is_square_corners(pts:np.ndarray, tol:float, max_side_length:float = np.inf) -> bool:
    '''Checks whether 3 or 4 input points form corners of a square (two equal sides, correct diagonal).
    max_side_length defines the maximum allowable side length of the square.'''
    
    n = len(pts)

    assert n in [3, 4], f"Number of points ({n}) invalid; must be 3 or 4."

    # Compute and order distances by size
    dists = pdist(pts)
    d = np.sort(dists)

    # Check if first n distances are equal (shortest distances)
    sides_idx = 2*(n//2)
    sides_equal = np.sum((d[:sides_idx] - d[:sides_idx,np.newaxis]) < tol) == sides_idx**2

    if d[:sides_idx].mean() > max_side_length:
        return False
    
    # Check if last distances are proper diagonals
    diags_ok = np.sum((d[sides_idx:] - np.sqrt(2)*d[:sides_idx].mean()) < tol) == n//2
    # print(sides_equal, diags_ok)
    return sides_equal and diags_ok

def sort_pts(pts):
    ''' Match lights across frames by sorting consistently (by x then y) so that averaging is done over the same physical light each time.'''
    pts = np.asarray(pts)
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[idx]

def validate_region_of_interest_mocap(df: pd.DataFrame, regions: list[dict], mocap_arena_center:np.ndarray[float], mocap_arena_r:float, square_side_tolerance: float = 0.15):
    '''Uses single MoCap file and regions of interest to determine which regions contain the calibration block.'''

    # Score each region over all frames (one point for every frame in which there are exactly 3 points in a square-ish formation) and collect these points
    region_scores = [0] * len(regions)
    light_points = {i: [] for i in range(len(regions))}
    blink_frames = {i: np.full(np.unique(df['frame']).shape, False) for i in range(len(regions))}
    for frame_idx, frame_df in df.groupby('frame'):

        for r_idx, region in enumerate(regions):
            x_min, x_max, y_min, y_max = region['bounds']
            in_region = ((frame_df['x'] >= x_min) & (frame_df['x'] <= x_max) & (frame_df['y'] >= y_min) & (frame_df['y'] <= y_max))
            pts = frame_df.loc[in_region, ['x', 'y', 'z']].values

            if len(pts) == 3 and is_square_corners(pts[:,:2], square_side_tolerance):
                region_scores[r_idx] += 1

                # Store points
                light_points[r_idx].append(pts)
                blink_frames[r_idx][frame_idx - 1] = True # Frame indices in MoCap file start at 1

    best_idx = int(np.argmax(region_scores))
    print(f"Best region: index {best_idx} with {region_scores[best_idx]} square-corner frames.")
    # print(f"  Scores: {region_scores}")

    if region_scores[best_idx] == 0:
        raise ValueError("No region produced valid square-corner detections. Check tolerance or region bounds.")

    # Keep points in the best region
    light_points = np.array(light_points[best_idx])  # shape: (n_valid_frames, 3, 3)

    sorted_lights = np.array([sort_pts(f) for f in light_points])  # (n_frames, 3, 3)
    centroids = sorted_lights.mean(axis=0)  # (3, 3) — one centroid per light

    # print("IR light centroids (x, y, z):")
    # for i, c in enumerate(centroids):
    #     print(f"  Light {i}: ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")

    return centroids, blink_frames[best_idx]


def find_lights_video(images:list[np.ndarray], n_images:int, region:tuple, square_side_tolerance:float = 5, max_side_length:float = 55, pixel_thresh:int = 75, circle_thresh:float = 0.3, mocap_over_vis_fps:float = 5, not_ir_thresh:float = 3):
    '''Use thresholding and contours to find the calibration box (and the lights' positions).
    region defines bounds for looking for box and are important due to blurriness of photos: (ymin, ymax, xmin, xmax) -> image plotting convention.
    max_side_length defines the maximum side length of a square that fits the 4 lights. This is to exclude the screw holes in the corners of the calibration box that also form a square and have roughly the same colour and shape.'''

    # Use multiple images for precise estimate of light positions and to ensure there are some frames in which the fourth (visual) light is on/off
    images = images[:n_images]
    all_lights = []
    ir_lights = []
    vis_on_frames = np.full(len(images), False)
    y0, y1, x0, x1 = region

    # Process all images being used
    for i, img in enumerate(images):

        # Crop and threshold image to only show dark regions (IR lights are always dark, visual light is sometimes dark)
        img = cv2.imread(img)[y0:y1, x0:x1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, pixel_thresh, 255, cv2.THRESH_BINARY_INV)

        # Find contours and keep positions of circular-ish ones
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dot_list = [] # Store dots that are potentially the lights

        for cnt in contours:
            if cv2.arcLength(cnt, True) > 0:

                # Check if it's vaguely circular
                circularity = 4 * np.pi * (cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2))
                if circularity > circle_thresh:
                    cnt = cnt.squeeze()
                    M = cv2.moments(cnt)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dot_list.append(np.array([cx, cy]))

        # Check every combination of 4 points
        for combo in itertools.combinations(dot_list, 4):
            valid = is_square_corners(combo, square_side_tolerance, max_side_length)
            if valid:
                all_lights.append(sort_pts(combo))
                break

        if not valid:
            # Check every combination of 3 points (should be the case when the visual light is ON)
            for combo in itertools.combinations(dot_list, 3):
                valid = is_square_corners(combo, square_side_tolerance, max_side_length)
                if valid:
                    ir_lights.append(sort_pts(combo))
                    vis_on_frames[i] = True
                    break

    # Get median positions of the lights
    all_lights = np.median(all_lights, axis = 0) + np.array([x0, y0]) # (4, x/y) -> normal convention; adding offsets to make coordinates absolute
    ir_lights = np.median(ir_lights, axis = 0) + np.array([x0, y0])
    
    # Check which light is not an IR light
    vis_light = all_lights[~np.all(all_lights==ir_lights[:, None], axis = 2).any(axis = 0)]

    # Make sure no IR lights were erroneously counted as visual-spectrum lights because they're not exactly equal to the median in all_lights
    if len(vis_light) > 1:
        not_vis_mask = np.full(len(vis_light), False)

        # Check proximity to known IR lights
        for i, vis in enumerate(vis_light):
            not_vis_mask[i] = np.sum(np.linalg.norm(vis - ir_lights, axis = 1) < not_ir_thresh, axis = 0).astype(bool) # not_ir_thresh is in pixels
        vis_light = vis_light[~not_vis_mask]

    plt.imshow(cv2.imread(images[0])) # Plot IR coordinates on last image
    for (cx, cy) in ir_lights:
        plt.scatter(cx, cy, s = 0.75, c = 'blue')
    plt.scatter(vis_light[0][0], vis_light[0][1], s = 0.75, c = 'orange')
    plt.xlim([np.min(ir_lights, axis = 0)[0] - 100, np.max(ir_lights, axis = 0)[0] + 100])
    plt.ylim([np.min(ir_lights, axis = 0)[1] - 100, np.max(ir_lights, axis = 0)[1] + 100])
    plt.savefig('mocap_calibration2.png')

    assert len(vis_light) == 1, f"Anomalous number of visual-spectrum lights detected ({len(vis_light)}) - check pixel_thresh, square_side_tolerance, circle_thresh, max_side_length, and not_ir_thresh parameters."

    return ir_lights, vis_light[0], np.repeat(vis_on_frames, mocap_over_vis_fps) # Put frames in MoCap fps

# def estimate_similarity_transform(points_a, points_b):
#     """
#     Estimate rotation R, scale s, translation t such that:
#         points_b[i] ≈ s * R @ points_a[i] + t
#     Uses the Umeyama least-squares method. points_a, points_b: (N,2) arrays, N>=2.
#     """
#     points_a = np.asarray(points_a, dtype=np.float64)
#     points_b = np.asarray(points_b, dtype=np.float64)

#     centroid_a = points_a.mean(axis=0)
#     centroid_b = points_b.mean(axis=0)
#     a_centered = points_a - centroid_a
#     b_centered = points_b - centroid_b

#     H = a_centered.T @ b_centered
#     U, S, Vt = np.linalg.svd(H)
#     d = np.sign(np.linalg.det(Vt.T @ U.T))
#     R = Vt.T @ np.diag([1.0, d]) @ U.T

#     var_a = np.sum(a_centered ** 2)
#     s = float(np.sum(S * np.array([1.0, d])) / var_a)

#     t = centroid_b - s * (R @ centroid_a)
#     return R, s, t

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

def estimate_best_similarity_transform(points_a, points_b, allow_reflection=True):
    """
    Estimate the best similarity transform while also solving the correspondence
    ambiguity between calibration lights.

    Returns a dict containing R, s, t, rmse, permutation, determinant, and prediction.

    The returned transform maps:

        points_a -> points_b[permutation]
    """
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)

    if points_a.shape != points_b.shape:
        raise ValueError(f"Shape mismatch: {points_a.shape=} vs {points_b.shape=}")

    best = None

    for perm in itertools.permutations(range(len(points_b))):
        target = points_b[list(perm)]

        R, s, t = estimate_similarity_transform(
            points_a,
            target,
            allow_reflection=allow_reflection,
        )

        pred = apply_transform(points_a, R, s, t)
        residuals = pred - target
        rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

        candidate = {
            "R": R,
            "s": s,
            "t": t,
            "rmse": rmse,
            "permutation": perm,
            "det": float(np.linalg.det(R)),
            "predicted": pred,
            "target": target,
            "residuals": residuals,
        }

        if best is None or candidate["rmse"] < best["rmse"]:
            best = candidate

    return best

def evaluate_transform_error(points_a: np.ndarray, points_b: np.ndarray, R: np.ndarray, s: float, t: np.ndarray) -> dict:
    """
    An error report for a fitted similarity transform than a raw
    per-point distance gives you. Particularly useful when the fit came
    from a small number of points (e.g. 3 square corners), where a single
    bad point can dominate the fit without it being obvious from residuals
    alone.
 
    Returns a dict with:
 
    per_point_error : (N,) array
        Same as transform_residuals - euclidean distance between predicted
        and actual points_b, in system B's units.
 
    rmse : float
        Root-mean-square error across all points. The standard single-number
        summary; penalizes large outliers more than mean error does.
 
    mean_error, max_error, median_error : float
        Simple summary stats of per_point_error. max_error flags whether one
        point is dominating the fit (common failure mode with N=3: if
        max_error >> median_error, suspect that one corner was measured/
        labeled less accurately than the other two).
 
    error_x, error_y : (N,) arrays
        Signed per-axis residuals (predicted - actual) in system B's frame,
        *before* taking the magnitude. Lets you check for anisotropy - e.g.
        if error_x is small but error_y is large and one-sided, that's a
        different problem (axis-specific bias, or a reflection/rotation
        sign error) than if both are small and symmetric (just noise).
 
    bias_x, bias_y : float
        Mean of the signed residuals per axis. Near zero is good (errors are
        unbiased/random). Consistently nonzero in one direction suggests a
        systematic problem - e.g. a mislabeled point, or that the two
        systems aren't related by a *pure* similarity transform (a small
        amount of shear/perspective distortion, for instance, would show up
        as a directional bias that residual magnitude alone wouldn't reveal).
 
    relative_error : float
        RMSE expressed as a fraction of the average pairwise distance
        between the input points_b (i.e. error relative to the size of the
        shape itself). A 0.5-unit RMSE means very different things for a
        5-unit square versus a 1000-unit square; this normalizes for that,
        so you have one number that's comparable across different scales/
        sessions.
 
    condition_warning : str or None
        A plain-language flag if the fit looks under-constrained or
        unreliable (e.g. exactly 3 collinear-ish points, which can't fully
        pin down a similarity transform), otherwise None.
    """
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
 
    predicted_b = apply_transform(points_a, R, s, t)
    diff = predicted_b - points_b  # signed, per-axis
    per_point_error = np.linalg.norm(diff, axis=1)
 
    rmse = float(np.sqrt(np.mean(per_point_error ** 2)))
    mean_error = float(np.mean(per_point_error))
    max_error = float(np.max(per_point_error))
    median_error = float(np.median(per_point_error))
 
    error_x = diff[:, 0]
    error_y = diff[:, 1]
    bias_x = float(np.mean(error_x))
    bias_y = float(np.mean(error_y))
 
    n = points_b.shape[0]
    if n >= 2:
        pairwise = [
            np.linalg.norm(points_b[i] - points_b[j])
            for i in range(n) for j in range(i + 1, n)
        ]
        scale_ref = float(np.mean(pairwise))
    else:
        scale_ref = 1.0
    relative_error = rmse / scale_ref if scale_ref > 1e-9 else float("inf")
 
    condition_warning = None
    if n == 3:
        # Check how close the 3 points are to collinear: area of the triangle they form, normalized by the squared scale reference.
        # A near-zero area means the 3 "corners" don't actually constrain rotation well (e.g. two corners that are nearly the same point,
        # or three points that fall on close to a straight line).
        area = 0.5 * abs(
            (points_a[1, 0] - points_a[0, 0]) * (points_a[2, 1] - points_a[0, 1])
            - (points_a[2, 0] - points_a[0, 0]) * (points_a[1, 1] - points_a[0, 1])
        )
        ref_sq = scale_ref ** 2 if scale_ref > 1e-9 else 1.0
        if area / ref_sq < 0.05:
            condition_warning = (
                "The 3 input points are close to collinear in system A - the fitted rotation/scale may be poorly constrained even "
                "if residuals look small. Prefer well-separated, non-collinear points (e.g. true square corners, not 3 points along one edge).")
 
    
    if n == 3 and max_error > 1.2 * median_error and median_error > 1e-9:
        condition_warning = (
            (condition_warning + " Also: " if condition_warning else "") +
            "Residuals are uneven across the 3 points rather than roughly equal, which can indicate one point is less accurate than the "
            "other two - but with only 3 points there isn't enough information to tell which one from the fit alone.")
 
    return {"per_point_error": per_point_error, "rmse": rmse, "mean_error": mean_error, "max_error": max_error, "median_error": median_error, "error_x": error_x, "error_y": error_y,
            "bias_x": bias_x, "bias_y": bias_y, "relative_error": relative_error, "condition_warning": condition_warning}

def apply_transform(points_a, R, s, t):
    points_a = np.asarray(points_a, dtype=np.float64)
    single = (points_a.ndim == 1)
    pts = points_a.reshape(-1, 2)
    out = s * (pts @ R.T) + t
    return out[0] if single else out

def check_transform(ds, df, R, s, t, frame, lag):

    # Transform MoCap data
    mocap_positions = df_to_padded_array(df, R, s, t) # (x/y, n_mocap_frames, max_n)

    # Get positions from video detections
    vid_positions = np.array([ds['centroid_x'].values, ds['centroid_y'].values]) # (x/y, n_vid_frames, max_n)

    fig = plt.figure()
    plt.scatter(vid_positions[0,frame,:], vid_positions[1,frame,:], alpha = 0.1)
    plt.scatter(mocap_positions[0,frame - lag,:], mocap_positions[1,frame-lag,:], alpha = 0.1)
    plt.savefig('mocap_calibration5.png')

    min_x = min(np.nanmin(mocap_positions[0,:,:]), np.nanmin(vid_positions[0,:,:]))
    max_x = min(np.nanmax(mocap_positions[0,:,:]), np.nanmax(vid_positions[0,:,:]))
    min_y = min(np.nanmin(mocap_positions[1,:,:]), np.nanmin(vid_positions[1,:,:]))
    max_y = min(np.nanmax(mocap_positions[1,:,:]), np.nanmax(vid_positions[1,:,:]))

    n_bins = 150
    x_edges = np.linspace(min_x, max_x, n_bins)
    y_edges = np.linspace(min_y, max_y, n_bins)

    vid_hist, _, _ = np.histogram2d(vid_positions[0,:,:][np.isfinite(vid_positions[0,:,:])], vid_positions[1,:,:][np.isfinite(vid_positions[1,:,:])], bins = [x_edges, y_edges], density = True)
    moc_hist, x_moc, y_moc = np.histogram2d(mocap_positions[0,:,:][np.isfinite(mocap_positions[0,:,:])], mocap_positions[1,:,:][np.isfinite(mocap_positions[1,:,:])], bins = [x_edges, y_edges], density = True)

    fig = plt.figure()
    map_min = np.min(vid_hist - moc_hist)
    map_max = np.max(vid_hist - moc_hist)
    extrema = [map_min, map_max]
    extrema[1 - np.argmax(np.abs(extrema))] = -1*np.max(np.abs(extrema))
    plt.imshow(vid_hist - moc_hist, cmap = 'bwr', vmin = extrema[0], vmax = extrema[1])
    plt.colorbar(cmap = 'bwr')
    plt.savefig('mocap_calibration7.png')

    _, ax = plt.subplots(2)
    map_min = np.min([np.min(vid_hist), np.min(moc_hist)])
    map_max = np.max([np.max(vid_hist), np.max(moc_hist)])
    extrema = [map_min, map_max]
    ax[0].imshow(vid_hist, vmin = extrema[0], vmax = extrema[1], cmap = 'viridis')
    map=ax[1].imshow(moc_hist, vmin = extrema[0], vmax = extrema[1], cmap = 'viridis')
    plt.colorbar(map, cmap = 'viridis')
    plt.savefig('mocap_calibration8.png')    

def df_to_padded_array(df, R, s, t):

    # How many detections in each frame -> max gives you the 3rd dimension
    counts = df.groupby('frame').size()
    max_detections = counts.max()

    n_frames = int(df['frame'].max()) + 1
    out = np.full((2, n_frames, max_detections), np.nan)
    frame_idx = df['frame'].to_numpy() - 1  # Use raw frame number directly as index (but correct for it starting at 1)

    # Within each frame, assign each row a slot index 0..count-1
    slot = df.groupby('frame').cumcount()
    slot_idx = slot.to_numpy()

    out[:2, frame_idx, slot_idx] = apply_transform(np.array([df['x'].to_numpy(), df['y'].to_numpy()]), R, s, t).T

    return out

def _zscore_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-12] = np.nan
    return (x - mu) / sd

def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 5:
        return np.nan
    a = a[valid]
    b = b[valid]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def _points_by_frame_from_ds(ds: xr.Dataset) -> list[np.ndarray]:
    positions = np.array([ds['centroid_x'].values, ds['centroid_y'].values])
    frames = []
    for i in range(positions.shape[1]):
        pts = positions[:, i, :].T
        frames.append(pts[np.isfinite(pts).all(axis=1)])
    return frames

def _points_by_frame_from_df(df: pd.DataFrame) -> list[np.ndarray]:
    frame_min = int(df['frame'].min())
    frame_max = int(df['frame'].max())
    frames = [np.empty((0, 2), dtype=float) for _ in range(frame_max - frame_min + 1)]

    for frame_idx, frame_df in df.groupby('frame'):
        pts = frame_df[['x', 'y']].to_numpy(dtype=float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        frames[int(frame_idx) - frame_min] = pts

    return frames

def median_knn_density_timeseries(
    frames: list[np.ndarray],
    k: int = 10,
    log_density: bool = True,
    smooth_window: int = 1,
) -> np.ndarray:
    """
    Estimate a median local-density signal from untracked point detections.

    For each frame, local density for each point is estimated from its kth
    nearest-neighbour radius: k / (pi r_k^2). The returned frame-level signal
    is the median across points. This does not require identities or a shared
    coordinate system; downstream z-scoring removes absolute unit differences.
    """

    if k < 1:
        raise ValueError("k must be >= 1.")
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1.")

    density = np.full(len(frames), np.nan, dtype=float)

    for frame_idx, pts in enumerate(tqdm(frames, desc="Computing median kNN density")):
        pts = np.asarray(pts, dtype=float)
        pts = pts[np.isfinite(pts).all(axis=1)]

        if len(pts) <= k:
            continue

        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=k + 1)
        kth_dist = dists[:, k]
        valid = np.isfinite(kth_dist) & (kth_dist > 0)

        if not np.any(valid):
            continue

        local_density = k / (np.pi * kth_dist[valid]**2)
        if log_density:
            local_density = np.log(local_density)

        density[frame_idx] = np.nanmedian(local_density)

    if smooth_window > 1:
        density = _rolling_nanmean(density, smooth_window)

    return density

def _rolling_nanmean(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    half = window // 2

    for i in range(len(x)):
        start = max(0, i - half)
        stop = min(len(x), i + half + 1)
        if np.isfinite(x[start:stop]).any():
            out[i] = np.nanmean(x[start:stop])

    return out

def score_lags_with_median_density(
    df: pd.DataFrame,
    ds: xr.Dataset,
    candidate_lags: np.ndarray,
    mocap_batch: int = 0,
    mocap_batch_len: int = 7500,
    mocap_fps: float = 25,
    vid_fps: float = 5,
    k: int = 10,
    log_density: bool = True,
    smooth_seconds: float = 1.0,
    mocap_agg_seconds: float = 0.2,
    min_overlap: int = 25,
) -> dict:
    """
    Rank candidate lags using median local-density fluctuations.

    This assumes the MoCap subset is large enough to preserve population-level
    density oscillations. Absolute density units may differ between video and
    MoCap; both signals are z-scored before scoring. Positive lags use the same
    convention as temporal_synch: MoCap occurs before video.
    """

    candidate_lags = np.asarray(candidate_lags, dtype=int)
    video_frames = _points_by_frame_from_ds(ds)
    mocap_frames = _points_by_frame_from_df(df)

    vid_smooth = max(1, int(round(smooth_seconds * vid_fps)))
    mocap_smooth = max(1, int(round(smooth_seconds * mocap_fps)))
    mocap_agg = max(1, int(round(mocap_agg_seconds * mocap_fps)))

    vid_density = median_knn_density_timeseries(
        video_frames,
        k=k,
        log_density=log_density,
        smooth_window=vid_smooth,
    )
    mocap_density = median_knn_density_timeseries(
        mocap_frames,
        k=k,
        log_density=log_density,
        smooth_window=mocap_smooth,
    )

    scores = _score_lags_from_density_traces(
        vid_density,
        mocap_density,
        candidate_lags,
        mocap_batch=mocap_batch,
        mocap_batch_len=mocap_batch_len,
        mocap_fps=mocap_fps,
        vid_fps=vid_fps,
        mocap_agg_seconds=mocap_agg_seconds,
        min_overlap=min_overlap,
    )

    best_lag = candidate_lags[np.nanargmax(scores)] if np.isfinite(scores).any() else None

    return {
        "candidate_lags": candidate_lags,
        "score_by_lag": scores,
        "best_lag": best_lag,
        "video_density": vid_density,
        "mocap_density": mocap_density,
        "video_density_z": _zscore_columns(vid_density[:, np.newaxis])[:, 0],
        "mocap_density_z": _zscore_columns(mocap_density[:, np.newaxis])[:, 0],
    }

def _score_lags_from_density_traces(
    vid_density: np.ndarray,
    mocap_density: np.ndarray,
    candidate_lags: np.ndarray,
    mocap_batch: int = 0,
    mocap_batch_len: int = 7500,
    mocap_fps: float = 25,
    vid_fps: float = 5,
    mocap_agg_seconds: float = 0.2,
    min_overlap: int = 25,
) -> np.ndarray:
    vid_z = _zscore_columns(vid_density[:, np.newaxis])[:, 0]
    mocap_z = _zscore_columns(mocap_density[:, np.newaxis])[:, 0]

    scores = np.full(len(candidate_lags), np.nan)
    video_frame_idx = np.arange(len(vid_z))
    fps_ratio = mocap_fps / vid_fps
    batch_offset = mocap_batch_len * mocap_batch
    mocap_agg = max(1, int(round(mocap_agg_seconds * mocap_fps)))
    half_agg = mocap_agg // 2

    for lag_idx, lag in enumerate(candidate_lags):
        mocap_idx = np.round(video_frame_idx * fps_ratio - lag - batch_offset).astype(int)
        mocap_vals = np.full(len(video_frame_idx), np.nan)

        for i, m_idx in enumerate(mocap_idx):
            start = max(0, m_idx - half_agg)
            stop = min(len(mocap_z), m_idx + half_agg + 1)
            if start < stop and np.isfinite(mocap_z[start:stop]).any():
                mocap_vals[i] = np.nanmean(mocap_z[start:stop])

        valid = np.isfinite(vid_z) & np.isfinite(mocap_vals)
        if np.sum(valid) < min_overlap:
            continue

        scores[lag_idx] = _corr_1d(vid_z[valid], mocap_vals[valid])

    return scores

def score_lags_with_median_density_sweep(
    df: pd.DataFrame,
    ds: xr.Dataset,
    candidate_lags: np.ndarray,
    k_values: list[int] | tuple[int, ...] = (5, 10, 20, 40),
    smooth_seconds_values: list[float] | tuple[float, ...] = (0.4, 0.8, 1.5, 2.5),
    mocap_batch: int = 0,
    mocap_batch_len: int = 7500,
    mocap_fps: float = 25,
    vid_fps: float = 5,
    log_density: bool = True,
    mocap_agg_seconds: float = 0.2,
    min_overlap: int = 25,
) -> dict:
    """
    Robustness sweep for the median-density lag scorer.

    Each k/smoothing combination gets its own score-by-lag curve. Stable lag
    choices across this grid are much stronger evidence than one best curve.
    """

    candidate_lags = np.asarray(candidate_lags, dtype=int)
    k_values = list(k_values)
    smooth_seconds_values = list(smooth_seconds_values)
    video_frames = _points_by_frame_from_ds(ds)
    mocap_frames = _points_by_frame_from_df(df)

    raw_video_density = {}
    raw_mocap_density = {}
    scores = np.full((len(k_values), len(smooth_seconds_values), len(candidate_lags)), np.nan)
    best_lags = np.full((len(k_values), len(smooth_seconds_values)), np.nan)

    for k_idx, k in enumerate(k_values):
        raw_video_density[k] = median_knn_density_timeseries(
            video_frames,
            k=k,
            log_density=log_density,
            smooth_window=1,
        )
        raw_mocap_density[k] = median_knn_density_timeseries(
            mocap_frames,
            k=k,
            log_density=log_density,
            smooth_window=1,
        )

        for smooth_idx, smooth_seconds in enumerate(smooth_seconds_values):
            vid_smooth = max(1, int(round(smooth_seconds * vid_fps)))
            mocap_smooth = max(1, int(round(smooth_seconds * mocap_fps)))
            vid_density = _rolling_nanmean(raw_video_density[k], vid_smooth)
            mocap_density = _rolling_nanmean(raw_mocap_density[k], mocap_smooth)

            scores[k_idx, smooth_idx] = _score_lags_from_density_traces(
                vid_density,
                mocap_density,
                candidate_lags,
                mocap_batch=mocap_batch,
                mocap_batch_len=mocap_batch_len,
                mocap_fps=mocap_fps,
                vid_fps=vid_fps,
                mocap_agg_seconds=mocap_agg_seconds,
                min_overlap=min_overlap,
            )

            if np.isfinite(scores[k_idx, smooth_idx]).any():
                best_lags[k_idx, smooth_idx] = candidate_lags[np.nanargmax(scores[k_idx, smooth_idx])]

    finite_best = best_lags[np.isfinite(best_lags)].astype(int)
    if len(finite_best) > 0:
        unique_lags, counts = np.unique(finite_best, return_counts=True)
        consensus_lag = int(unique_lags[np.argmax(counts)])
        consensus_fraction = float(np.max(counts) / len(finite_best))
    else:
        consensus_lag = None
        consensus_fraction = np.nan

    median_score_by_lag = np.nanmedian(scores.reshape(-1, len(candidate_lags)), axis=0)
    median_best_lag = candidate_lags[np.nanargmax(median_score_by_lag)] if np.isfinite(median_score_by_lag).any() else None

    return {
        "candidate_lags": candidate_lags,
        "k_values": np.asarray(k_values),
        "smooth_seconds_values": np.asarray(smooth_seconds_values),
        "score_by_lag": scores,
        "best_lags": best_lags,
        "consensus_lag": consensus_lag,
        "consensus_fraction": consensus_fraction,
        "median_score_by_lag": median_score_by_lag,
        "median_best_lag": median_best_lag,
        "raw_video_density": raw_video_density,
        "raw_mocap_density": raw_mocap_density,
    }

def temporal_synch(mocap_blinks:np.ndarray[bool], vid_blinks:np.ndarray[bool], df:pd.DataFrame, ds:xr.Dataset, R:np.ndarray, s:np.ndarray, t:np.ndarray, mocap_batch:int, mocap_batch_len:int = 7500, mocap_fps:float = 25, vid_fps:float = 5, xcorr_buffer_frames:int = 1000, n_frames_per_lag:int = 50):
    '''Cross-correlate blinking time series to find relative offset. Due to periodicity of signals, need to disambiguate offset by evaluating overlap of (transformed) MoCap and visual detections.'''

    # Cross correlate MoCap w.r.t. video
    buff = min(xcorr_buffer_frames, min(len(mocap_blinks) - 1, len(vid_blinks) - 1))
    xcorr = np.correlate(vid_blinks.astype(int), mocap_blinks.astype(int), mode = 'full')[len(mocap_blinks) - 1 - buff:len(mocap_blinks) + buff] # Excerpt chunk of values around lag of 0
    lags = np.arange(len(xcorr)) - buff # Positive means MoCap before vid, and vice versa
    # plt.plot(lags, xcorr)
    # plt.savefig('mocap_calibration4.png')

    # Get dominant frequency
    _, _, _, dominant_freq = fft_timeseries(xcorr[:buff], mocap_fps) # Only using first excerpt to avoid tapering causing harmonic issues

    # Find number of frames per period (approximately)
    period_frames = int(round(mocap_fps / dominant_freq))

    # Find peaks in xcorr signal
    peak_frames, _ = find_peaks(xcorr,
                                distance=period_frames * 0.5,   # troughs at least half a period apart
                                prominence=0.1)                  # ignore shallow noise fluctuations
    
    # Use median density signal to align video and MoCap in time
    candidate_lags = lags[peak_frames]
    results = score_lags_with_median_density_sweep(df, ds, candidate_lags, mocap_batch=mocap_batch, mocap_batch_len=mocap_batch_len, mocap_fps=mocap_fps, vid_fps=vid_fps)
    lag = results['consensus_lag']

    # fig = plt.figure()
    # plt.plot(candidate_lags, results['median_score_by_lag'])
    # plt.xlabel('Lag (MoCap frames)')
    # plt.ylabel('Median score across density parameters')
    # plt.savefig('mocap_calibration9.png')

    # fig, ax = plt.subplots()
    # im = ax.imshow(results['best_lags'], aspect='auto', cmap='viridis')
    # ax.set_xticks(np.arange(len(results['smooth_seconds_values'])))
    # ax.set_xticklabels(results['smooth_seconds_values'])
    # ax.set_yticks(np.arange(len(results['k_values'])))
    # ax.set_yticklabels(results['k_values'])
    # ax.set_xlabel('Smoothing window (s)')
    # ax.set_ylabel('k nearest neighbours')
    # plt.colorbar(im, ax=ax, label='Best lag (MoCap frames)')
    # plt.savefig('mocap_calibration10.png')

    # print("Median-density median-curve best lag:", results["median_best_lag"])
    # print("Median-density consensus lag:", results["consensus_lag"])
    # print("Median-density consensus fraction:", results["consensus_fraction"])

    # Pre-load ds data
    # vid_positions = np.array([ds['centroid_x'].values, ds['centroid_y'].values]) # (x/y, n_vid_frames, max_n)
    # mocap_positions = df_to_padded_array(df, R, s, t) # (x/y, n_mocap_frames, max_n) -> also transformed into video coordinates

    # # Iterate over each peak lag
    # frame_costs = np.full((len(peak_frames), n_frames_per_lag), np.inf)
    # mocap_frames = np.random.randint(buff, len(mocap_blinks) - buff, n_frames_per_lag) # Assure there can always be a corresponding video frame regardless of lag

    # for f, lag in enumerate(lags[peak_frames]):

    #     # Determine corresponding video frames
    #     video_frames = np.round((mocap_frames + lag + mocap_batch_len * mocap_batch) * vid_fps / mocap_fps).astype(int)

    #     for i in range(n_frames_per_lag):

    #         # Get video and MoCap detections
    #         vid_f = vid_positions[:, video_frames[i], :] # Contains nans
    #         mocap_f = mocap_positions[:, mocap_frames[i], :]
    #         # print(mocap_f)

    #         # Drop nan (missing) video detections before building the tree
    #         valid = ~np.isnan(vid_f).any(axis=0)
    #         vid_pts = vid_f[:, valid].T # (n_valid_vid, 2)

    #         # Drop nan mocap detections
    #         valid = ~np.isnan(mocap_f).any(axis=0)
    #         mocap_pts = mocap_f[:, valid].T # (n_valid_mocap, 2)

    #         if vid_pts.shape[0] == 0 or mocap_pts.shape[0] == 0:
    #             frame_costs[f,i] = np.inf
    #             print('No video/MoCap points in the frame.')
    #             continue

    #         # For each MoCap detection, find its nearest video detection
    #         tree = cKDTree(vid_pts)
    #         dists, _ = tree.query(mocap_pts, k=1)

    #         # Compute score for this frame
    #         frame_costs[f, i] = np.median(dists)

    # # Find temporal offset
    # frame_costs = np.median(frame_costs, axis = 1)
    # offset = lags[peak_frames[np.argmin(frame_costs)]]  # +ve is MoCap before vid, and vice versa
    # print(offset)
    # print(np.min(frame_costs))
    # print(frame_costs)


def calibrate_mocap(vid_folder:str, mocap_folder:str, ds:xr.Dataset, video_box_region:tuple, mocap_fps:float = 25, vid_fps:float = 5):
    ''' Calibrate mocap data spatially and temporally. MoCap csvs have columns: frame, time, particle_id, x, y, z. Frame and time (and also particle ids, probably) reset every file.'''

    # Load image and mocap files
    images = load_image_sequence(vid_folder)
    mocap_files = sorted(Path(mocap_folder).glob("*.csv"))

    # Load all data from first mocap file
    mocap_batch = 0
    with open(mocap_files[mocap_batch], mode='r') as mocap:
        df = pd.read_csv(mocap)

    # Create heatmap of MoCap points throughout first file and binarize it
    heatmap, x_bins, y_bins = np.histogram2d(df['x'], df['y'], 300)
    thresholded = (heatmap > np.max(heatmap)*0.02).T # Transpose to have axis 0 be columns and axis 1 be rows (image convention)

    # Find regions that might contain the calibration box
    regions, mocap_arena_center, mocap_arena_r = find_regions_of_interest_mocap(thresholded, x_bins, y_bins)

    # Determine which region contains the blinking IR lights
    mocap_irs, mocap_blinks = validate_region_of_interest_mocap(df, regions, mocap_arena_center, mocap_arena_r, square_side_tolerance = 5)

    # Find lights in the video
    vid_irs, _, vid_blinks = find_lights_video(images, 150, video_box_region, mocap_over_vis_fps = mocap_fps/vid_fps)

    # Find spatial transformation matrices
    best = estimate_best_similarity_transform(mocap_irs[:,:2], vid_irs)
    R, s, t = best['R'], best['s'], best['t']

    # err = evaluate_transform_error(mocap_irs[:,:2], vid_irs, R, s, t)
    # print(err)
    # print(vid_irs)
    # print(mocap_irs[:,:2])
    # print(apply_transform(mocap_irs[:,:2], R, s, t))
    # print(R, s, t)

    # fig = plt.figure()
    # plt.scatter(vid_irs[:,0], vid_irs[:,1])
    # transformed = apply_transform(mocap_irs[:,:2], R, s, t)
    # print(vid_irs, mocap_irs[:,:2])
    # print(np.linalg.det(R))
    # plt.scatter(transformed[:,0], transformed[:,1])
    # plt.savefig('mocap_calibration6.png')
    # check_transform(ds, df, R, s, t, 200, -455)

    # # Find time series of blinking visual light
    temporal_synch(mocap_blinks, vid_blinks, df, ds, R, s, t, mocap_batch, mocap_fps = mocap_fps, vid_fps = vid_fps)



h5_prep = f'/keypoints/20230329_preprocessed_complete_batch_1_5.0Hz.hdf5'
ds = load_preprocessed_data(h5_prep)
calibrate_mocap('/original/20230329/video/', '/mocap/20230329/', ds, (600, 900, 6300, 6600)) #(6300, 6600, 600, 900))

# is_square_corners(pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), tol = 0.15)
