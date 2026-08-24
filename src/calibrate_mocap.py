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
import itertools
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from helper_fns import fft_timeseries
from data_handling import load_preprocessed_data
from tqdm import tqdm
from dewarping import dewarp_img

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

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

def sort_pts(pts):
    ''' Match lights across frames by sorting consistently (by x then y) so that averaging is done over the same physical light each time.'''
    pts = np.asarray(pts)
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[idx]

def validate_region_of_interest_mocap(df: pd.DataFrame, regions: list[dict], mocap_arena_center:np.ndarray[float], mocap_arena_r:float, square_tol:float = 5, diag_tol:float = 0.3, max_side_length:float = np.inf):
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

            if len(pts) == 3 and is_square_corners(pts[:,:2], square_tol, diag_tol, max_side_length):
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


def find_lights_video(images:list[np.ndarray], n_images:int, region:tuple, calibration_path:str, square_tol:float = 5, diag_tol:float = 0.3, max_side_length:float = 55, pixel_thresh:int = 75, circle_thresh:float = 0.3, mocap_over_vis_fps:float = 5, not_ir_thresh:float = 3, frame_width:int = 7000, frame_height:int = 7000, arena_radius:int = 2):
    '''Use thresholding and contours to find the calibration box (and the lights' positions).
    region defines bounds for looking for box and are important due to blurriness of photos: (ymin, ymax, xmin, xmax) -> image plotting convention.
    max_side_length defines the maximum side length of a square that fits the 4 lights. This is to exclude the screw holes in the corners of the calibration box that also form a square and have roughly the same colour and shape.'''

    # Use multiple images for precise estimate of light positions and to ensure there are some frames in which the fourth (visual) light is on/off
    images = images[:n_images]

    # Read images if they are file paths and dewarp them. Also compute the warp matrix
    # so we can map the requested `region` (which is in original image pixel coords)
    # into rectified/image-arena pixel coordinates for cropping.
    dewarped_images = []
    warp_matrix = None
    for _, img in enumerate(images):

        im = cv2.imread(str(img))
        rectified, world_bounds, px_per_m, wm, _ = dewarp_img(im, calibration_path, frame_width, frame_height, arena_radius)
        dewarped_images.append(rectified)
        if warp_matrix is None:
            warp_matrix = wm

    images = dewarped_images
    all_lights = []
    ir_lights = []
    vis_on_frames = np.full(len(images), False)
    y0, y1, x0, x1 = region

    # Map the requested region (y0,y1,x0,x1 in original image pixels) into
    # rectified/output-pixel coordinates using the warp matrix computed above.
    # Region is provided as (ymin, ymax, xmin, xmax).
    if warp_matrix is None:
        raise RuntimeError("Warp matrix not available for mapping region to rectified image.")

    # Corners in original image pixel coordinates: (x, y)
    corners = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x0, y1, 1.0], [x1, y1, 1.0]], dtype=float).T
    mapped = warp_matrix @ corners
    mapped = (mapped[:2, :] / mapped[2, :]).T  # shape (4,2) as (x', y') in rectified image pixels

    x_coords = mapped[:, 0]
    y_coords = mapped[:, 1]
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

    print(world_bounds)
    print(rx0, rx1, ry0, ry1)

    # Process all images being used
    for i, img in enumerate(images):

        # Crop and threshold image to only show dark regions (IR lights are always dark, visual light is sometimes dark)
        img = img[ry0:ry1, rx0:rx1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, pixel_thresh, 255, cv2.THRESH_BINARY_INV)

        # Find contours and keep positions of circular-ish ones
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dot_list = [] # Store dots that are potentially the lights

        # plt.imshow(thresh)
        # plt.savefig(f'mocap_calibration2.png')
        # raise ValueError("Debugging: check thresholded images.")

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
        found4 = False
        if len(dot_list) >= 4:
            for combo in itertools.combinations(dot_list, 4):
                if is_square_corners(combo, square_tol, diag_tol, max_side_length):
                    all_lights.append(sort_pts(combo))
                    found4 = True
                    break

        if not found4 and len(dot_list) >= 3:
            # Check every combination of 3 points (should be the case when the visual light is ON)
            for combo in itertools.combinations(dot_list, 3):
                if is_square_corners(combo, square_tol, diag_tol, max_side_length):
                    ir_lights.append(sort_pts(combo))
                    vis_on_frames[i] = True
                    break

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

    distances = np.linalg.norm(all_lights[:, None, :] - ir_lights[None, :, :], axis=2)
    mean_dist = np.mean(distances, axis=1)
    vis_idx = int(np.argmax(mean_dist))
    vis_light = all_lights[vis_idx:vis_idx+1]
    ir_lights = np.delete(all_lights, vis_idx, axis=0)

    if vis_light.shape != (1, 2) or ir_lights.shape != (3, 2):
        raise ValueError(
            f"Unexpected visual/IR light split after selection: vis_light={vis_light.shape}, ir_lights={ir_lights.shape}. "
            "Check pixel_thresh, square_tol, diag_tol, max_side_length, and the calibration image region."
        )

    ir_lights = ir_lights.astype(float) + np.array([rx0, ry0], dtype=float)  # Add offsets to put coordinates in absolute pixel coordinates (not cropped image coordinates)
    vis_light = vis_light.astype(float) + np.array([rx0, ry0], dtype=float)

    assert len(vis_light) == 1, f"Anomalous number of visual-spectrum lights detected ({len(vis_light)}) - check pixel_thresh, square_tol, diag_tol, max_side_length, and not_ir_thresh parameters."

    plt.imshow(images[0]) # Plot IR coordinates on last image

    # Plot IR lights in blue and visual light in orange
    ir_x = ir_lights[:, 0]
    ir_y = ir_lights[:, 1]
    plt.scatter(ir_x, ir_y, s = 0.75, c = 'blue')
    plt.scatter(vis_light[0][0], vis_light[0][1], s = 0.75, c = 'orange')
    plt.xlim([np.min(ir_lights, axis = 0)[0] - 100, np.max(ir_lights, axis = 0)[0] + 100])
    plt.ylim([np.min(ir_lights, axis = 0)[1] - 100, np.max(ir_lights, axis = 0)[1] + 100])
    plt.savefig('mocap_calibration3.png')

    # Rescale the points to pixel coordinates for visualization
    xmin = world_bounds["xmin"]
    ymax = world_bounds["ymax"]

    ir_lights = np.column_stack([ir_lights[:, 0] / px_per_m + xmin, ymax - ir_lights[:, 1] / px_per_m])
    vis_light = np.column_stack([vis_light[:, 0] / px_per_m + xmin, ymax - vis_light[:, 1] / px_per_m])

    return ir_lights, vis_light[0], np.repeat(vis_on_frames, int(mocap_over_vis_fps)) # Put frames in MoCap fps

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

def check_transform(ds, df, R, s, t, frame, lag, mocap_fps: float = 25, vid_fps: float = 5):

    # Transform MoCap data
    mocap_positions = df_to_padded_array(df, R, s, t) # (x/y, n_mocap_frames, max_n)

    # Get positions from video detections (dataset-relative index)
    vid_positions = np.array([ds['centroid_x'].values, ds['centroid_y'].values]) # (x/y, n_vid_frames, max_n)

    mocap_frame = int(round(frame * mocap_fps / vid_fps - lag))
    if mocap_frame < 0 or mocap_frame >= mocap_positions.shape[1]:
        raise IndexError(f"Computed mocap_frame {mocap_frame} outside MoCap range 0-{mocap_positions.shape[1] - 1}.")

    fig = plt.figure()
    plt.scatter(vid_positions[0, frame, :], vid_positions[1, frame, :], alpha=0.1)
    plt.scatter(mocap_positions[0, mocap_frame, :], mocap_positions[1, mocap_frame, :], alpha=0.1)
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

def assess_calibration(ds: xr.Dataset, df: pd.DataFrame, R: np.ndarray, s: float, t: np.ndarray, video_frame: int, lag: int, image_path: str | Path | None = None, calibration_path: str | None = None, 
                       frame_width: int | None = 7000, frame_height: int | None = 7000, arena_radius: float | None = 2, vid_folder: str | Path | None = None,
                       mocap_batch_len: int = 7500, mocap_fps: float = 25, vid_fps: float = 5, zoom_padding: float = 250, zoom_quantile: float = 0.5,
                       save_path: str | Path = "mocap_calibration_overlay.png", ax=None):
    """
    Plot transformed MoCap detections over video detections and, optionally, the
    corresponding video frame.

    Positive lag follows temporal_synch: MoCap starts after video. The plotted
    MoCap frame is therefore:

        video_frame * mocap_fps / vid_fps - lag - batch_offset

    where batch_offset is mocap_batch_len * mocap_batch.
    """

    print('WARNING: make sure appropriate MoCap batch is input to this function, otherwise the lag will be wrong and the overlay will be meaningless.')

    video_frame_idx = int(video_frame)
    if video_frame_idx < 0 or video_frame_idx >= ds['frame'].size:
        raise IndexError(f"video_frame {video_frame_idx} outside dataset frame range 0-{ds['frame'].size - 1}.")

    # Translate the dataset-relative frame index into the actual video frame label
    # stored in the xarray coordinate. This is necessary when ds.frame does not
    # start at 0 and the image sequence is indexed by absolute video frame number.
    video_frame_label = int(ds['frame'].values[video_frame_idx])

    # Find the corresponding MoCap frame for the given actual video frame label and lag.
    batch_offset = ((video_frame_label * mocap_fps / vid_fps) // mocap_batch_len) * mocap_batch_len
    mocap_frame = int(round(video_frame_label * mocap_fps / vid_fps - lag - batch_offset))

    # Get the positions for the video
    vid_positions = np.array([ds['centroid_x'].values, ds['centroid_y'].values])
    if video_frame_idx < 0 or video_frame_idx >= vid_positions.shape[1]:
        raise IndexError(f"video_frame {video_frame_idx} outside video range 0-{vid_positions.shape[1] - 1}.")

    # Transform the positions for the MoCap data
    mocap_positions = df_to_padded_array(df, R, s, t)
    if mocap_frame < 0 or mocap_frame >= mocap_positions.shape[1]:
        raise IndexError(f"Computed mocap_frame {mocap_frame} outside MoCap range 0-{mocap_positions.shape[1] - 1}.")

    # Get rid of nans
    vid_pts = vid_positions[:, video_frame_idx, :].T
    vid_pts = vid_pts[np.isfinite(vid_pts).all(axis=1)]

    mocap_pts = mocap_positions[:, mocap_frame, :].T
    mocap_pts = mocap_pts[np.isfinite(mocap_pts).all(axis=1)]

    # Add image to plot
    image = None
    if image_path is None and vid_folder is not None:
        images = load_image_sequence(str(vid_folder))
        if video_frame_label >= len(images):
            raise IndexError(f"video_frame_label {video_frame_label} outside image sequence length {len(images)}.")
        image_path = images[video_frame_label]

    if image_path is not None:
        image = cv2.imread(str(image_path))
        
        # Dewarp image
        image, world_bounds, px_per_m, _, _ = dewarp_img(image, calibration_path, frame_width, frame_height, arena_radius)

    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))

    if image is not None:
        ax.imshow(image)

    xmin = world_bounds["xmin"]
    ymax = world_bounds["ymax"]

    # Rescale the points to pixel coordinates for visualization
    vid_x = (vid_pts[:,0] - xmin) * px_per_m
    vid_y = (ymax - vid_pts[:,1]) * px_per_m
    mocap_x = (mocap_pts[:,0] - xmin) * px_per_m
    mocap_y = (ymax - mocap_pts[:,1]) * px_per_m

    ax.scatter(vid_x, vid_y, s=5, c="deepskyblue", alpha=0.25, linewidths=0, label="Video detections")
    ax.scatter(mocap_x, mocap_y, s=22, facecolors="none", edgecolors="orange", alpha=0.9, linewidths=1.1, label="Transformed MoCap")

    # nonempty_pts = [pts for pts in [vid_pts, mocap_pts] if len(pts) > 0]
    # if len(nonempty_pts) == 0:
    #     raise ValueError("No finite video or MoCap points available for this frame/lag.")
    # zoom_pts = np.vstack(nonempty_pts)
    # q_low = (1 - zoom_quantile) / 2
    # q_high = 1 - q_low
    # x0, y0 = np.nanquantile(zoom_pts, q_low, axis=0) - zoom_padding
    # x1, y1 = np.nanquantile(zoom_pts, q_high, axis=0) + zoom_padding

    # if image is not None:
    #     h, w = image.shape[:2]
    #     x0, x1 = np.clip([x0, x1], 0, w - 1)
    #     y0, y1 = np.clip([y0, y1], 0, h - 1)

    # ax.set_xlim(x0, x1)
    # ax.set_ylim(y1, y0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Video frame {video_frame}, MoCap frame {mocap_frame}, lag {lag}")
    ax.legend(loc="upper right")

    if save_path is not None and created_fig:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    return {
        "video_frame": video_frame,
        "mocap_frame": mocap_frame,
        "video_points": vid_pts,
        "mocap_points": mocap_pts,
        "ax": ax,
    }

def compare_arena_scale_from_occupancy(
    ds: xr.Dataset,
    df: pd.DataFrame,
    R: np.ndarray,
    s: float,
    t: np.ndarray,
    n_bins: int = 180,
    center: np.ndarray | None = None,
    threshold_fraction: float = 0.15,
    save_path: str | Path = "mocap_arena_scale_check.png",
) -> dict:
    """
    Compare video and transformed-MoCap occupancy as radial profiles.

    This is a coarse arena-scale diagnostic. It avoids relying on instantaneous
    nearest-neighbour matches and instead asks whether the long-time occupancy
    fields fall off at similar radii around a shared center.
    """

    vid_positions = np.array([ds['centroid_x'].values, ds['centroid_y'].values])
    mocap_positions = df_to_padded_array(df, R, s, t)

    vid_pts = vid_positions.reshape(2, -1).T
    vid_pts = vid_pts[np.isfinite(vid_pts).all(axis=1)]
    mocap_pts = mocap_positions.reshape(2, -1).T
    mocap_pts = mocap_pts[np.isfinite(mocap_pts).all(axis=1)]

    all_pts = np.vstack([vid_pts, mocap_pts])
    x_edges = np.linspace(np.nanmin(all_pts[:, 0]), np.nanmax(all_pts[:, 0]), n_bins + 1)
    y_edges = np.linspace(np.nanmin(all_pts[:, 1]), np.nanmax(all_pts[:, 1]), n_bins + 1)

    vid_hist, _, _ = np.histogram2d(vid_pts[:, 0], vid_pts[:, 1], bins=[x_edges, y_edges], density=True)
    mocap_hist, _, _ = np.histogram2d(mocap_pts[:, 0], mocap_pts[:, 1], bins=[x_edges, y_edges], density=True)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xs, ys = np.meshgrid(x_centers, y_centers, indexing="ij")

    if center is None:
        # Video occupancy has the clearer arena boundary, so use it to estimate
        # the shared center for radial diagnostics.
        weights = np.nan_to_num(vid_hist, nan=0.0)
        if weights.sum() == 0:
            raise ValueError("Video occupancy is empty.")
        center = np.array([
            np.sum(xs * weights) / np.sum(weights),
            np.sum(ys * weights) / np.sum(weights),
        ])
    else:
        center = np.asarray(center, dtype=float)

    radii = np.sqrt((xs - center[0])**2 + (ys - center[1])**2)
    max_r = np.nanpercentile(radii, 99)
    r_edges = np.linspace(0, max_r, n_bins + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    def radial_profile(hist):
        prof = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = (radii >= r_edges[i]) & (radii < r_edges[i + 1])
            if np.any(mask):
                prof[i] = np.nanmean(hist[mask])
        return prof

    vid_profile = radial_profile(vid_hist)
    mocap_profile = radial_profile(mocap_hist)

    def estimate_radius(profile):
        if not np.isfinite(profile).any():
            return np.nan
        smooth = _rolling_nanmean(profile, max(3, n_bins // 40))
        peak = np.nanmax(smooth)
        if peak <= 0 or not np.isfinite(peak):
            return np.nan
        below = np.where(smooth < threshold_fraction * peak)[0]
        below = below[below > np.nanargmax(smooth)]
        return float(r_centers[below[0]]) if len(below) > 0 else np.nan

    vid_radius = estimate_radius(vid_profile)
    mocap_radius = estimate_radius(mocap_profile)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(r_centers, vid_profile / np.nanmax(vid_profile), label="video")
    ax[0].plot(r_centers, mocap_profile / np.nanmax(mocap_profile), label="MoCap")
    ax[0].axvline(vid_radius, color="tab:blue", linestyle="--")
    ax[0].axvline(mocap_radius, color="tab:orange", linestyle="--")
    ax[0].set_xlabel("Radius from shared center (px)")
    ax[0].set_ylabel("Normalized occupancy")
    ax[0].legend()

    ax[1].imshow(vid_hist.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap="Blues", alpha=0.65)
    ax[1].imshow(mocap_hist.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap="Oranges", alpha=0.45)
    ax[1].scatter(center[0], center[1], c="white", s=20)
    ax[1].set_aspect("equal", adjustable="box")
    ax[1].set_title("Occupancy overlay")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    return {
        "center": center,
        "radii": r_centers,
        "video_profile": vid_profile,
        "mocap_profile": mocap_profile,
        "video_radius": vid_radius,
        "mocap_radius": mocap_radius,
        "radius_ratio_mocap_over_video": mocap_radius / vid_radius if vid_radius > 0 else np.nan,
    }

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
    convention as temporal_synch: MoCap starts after video.
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

def temporal_synch(mocap_blinks:np.ndarray[bool], vid_blinks:np.ndarray[bool], df:pd.DataFrame, ds:xr.Dataset, R:np.ndarray, s:np.ndarray, t:np.ndarray, mocap_batch:int, mocap_batch_len: int = 7500, mocap_fps:float = 25, vid_fps:float = 5, xcorr_buffer_frames:int = 1000, n_frames_per_lag:int = 50):
    '''Cross-correlate blinking time series to find relative offset. Due to periodicity of signals, need to disambiguate offset by evaluating overlap of (transformed) MoCap and visual detections.'''

    # Cross correlate MoCap w.r.t. video
    buff = min(xcorr_buffer_frames, min(len(mocap_blinks) - 1, len(vid_blinks) - 1))
    xcorr = np.correlate(vid_blinks.astype(int), mocap_blinks.astype(int), mode = 'full')[len(mocap_blinks) - 1 - buff:len(mocap_blinks) + buff] # Excerpt chunk of values around lag of 0
    lags = np.arange(len(xcorr)) - buff # Positive means MoCap starts after video
    # plt.plot(lags, xcorr)
    # plt.savefig('mocap_calibration4.png')

    # Get dominant frequency
    _, _, _, dominant_freq = fft_timeseries(xcorr[:buff], mocap_fps) # Only using first excerpt to avoid tapering causing harmonic issues

    # Find number of frames per period (approximately)
    period_frames = int(round(mocap_fps / dominant_freq))

    # Find peaks in xcorr signal
    peak_frames, _ = find_peaks(xcorr,
                                distance=period_frames * 0.8,   # troughs at least x percent of a period apart
                                prominence=0.1)                  # ignore shallow noise fluctuations
    
    # Use median density signal to align video and MoCap in time
    candidate_lags = lags[peak_frames]
    # results = score_lags_with_median_density_sweep(df, ds, candidate_lags, mocap_batch=mocap_batch, mocap_batch_len=mocap_batch_len, mocap_fps=mocap_fps, vid_fps=vid_fps)
    # lag = results['consensus_lag']

    results = score_lags_with_median_density(df, ds, candidate_lags, mocap_batch, mocap_batch_len, mocap_fps, vid_fps, k = 5, smooth_seconds = 0.4) # Faster than doing the whole sweep
    lag = results['best_lag']
    print('Best lag: ', lag)

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

    return lag

def full_synchronization(vid_folder:str, calibration_path:str, mocap_folder:str, ds:xr.Dataset, video_box_region:tuple, mocap_fps:float = 25, vid_fps:float = 5, frame_width:int = 7000, frame_height:int = 7000, arena_radius:int = 2):
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
    mocap_irs, mocap_blinks = validate_region_of_interest_mocap(df, regions, mocap_arena_center, mocap_arena_r, square_tol = 5, diag_tol = 0.3, max_side_length = 55)

    # Find lights in the video
    vid_irs, _, vid_blinks = find_lights_video(images, 150, video_box_region, calibration_path, square_tol = 5, diag_tol = 0.3, max_side_length = 55, mocap_over_vis_fps = mocap_fps/vid_fps, frame_width = frame_width, frame_height = frame_height, arena_radius = arena_radius)

    # Find spatial transformation matrices
    best = estimate_best_similarity_transform(mocap_irs[:,:2], vid_irs)
    R, s, t = best['R'], best['s'], best['t']
    print("Spatial transform calibration-light RMSE:", best["rmse"])
    print("Spatial transform calibration-light residuals:", best["residuals"])
    print("Spatial transform determinant:", best["det"])
    print("Spatial transform permutation:", best["permutation"])

    # err = evaluate_transform_error(mocap_irs[:,:2], vid_irs, R, s, t)
    # print(err)
    # print(vid_irs)
    # print(mocap_irs[:,:2])
    # print(apply_transform(mocap_irs[:,:2], R, s, t))
    # print(R, s, t)

    fig = plt.figure()
    plt.scatter(vid_irs[:,0], vid_irs[:,1])
    transformed = apply_transform(mocap_irs[:,:2], R, s, t)
    print(vid_irs, mocap_irs[:,:2])
    print(np.linalg.det(R))
    plt.scatter(transformed[:,0], transformed[:,1])
    plt.savefig('mocap_calibration6.png')
    

    # Find time series of blinking visual light
    lag = temporal_synch(mocap_blinks, vid_blinks, df, ds, R, s, t, mocap_batch, mocap_fps = mocap_fps, vid_fps = vid_fps)
    check_transform(ds, df, R, s, t, 50, lag)

    # Check quality of time (and spatial) calibration by plotting a subset of a video frame with MoCap detections overlayed
    vid_frame = 20
    image_index = int(ds['frame'].values[vid_frame])
    assess_calibration(ds, df, R, s, t, vid_frame, lag, images[image_index], calibration_path = calibration_path, frame_width = frame_width, frame_height = frame_height,
                       arena_radius = arena_radius, vid_folder = vid_folder, mocap_fps = mocap_fps, vid_fps = vid_fps)

    # results = compare_arena_scale_from_occupancy(ds, df, R, s, t)
    # print(results)

h5_prep = f'/keypoints/20230329_preprocessed_complete_dewarped_batch_0_5.0Hz.hdf5'
calibration_path = '/full_intrinsics_output/calibration.yaml'
ds = load_preprocessed_data(h5_prep)
full_synchronization('/original/20230329/video/', calibration_path, '/mocap/20230329/', ds, (600, 900, 6300, 6600)) #(6300, 6600, 600, 900))
