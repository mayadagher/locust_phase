'''_____________________________________________________IMPORTS____________________________________________________________'''
import numpy as np
import xarray as xr
import cv2
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.optimize import least_squares
from scipy.ndimage import binary_dilation, label
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from data_handling import load_preprocessed_data
from tqdm import tqdm
from dewarping import dewarp_img_sequence
import itertools
from skimage import measure
from scipy.stats import binned_statistic_2d

'''_____________________________________________________TRANSFORM FUNCTIONS____________________________________________________________'''

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
        candidate = {"R": R,
                     "s": s,
                     "t": t,
                     "rmse": rmse,
                     "permutation": permutation,
                     "det": float(np.linalg.det(R)),
                     "predicted": predicted,
                     "target": target,
                     "residuals": residuals}
        
        if best is None or candidate["rmse"] < best["rmse"]:
            best = candidate

    return best

'''_____________________________________________________LIGHT FINDING FUNCTIONS____________________________________________________________'''

def sort_pts(pts):
    ''' Match lights across frames by sorting consistently (by x then y) so that averaging is done over the same physical light each time.'''
    pts = np.asarray(pts)
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[idx]

def is_square_corners(pts:np.ndarray, square_tol:float, diag_tol:float = 0.1, max_side_length:float = np.inf) -> bool:
    '''Checks whether 3 or 4 input points form corners of a square (two equal sides, correct diagonal).
    max_side_length defines the maximum allowable side length of the square.'''
    
    n = len(pts)

    assert n in [3, 4], f"Number of points ({n}) invalid; must be 3 or 4."

    # Compute and order distances by size
    dists = pdist(pts)
    d = np.sort(dists)

    # Check if first distances are equal (shortest distances)
    sides_idx = 2*(n//2)
    sides_equal = np.sum(np.abs(d[:sides_idx] - d[:sides_idx,np.newaxis]) < square_tol) == sides_idx**2

    # Check if the average side length is less than the maximum allowed
    if d[:sides_idx].mean() > max_side_length:
        return False
    
    # Check if last distances are proper diagonals
    diags_ok = np.sum(np.abs(d[sides_idx:] - np.sqrt(2)*d[:sides_idx].mean()) < diag_tol) == n//2

    return sides_equal and diags_ok

def find_lights_video(images:list[np.ndarray], world_bounds:dict, px_per_m:float, warp_matrix:np.ndarray, video_box_region:tuple, pixel_thresh:int = 75, circle_thresh = 0.3, 
                      square_tol:float = 5, diag_tol:float = 0.5, max_side_length:float = 55, mocap_fps:float = 25, vid_fps:float = 5, plot:bool = False, plots_path:str = None):
    ''' Find positions of IR lights in dewarped video coordinates, as well as the frames in which they are on.'''

    # STEP 1: Transform video_box_region from undistorted image pixel coordinates to output image pixel coordinates using the warp matrix.
    y0, y1, x0, x1 = video_box_region
    corners = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x0, y1, 1.0], [x1, y1, 1.0]], dtype=float).T # Added 1 row at the bottom because warp_matrix is 3x3
    mapped = warp_matrix @ corners
    mapped = (mapped[:2, :] / mapped[2, :]).T  # (4,2) as (x', y') in rectified image pixels -> bottom row is a scaling factor
    x_coords, y_coords = mapped[:, 0], mapped[:, 1]

    # Turn coordinates into integer pixel indices
    rx0, rx1 = int(np.floor(np.min(x_coords))), int(np.ceil(np.max(x_coords)))
    ry0, ry1 = int(np.floor(np.min(y_coords))), int(np.ceil(np.max(y_coords)))

    # Clip to rectified image bounds (use first dewarped image as reference)
    h_ref, w_ref = images[0].shape[:2]
    rx0, rx1 = max(0, min(rx0, w_ref - 1)), max(0, min(rx1, w_ref))
    ry0, ry1 = max(0, min(ry0, h_ref - 1)), max(0, min(ry1, h_ref))

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

    # STEP 5: Optional -> visualize the detected lights on the first image for verification
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
    return ir_lights_arena, vis_light_arena[0], vis_on_frames # vis_on_frames is in video fps

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

    # Load MoCap data and filter out invalid points
    x = mocap_df["x"].to_numpy()
    y = mocap_df["y"].to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # Find extrema to define the bin edges for the 2D histogram, assuring dx = dy
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    span = max(xmax - xmin, ymax - ymin)

    # Get center of x and y
    cx0 = (xmin + xmax) / 2
    cy0 = (ymin + ymax) / 2
    half = span / 2

    # Define bin edges so that bins are square and cover the entire range of x and y
    x_edges = np.linspace(cx0 - half, cx0 + half, n_hist_bins + 1)
    y_edges = np.linspace(cy0 - half, cy0 + half, n_hist_bins + 1)

    # Make heatmap and binarize it
    heatmap, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
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
    for (frame_val, frame_df) in tqdm(mocap_df.groupby('frame'), 'Checking MoCap frames for square formations.'):

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
                blink_frames[r_idx][frame_val - 1] = True # Frame indices in MoCap file start at 1

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

def find_lights_mocap(mocap_df:pd.DataFrame, n_hist_bins:int = 300, occupancy_pct:float = 0.02, n_buffer:int = 3, square_tol:float = 5, diag_tol:float = 0.3,
                      plot:bool = False, plots_path:str = None):
    '''Find MoCap IR positions and blinks while loading only one CSV at a time.'''

    # STEP 1: Find candidate regions of interest in the MoCap data
    regions, mocap_center, mocap_radius = find_regions_of_interest_mocap(mocap_df, n_hist_bins, occupancy_pct, n_buffer, plot, plots_path)

    # STEP 2: Validate candidate regions by looking for square light formations and scoring them
    mocap_irs, mocap_blinks = validate_region_of_interest_mocap(mocap_df, regions, square_tol, diag_tol, plot = plot, plots_path = plots_path)

    return mocap_irs, mocap_blinks, mocap_center, mocap_radius # mocap_blinks is in MoCap fps

def estimate_transform_from_arena(mocap_irs, video_irs, mocap_center, video_center):
    """
    Estimate a globally conditioned MoCap -> video-world transform.

    Scale comes from arena radius.
    Translation comes from arena center.
    Rotation comes from the long-baseline vector from arena center
    to the IR-cluster centroid.

    Assumes a proper rotation (det(R) = +1).
    """

    mocap_irs = np.asarray(mocap_irs, dtype=float)[:, :2]
    video_irs = np.asarray(video_irs, dtype=float)

    mocap_center = np.asarray(mocap_center, dtype=float)
    video_center = np.asarray(video_center, dtype=float)

    # 1. SCALE: physical arena radius provides a long-baseline estimate
    # s = float(video_radius / mocap_radius)

    # 1. SCALE: distance from arena center to IR centroid provides a long-baseline estimate (more robust than arena radii, which have sampling noise)
    s = float(np.linalg.norm(video_irs.mean(axis=0) - video_center) / np.linalg.norm(mocap_irs.mean(axis=0) - mocap_center))

    # 2. ROTATION: arena center -> IR centroid gives a ~metre-scale vector
    v_m = mocap_irs.mean(axis=0) - mocap_center
    v_v = video_irs.mean(axis=0) - video_center

    theta_m = np.arctan2(v_m[1], v_m[0])
    theta_v = np.arctan2(v_v[1], v_v[0])

    theta = theta_v - theta_m

    c = np.cos(theta)
    ss = np.sin(theta)

    R = np.array([
        [c, -ss],
        [ss,  c],
    ])

    # 3. TRANSLATION: force arena centers to coincide
    t = video_center - s * (R @ mocap_center)

    # 4. Validate against the IRs
    predicted_irs = apply_transform(
        mocap_irs,
        R,
        s,
        t,
    )

    # Since the IR order may differ, evaluate best correspondence
    best_rmse = np.inf
    best_perm = None

    for perm in itertools.permutations(range(len(video_irs))):
        target = video_irs[list(perm)]

        rmse = np.sqrt(np.mean(np.sum((predicted_irs - target) ** 2, axis=1)))

        if rmse < best_rmse:
            best_rmse = rmse
            best_perm = perm

    return {
        "R": R,
        "s": s,
        "t": t,
        "theta_deg": np.degrees(theta),
        "ir_rmse": best_rmse,
        "ir_permutation": best_perm,
        "predicted_irs": predicted_irs,
    }

def check_mocap_tilt(mocap_df:pd.DataFrame, plots_path:str, n_hist_bins:int = 100, q:float = 0.1):
    ''' Check for tilt in the MoCap data by making a 2D histogram of the median z-coordinates in each bin.'''

    # Get valid coordinate values (in original MoCap coordinate system)
    x = mocap_df["x"].to_numpy()
    y = mocap_df["y"].to_numpy()
    z = mocap_df["z"].to_numpy()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid]
    y = y[valid]
    z = z[valid]

    # Find extrema of x and y to define the bin edges for the 2D histogram, assuring dx = dy
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    span = max(xmax - xmin, ymax - ymin)

    # Get center of x and y
    cx0 = (xmin + xmax) / 2
    cy0 = (ymin + ymax) / 2
    half = span / 2

    # Define bin edges so that bins are square and cover the entire range of x and y
    x_edges = np.linspace(cx0 - half, cx0 + half, n_hist_bins + 1)
    y_edges = np.linspace(cy0 - half, cy0 + half, n_hist_bins + 1)

    def get_quant(x):
        return np.quantile(x, q)

    # Quantized ln absolute z value in each x-y bin
    q_ln_z, x_edge, y_edge, _ = binned_statistic_2d(x, y, np.log(np.abs(z)), statistic=get_quant, bins=[x_edges, y_edges])

    # Initiate figure
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

    # Transpose because pcolormesh expects the array orientation this way
    mesh = ax[0].pcolormesh(x_edge, y_edge, q_ln_z.T, shading="auto", cmap="viridis")

    ax[0].set_xlabel('x (MoCap coordinates)')
    ax[0].set_ylabel('y (MoCap coordinates)')

    # Estimate tilt by fitting a plane to the z values in the bins
    q_z, _, _, _ = binned_statistic_2d(x, y, z, statistic=get_quant, bins=[x_edges, y_edges])

    def plane(params, x, y):
        a, b, c = params
        return a * x + b * y + c

    params0 = [0, 0, 0]  # Initial guess for plane parameters
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xs, ys = np.meshgrid(x_centers, y_centers)
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()
    z_flat = q_z.flatten()

    # Only use bins with valid z values for fitting
    valid_bins = np.isfinite(z_flat)
    result = least_squares(lambda params: plane(params, xs_flat[valid_bins], ys_flat[valid_bins]) - z_flat[valid_bins], x0=params0, method="lm")
    a, b, c = result.x

    mesh2 = ax[1].pcolormesh(x_edge, y_edge, np.log(np.abs(plane(result.x, xs, ys))).T, shading="auto", cmap="viridis", vmin=np.nanmin(q_ln_z), vmax=np.nanmax(q_ln_z))

    plt.colorbar(mesh2, label=f'q{q*100:.0f} ln|z| (MoCap coordinates)')
    plt.savefig(f'{plots_path}mocap_tilt_heatmap.png')

    # Estimate tilt angle in degrees
    tilt_angle = np.degrees(np.arctan(np.sqrt(a**2 + b**2)))
    print('Estimated tilt angle of MoCap data: {:.2f} degrees'.format(tilt_angle))
    print('Constant: {:.2f}'.format(c))

def check_mocap_tilt(mocap_df: pd.DataFrame, plots_path: str, n_hist_bins: int = 100, q: float = 0.5, min_bin_count: int = 20):
    """
    Estimate tilt in MoCap coordinates using a low quantile of z within
    spatial x-y bins.

    The fitted plane is:

        z = a*x + b*y + c

    Returns
    -------
    dict
        Plane coefficients, plane normal, and estimated tilt angle.
    """

    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1.")

    # ------------------------------------------------------------
    # STEP 1: Get finite coordinates
    # ------------------------------------------------------------

    x = mocap_df["x"].to_numpy(dtype=float)
    y = mocap_df["y"].to_numpy(dtype=float)
    z = mocap_df["z"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > - 10) # Exclude largely negative data, which are artefacts from the system

    x = x[valid]
    y = y[valid]
    z = z[valid]

    # ------------------------------------------------------------
    # STEP 2: Make square spatial bins
    # ------------------------------------------------------------

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    span = max(xmax - xmin, ymax - ymin)

    cx0 = (xmin + xmax) / 2
    cy0 = (ymin + ymax) / 2
    half = span / 2

    x_edges = np.linspace(cx0 - half, cx0 + half, n_hist_bins + 1)
    y_edges = np.linspace(cy0 - half, cy0 + half, n_hist_bins + 1)

    def get_quant(values):
        if len(values) == 0:
            return np.nan
        return np.quantile(values, q)

    # ------------------------------------------------------------
    # STEP 3: Calculate signed z quantile for plane fitting
    # ------------------------------------------------------------

    q_z, _, _, _ = binned_statistic_2d(x, y, z, statistic=get_quant, bins=[x_edges, y_edges])

    # Number of observations per spatial bin
    counts, _, _, _ = binned_statistic_2d(x, y, z, statistic="count", bins=[x_edges, y_edges])

    # ------------------------------------------------------------
    # STEP 4: Construct correctly indexed bin-center coordinates
    # ------------------------------------------------------------

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # IMPORTANT:
    # q_z has shape (n_x_bins, n_y_bins), so use indexing="ij".
    xs, ys = np.meshgrid(x_centers, y_centers, indexing="ij")

    # Only fit sufficiently populated, finite bins.
    valid_bins = (np.isfinite(q_z) & (counts >= min_bin_count))

    x_fit = xs[valid_bins]
    y_fit = ys[valid_bins]
    z_fit = q_z[valid_bins]

    if len(z_fit) < 3:
        raise ValueError("Too few valid spatial bins to fit a plane.")

    # ------------------------------------------------------------
    # STEP 5: Fit z = a*x + b*y + c
    # ------------------------------------------------------------
    #
    # This problem is linear, so least_squares() is unnecessary.
    # ------------------------------------------------------------

    A = np.column_stack([x_fit, y_fit, np.ones(len(x_fit))])

    params, _, _, _ = np.linalg.lstsq(A, z_fit, rcond=None)

    a, b, c = params

    fitted_plane = (a * xs + b * ys + c)

    # ------------------------------------------------------------
    # STEP 6: Calculate tilt
    # ------------------------------------------------------------

    slope_magnitude = np.sqrt(a**2 + b**2)

    tilt_angle = np.degrees(np.arctan(slope_magnitude))

    normal = np.array([-a, -b, 1.0])

    normal /= np.linalg.norm(normal)

    # ------------------------------------------------------------
    # STEP 7: Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Robust shared colour limits, excluding extreme bad-z bins
    combined = np.concatenate([q_z[np.isfinite(q_z)], fitted_plane[np.isfinite(fitted_plane)]])

    vmin, vmax = np.quantile(combined, [0.01, 0.99])

    ax[0].pcolormesh(x_edges, y_edges, q_z.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)

    mesh2 = ax[1].pcolormesh(x_edges, y_edges, fitted_plane.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)

    ax[0].set_title(f"Observed q{100*q:.0f} z")
    ax[1].set_title(f"Fitted plane\n" f"tilt = {tilt_angle:.2f}°")

    for axis in ax:
        axis.set_xlabel("x (MoCap coordinates)")
        axis.set_aspect("equal")

    ax[0].set_ylabel("y (MoCap coordinates)")

    fig.colorbar(mesh2,ax=ax[1], label=f"q{100*q:.0f} z " "(MoCap coordinates)",)

    fig.tight_layout()

    fig.savefig(f"{plots_path}mocap_tilt_heatmap.png", dpi=200)

    plt.close(fig)

    print(f"Estimated tilt angle of MoCap data: " f"{tilt_angle:.2f} degrees")

    print(f"Plane: z = {a:.6g} x " f"+ {b:.6g} y " f"+ {c:.6g}")

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "normal": normal,
        "tilt_deg": float(tilt_angle),
        "q_z": q_z,
        "counts": counts,
        "fitted_plane": fitted_plane,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }

'''_____________________________________________________WRAPPER FUNCTIONS____________________________________________________________'''

def find_mocap_to_video_matrices(path_to_vid_folder:str, calibration_path:str, path_to_mocap_batch:str, video_box_region:tuple, mocap_fps:float = 25, vid_fps:float = 5, 
                                 n_images:int = 10, pixel_thresh:int = 75, circle_thresh:float = 0.5, square_tol:float = 8, diag_tol:float = 0.5, max_side_length:float = 55, 
                                 n_hist_bins:int = 300, occupancy_pct:float = 0.02, n_buffer:int = 3, plot:bool = False, plots_path:str = None):
    '''Find the transformation matrices that map MoCap coordinates to video coordinates.'''

    # STEP 1: Load dewarped video frames and calibration parameters
    images, world_bounds, px_per_m, warp_matrix, _ = dewarp_img_sequence(path_to_vid_folder, calibration_path, end = n_images)
    
    # STEP 2: Find position of IR lights in the video frames and the frames in which they are "on"
    vid_irs, _, vid_blinks = find_lights_video(images, world_bounds, px_per_m, warp_matrix, video_box_region, pixel_thresh = pixel_thresh, circle_thresh = circle_thresh, 
                                               square_tol = square_tol, diag_tol = diag_tol, max_side_length = max_side_length, mocap_fps = mocap_fps, vid_fps = vid_fps, 
                                               plot = True, plots_path = plots_path)

    # STEP 3: Load a single batch of the MoCap data and find the IR light positions and blinks
    mocap_df = pd.read_csv(path_to_mocap_batch)
    mocap_irs, mocap_blinks, mocap_center, mocap_radius = find_lights_mocap(mocap_df, n_hist_bins, occupancy_pct, n_buffer, square_tol, diag_tol, plot, plots_path)

    # STEP 4: Estimate the similarity transform that maps MoCap IR positions to video IR positions
    best_transform = estimate_best_similarity_transform(mocap_irs[:, :2], vid_irs[:, :2], allow_reflection=True)
    R, s, t = best_transform["R"], best_transform["s"], best_transform["t"]
    print(f"Estimated similarity transform:\nR =\n{R}\ns = {s}\nt = {t}\nRMSE = {best_transform['rmse']:.4f}\nDet(R) = {best_transform['det']:.4f}")

    return R, s, t, vid_irs, mocap_irs, vid_blinks, mocap_blinks, mocap_center, mocap_radius

def compare_irs(path_to_vid_folder:str, calibration_path:str, R:np.ndarray, s:np.ndarray, t:np.ndarray, vid_irs:np.ndarray, mocap_irs:np.ndarray):
    '''Plot undistorted video IRs and transformed MoCap IRs on top of undistorted image.'''

    # STEP 1: Undistort first image from image folder
    img, world_bounds, px_per_m, warp_matrix, world_to_output = dewarp_img_sequence(path_to_vid_folder, calibration_path, start = 0, end = 1)

    # STEP 2: Transform video points into output points
    vid_irs_3d = np.column_stack([vid_irs[:,0], vid_irs[:,1], np.ones(vid_irs.shape[0])])
    vid_irs = (world_to_output @ vid_irs_3d.T).T[:,:2]

    # STEP 3: Transform MoCap points into (video) world coordinates
    mocap_irs = apply_transform(mocap_irs[:,:2], R, s, t)

    # STEP 4: Transform MoCap points into output points
    mocap_irs_3d = np.column_stack([mocap_irs[:,0], mocap_irs[:,1], np.ones(mocap_irs.shape[0])])
    mocap_irs = (world_to_output @ mocap_irs_3d.T).T[:,:2]

    # STEP 5: Plot video and MoCap points on image

    fig, ax = plt.subplots()
    ax.imshow(img[0])
    ax.scatter(vid_irs[:,0], vid_irs[:,1], c = 'blue', s = 5, alpha = 0.3, label = 'Video')
    ax.scatter(mocap_irs[:,0], mocap_irs[:,1], c = 'orange', s = 5, alpha = 0.3, label = 'MoCap')
    ax.legend()

    # Find point extrema
    buffer = 50
    xmin, xmax = min(np.min(vid_irs[:,0]), np.min(mocap_irs[:,0])), max(np.max(vid_irs[:,0]), np.max(mocap_irs[:,0]))
    ymin, ymax = min(np.min(vid_irs[:,1]), np.min(mocap_irs[:,1])), max(np.max(vid_irs[:,1]), np.max(mocap_irs[:,1]))
    ax.set_xlim([xmin - buffer, xmax + buffer])
    ax.set_ylim([ymax + buffer, ymin - buffer])
    plt.savefig(f'{plots_path}ir_transform_validation.png')
    plt.close(fig)


def compare_heatmaps(ds:xr.Dataset, mocap_df:pd.DataFrame, R:np.ndarray, s:np.ndarray, t:np.ndarray, plots_path:str | None = None, bin_width:float=0.1, occupancy_fraction:float=0.01):
    """Validate the estimated transform by comparing MoCap and video coordinates."""

    # STEP 1: Apply the transform to MoCap points
    mocap_points = mocap_df[['x', 'y']].values
    del mocap_df

    transformed_points = apply_transform(mocap_points, R, s, t)
    del mocap_points

    # STEP 2: Find extrema of datasets to make bin edges where dx = dy
    xmin = min(transformed_points[:,0].min(), ds.centroid_x.min())
    xmax = max(transformed_points[:,0].max(), ds.centroid_x.max())
    ymin = min(transformed_points[:,1].min(), ds.centroid_y.min())
    ymax = max(transformed_points[:,1].max(), ds.centroid_y.max())
    x_edges = np.arange(xmin, xmax, bin_width)
    y_edges = np.arange(ymin, ymax, bin_width)


    # STEP 3: Create a 2D histogram of transformed MoCap points and video points
    H_mocap, _, _ = np.histogram2d(transformed_points[:, 0], transformed_points[:, 1], bins = [x_edges, y_edges])
    del transformed_points
    H_video, _, _ = np.histogram2d(ds.centroid_x.values.ravel(), ds.centroid_y.values.ravel(), bins=[x_edges, y_edges])
    del ds
    x_centers = (x_edges[:-1] + x_edges[1:])/2
    y_centers = (y_edges[:-1] + y_edges[1:])/2

    # STEP 4: Binarize histograms using occupancy_fraction threshold
    H_mocap_bin = (H_mocap.T > max(1, np.max(H_mocap) * occupancy_fraction)).astype(int)
    H_video_bin = (H_video.T > max(1, np.max(H_video) * occupancy_fraction)).astype(int)


    # STEP 5: Fit circles to both binarized histograms
    cx_mocap_bin, cy_mocap_bin, r_mocap_bin = fit_circle_to_binary_heatmap(np.pad(H_mocap_bin, 1, mode="constant", constant_values=False))
    cx_video_bin, cy_video_bin, r_video_bin = fit_circle_to_binary_heatmap(np.pad(H_video_bin, 1, mode="constant", constant_values=False))

    # STEP 6: Transform arena parameters to world units
    bin_size = x_edges[1] - x_edges[0]
    arena_params = [] # Video, MoCap
    for (cx_bin, cy_bin, r_bin) in [(cx_video_bin, cy_video_bin, r_video_bin), (cx_mocap_bin, cy_mocap_bin, r_mocap_bin)]:
        center = np.array([np.interp(cx_bin - 1, np.arange(len(x_edges) - 1), x_centers), np.interp(cy_bin - 1, np.arange(len(y_edges) - 1), y_centers)])
        radius = r_bin*bin_size
        arena_params.append([*center, radius])

    print(f'Video arena parameters: center = ({arena_params[0][0]:.2f}, {arena_params[0][1]:.2f}), r = {arena_params[0][2]:.2f}')
    print(f'Transformed MoCap arena parameters: center = ({arena_params[1][0]:.2f}, {arena_params[1][1]:.2f}), r = {arena_params[1][2]:.2f}')

    # STEP 7: Optional -> visualize the detected regions on the heatmap for verification
    if plots_path is not None:
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6), sharex=True, sharey=True)

        # Prepare for iterating
        hmaps = [H_video, H_mocap]

        for i, hmap in enumerate(hmaps):

            # Plot heatmaps
            img = ax[i].imshow(np.log1p(hmap.T), origin = 'lower', cmap = 'viridis', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

            # Plot colour bars
            fig.colorbar(img, ax=ax[i], fraction=0.046, pad=0.04, label="log(1 + count)")

            # Plot arena centers
            ax[i].scatter(arena_params[i][0], arena_params[i][1], marker = '+', color = 'white')

            # Plot arena boundaries
            ax[i].add_patch(plt.Circle((arena_params[i][0], arena_params[i][1]), arena_params[i][2],  fill = False, color = "white", linewidth = 2))

            # Set aspect ratio
            ax[i].set_aspect('equal')

        # Add titles and subtitles
        fig.suptitle("Spatial occupancy heatmaps")
        ax[0].set_title(f"Video\nCenter = ({arena_params[0][0]:.2f}, {arena_params[0][1]:.2f}), r = {arena_params[0][2]:.2f}")
        ax[1].set_title(f"Transformed MoCap\nCenter = ({arena_params[1][0]:.2f}, {arena_params[1][1]:.2f}), r = {arena_params[1][2]:.2f}")

        fig.tight_layout()
        # fig.savefig(f'{plots_path}heatmap_transform_validation.png')
        fig.savefig(f'{plots_path}heatmap_untransformed_validation.png')
        plt.close(fig)

        return arena_params

if __name__ == '__main__':
    plots_path = '/output/20230329/kp_plots/calibration/'
    calibration_path = '/intrinsics/arena_board_calibration/calibration_official.yaml'
    h5_prep = f'/keypoints/dewarped/20230329_preprocessed_complete_dewarped_batch_0_5.0Hz.hdf5'
    # ds = load_preprocessed_data(h5_prep)
    # R_ir, s_ir, t_ir, vid_irs, mocap_irs, _, _, mocap_center, mocap_r = find_mocap_to_video_matrices('/original/20230329/video/', calibration_path, '/mocap/20230329/csvs/10K_Marching_0049.csv', (500, 800, 6400, 6700), plot = True, plots_path = plots_path)
    mocap_df = pd.read_csv('/mocap/20230329/csvs/10K_Marching_0049.csv')
    check_mocap_tilt(mocap_df, plots_path, n_hist_bins = 100)

    # vid_center = np.array([0.01, 0.00]) # From define_boundary
    # vid_r = 2.38 # From define_boundary



    # results = estimate_transform_from_arena(mocap_irs, vid_irs, mocap_center, vid_center)

    # R = results["R"]
    # s = results["s"]
    # t = results["t"]

    # print('\nChecking rotations.')
    # print('Rotation by vectors:', np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    # print('Rotation by IR lights:', np.degrees(np.arctan2(R_ir[1, 0], R_ir[0, 0])))

    # print('\nChecking scales.')
    # print('Ratio of vectors:', s)
    # dm = np.sort(pdist(mocap_irs[:, :2]))
    # dv = np.sort(pdist(vid_irs))
    # print('Ratio of IR distances:', dv/dm)
    # print('Median ratio of IR distances:', np.median(dv/dm))

    # print('\nChecking translations.')
    # print('Translation by vectors:', t)
    # print('Translation by IR lights:', t_ir)

    # print(R, s, t)

    # compare_irs('/original/20230329/video/', calibration_path, R, s, t, vid_irs, mocap_irs)
    # compare_heatmaps(ds, mocap_df, R, s, t, plots_path, bin_width = 0.3)