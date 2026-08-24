'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from pathlib import Path
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from locust_arena_calibration.cli import define_arena
from locust_arena_calibration.models import CalibrationBundle
from locust_arena_calibration.transforms import scale_calibration_bundle, image_points_to_arena, undistort_points

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def dewarp_pts(points:np.ndarray, calibration_path:str, frame_width:int = 7000, frame_height:int = 7000):
    """
    Dewarp points using calibration data.
    """
    bundle = CalibrationBundle.load_yaml(Path(calibration_path))
    if frame_width is not None and frame_height is not None:
        bundle = scale_calibration_bundle(bundle, frame_width, frame_height, allow_resolution_scaling=True)

    # Convert image points to arena coordinates
    points_dewarped = image_points_to_arena(points, bundle)

    return points_dewarped

def dewarp_img(img:np.ndarray, calibration_path:str, frame_width:int = 7000, frame_height:int = 7000, arena_radius:float = 4.781/2):
    ''' Dewarp an image and return matrices needed to plot dewarped points on top of the dewarped image.'''

    # STEP 1: Load calibration bundle and scale it to the desired frame width and height
    bundle = CalibrationBundle.load_yaml(Path(calibration_path))
    if frame_width is not None and frame_height is not None:
        bundle = scale_calibration_bundle(bundle, frame_width, frame_height, allow_resolution_scaling=True)

    # STEP 2: Undistort the image using the camera matrix and distortion coefficients from the calibration bundle
    camera_matrix = bundle.camera_matrix
    distortion_coeffs = bundle.distortion_coefficients
    undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs, None, camera_matrix)

    # STEP 3: Map the image corners (pixel coordinates) into arena/world coordinates
    corners_px_xy = np.array([[0.0, 0.0],
                              [float(frame_width), 0.0],
                              [float(frame_width), float(frame_height)],
                              [0.0, float(frame_height)]], 
                              dtype=float)

    # Undistort the corners in pixel coordinates
    undistorted_corners = undistort_points(corners_px_xy, camera_matrix, distortion_coeffs)

    # Homography: convert the (undistorted) pixel coordinates to arena/world coordinates
    H_px_to_m = np.array(bundle.homography_px_to_m, dtype=float)
    homogeneous = np.column_stack([undistorted_corners, np.ones(undistorted_corners.shape[0], dtype=float)]).T # Add homogeneous scaling column for matrix multiplication
    corners_world_h = (H_px_to_m @ homogeneous).T  # (4,3)
    corners_world = corners_world_h[:, :2] / corners_world_h[:, 2:3]

    # STEP 4: Determine the bounds of the rectified image in world coordinates

    # Determine bounds that cover both the imaged region and the arena circle
    xmin_c, xmax_c = float(corners_world[:, 0].min()), float(corners_world[:, 0].max())
    ymin_c, ymax_c = float(corners_world[:, 1].min()), float(corners_world[:, 1].max())

    # Add a small margin around the arena so the rectified image is not tightly cropped.
    pad_m = 0.2
    xmin = min(xmin_c, -arena_radius - pad_m)
    xmax = max(xmax_c, arena_radius + pad_m)
    ymin = min(ymin_c, -arena_radius - pad_m)
    ymax = max(ymax_c, arena_radius + pad_m)

    world_bounds = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

    # STEP 5: Determine the scale factor from world coordinates to pixel coordinates for the rectified image

    # Use homography as an estimate of the scale factor, since it is a linear transformation
    H = np.array(bundle.to_dict()["H_undistorted_px_to_arena_m"], dtype=float)
    H_inv = np.linalg.inv(H)
    A = H_inv[:2, :2]
    px_per_m = float(np.mean(np.linalg.svd(A, compute_uv=False)))

    # Set the height and width of the rectified image
    output_width = int(np.ceil((world_bounds["xmax"] - world_bounds["xmin"]) * px_per_m))
    output_height = int(np.ceil((world_bounds["ymax"] - world_bounds["ymin"]) * px_per_m))

    # Determine the transformation directly from world coordinates to output pixel coordinates
    world_to_output = np.array([[px_per_m, 0.0, -world_bounds["xmin"] * px_per_m],
                                [0.0, -px_per_m, world_bounds["ymax"] * px_per_m],
                                [0.0, 0.0, 1.0]], 
                                dtype=float)

    # Compute the warp matrix and apply the perspective transformation to the image
    warp_matrix = world_to_output @ bundle.homography_px_to_m # Maps input pixels to output pixels
    rectified = cv2.warpPerspective(undistorted_img, warp_matrix, (output_width, output_height))

    return rectified, world_bounds, px_per_m, warp_matrix, world_to_output

def dewarp_img_sequence(img_dir:str, calibration_path:str, start:int = 0, end:int = None, frame_width:int = 7000, frame_height:int = 7000, arena_radius:float = 4.781/2):
    '''Load and dewarp a sequence of images from a directory. Returns a list of dewarped images. start and end refer to the indices of the images in the directory.'''

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg'))])
    if end is None:
        end = len(img_files)
    img_files = img_files[start:end]

    rectified_images = []
    for img_file in tqdm(img_files, "De-warping images"):
        img = cv2.imread(os.path.join(img_dir, img_file))
        rectified, world_bounds, px_per_m, warp_matrix, _ = dewarp_img(img, calibration_path, frame_width, frame_height, arena_radius)
        rectified_images.append(rectified)

    return rectified_images, world_bounds, px_per_m, warp_matrix

def validate_dewarp(calibration_path:str, img_dir:str, ds:xr.Dataset, plot_dir:str, rel_idx:int = 0, frame_width:int = 7000, frame_height:int = 7000, arena_radius:float = 4.781/2):
    """
    Validate the dewarping of points and images using calibration data. ds should already be dewarped.
    """

    # Get index of the frame to validate
    abs_frame_idx = ds.frame.values[rel_idx]

    # Get sorted list of images
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg'))])
    img = cv2.imread(os.path.join(img_dir, img_files[abs_frame_idx]))

    # Dewarp the image
    rectified, _, _, _, world_to_output = dewarp_img(img, calibration_path, frame_width, frame_height, arena_radius)

    # Apply transform, preserving NaNs (matrix mult with NaN yields NaN)
    pts_world = np.column_stack([ds.centroid_x.values[rel_idx, :], ds.centroid_y.values[rel_idx, :], np.ones(ds.centroid_x.values[rel_idx, :].shape[0])])
    pts_out = (world_to_output @ pts_world.T).T
    x_px = pts_out[:, 0]
    y_px = pts_out[:, 1]

    # Plot points on top of the rectified image
    plt.imshow(rectified)
    plt.scatter(x_px, y_px, c='r', s=0.5, alpha=0.3)
    plt.savefig(f'{plot_dir}dewarp_abs_frame_{abs_frame_idx}.png')
    print(f'Dewarp validation plot saved to {plot_dir}dewarp_abs_frame_{abs_frame_idx}.png')

    return