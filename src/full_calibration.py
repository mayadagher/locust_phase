'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import datetime

from time_sync import *
from space_sync import apply_transform, find_mocap_to_video_matrices
from dewarping import dewarp_img

'''_____________________________________________________IMPORTS____________________________________________________________'''

def validate_full_calibration(path_to_vid_folder:str, calibration_path:str, path_to_mocap_batch:str, ts_path:str, video_box_region:tuple, mocap_fps:float = 25, vid_fps:float = 5, n_images:int = 10, 
                              pixel_thresh:int = 75, circle_thresh:float = 0.5, square_tol:float = 8, diag_tol:float = 0.5, max_side_length:float = 55, n_hist_bins:int = 300, occupancy_pct:float = 0.02, 
                              n_buffer:int = 3, plot:bool = False, plots_path:str = None, crop_fraction:float = 0.1, offset_s:float = 0):

    # STEP 1: Get transformation matrices for spatial synchronization
    # R, s, t, _, _, _, _ = find_mocap_to_video_matrices(path_to_vid_folder, calibration_path, path_to_mocap_batch, video_box_region, mocap_fps, vid_fps, n_images, pixel_thresh, 
    #                                              circle_thresh, square_tol, diag_tol, max_side_length, n_hist_bins, occupancy_pct, n_buffer, plot, plots_path)

    # R = np.array([[ 0.01042186, -0.99994569], [ 0.99994569,  0.01042186]])
    # s = 0.000969796500032655
    # t = np.array([ 0.0408203,  -0.20256947])

    R = np.array([[ 0.01645639, -0.99986458], [ 0.99986458,  0.01645639]])
    s = 0.0009699278288084318
    t = np.array([ 0.02765381, -0.19051078])

    # STEP 2: Find bounds of overlap between video and MoCap data
    vid_times = get_times_video(path_to_vid_folder, abs_start_frame = 0, abs_end_frame = None)
    mocap_times = get_times_mocap(ts_path, path_to_mocap_batch, rel_start_frame = 0, rel_end_frame = None, mocap_fps = mocap_fps) + datetime.timedelta(seconds = offset_s) # Adjust MoCap as found by time_sync

    start_overlap, end_overlap = get_temporal_overlap(vid_times, mocap_times)

    assert end_overlap.timestamp() - start_overlap.timestamp() > 0, "End of overlap time is before the start of overlap time."

    # STEP 3: Randomly select video frame within overlap range and find the associated MoCap frame
    time_candidates = vid_times[(vid_times >= start_overlap) & (vid_times <= end_overlap)]
    rng = np.random.default_rng()
    vid_t = rng.choice(time_candidates, 1)
    mocap_t_idx, mocap_t_valid = downsample_mocap_for_video(np.array(vid_t), mocap_times, mocap_fps)

    assert len(mocap_t_idx) == 1, f"Inappropriate number of MoCap indices returned while matching MoCap to video frame: {len(mocap_t_idx)}."
    assert mocap_t_valid, f"Selected MoCap index is too far temporally from the selected video timestamp. Assure there is actual overlap between the MoCap batch and the video."

    mocap_t = mocap_times[mocap_t_idx]
    print(f'Selected timestamps for video and MoCap data: {vid_t} and {mocap_t}, respectively.')

    # STEP 4: Plot transformed MoCap positions on top of the (dewarped) video frame
    
    # Find video image index
    vid_t_idx = int(np.where(vid_times == vid_t)[0][0])

    # Load video image and dewarp it
    frame_path = sorted(list(Path(path_to_vid_folder).glob('*.jpg')))[vid_t_idx]
    print(f"\nVIDEO FRAME DEBUG:")
    print(f"  vid_t = {vid_t[0]} (video timestamp)")
    print(f"  vid_t_idx = {vid_t_idx} (frame index)")
    print(f"  frame_path = {frame_path}")
    print(f"  Total video frames available: {len(list(Path(path_to_vid_folder).glob('*.jpg')))}")
    
    img = cv2.imread(frame_path)
    if img is None:
        print(f"  ERROR: Could not read image from {frame_path}")
    else:
        print(f"  Image shape: {img.shape}")
    
    rectified, world_bounds, px_per_m, _, _ = dewarp_img(img, calibration_path)

    # Get actual dewarped image dimensions
    h_dewarped, w_dewarped = rectified.shape[:2]

    # Crop image (in center) to make validation easier by eye
    crop_side = max(2, int(round(min(h_dewarped, w_dewarped) * crop_fraction)))
    crop_side = min(crop_side, h_dewarped, w_dewarped)
    crop_x0 = (w_dewarped - crop_side) // 2
    crop_y0 = (h_dewarped - crop_side) // 2
    rectified = rectified[crop_y0:crop_y0 + crop_side, crop_x0:crop_x0 + crop_side]

    # Transform MoCap coordinates into world units
    mocap_df = pd.read_csv(path_to_mocap_batch)
    mocap_pts_raw = mocap_df[mocap_df["frame"] == (mocap_t_idx[0] + 1)][["x", "y"]].to_numpy(dtype=float) # Add 1 because MoCap frames start at 1
    
    # DEBUG: Print raw mocap coordinates
    if len(mocap_pts_raw) > 0:
        print(f"\nRAW MOCAP COORDINATES (before transform):")
        print(f"  x range: [{mocap_pts_raw[:, 0].min():.1f}, {mocap_pts_raw[:, 0].max():.1f}]")
        print(f"  y range: [{mocap_pts_raw[:, 1].min():.1f}, {mocap_pts_raw[:, 1].max():.1f}]")
        print(f"  Mean: ({mocap_pts_raw[:, 0].mean():.1f}, {mocap_pts_raw[:, 1].mean():.1f})")
        print(f"  Transform params: R=\n{R}\n  s={s:.6f}, t={t}")
    
    mocap_pts = apply_transform(mocap_pts_raw, R, s, t)
    
    # DEBUG: Print transformed mocap coordinates
    if len(mocap_pts) > 0:
        print(f"\nTRANSFORMED MOCAP COORDINATES (after transform, in world meters):")
        print(f"  x range: [{mocap_pts[:, 0].min():.3f}, {mocap_pts[:, 0].max():.3f}]")
        print(f"  y range: [{mocap_pts[:, 1].min():.3f}, {mocap_pts[:, 1].max():.3f}]")
        print(f"  Mean: ({mocap_pts[:, 0].mean():.3f}, {mocap_pts[:, 1].mean():.3f})")

    # Transform MoCap points from world units into full image units
    mocap_pts = np.column_stack([(mocap_pts[:, 0] - world_bounds['xmin']) * px_per_m, (world_bounds['ymax'] - mocap_pts[:, 1]) * px_per_m])

    # Transform MoCap points from full image units to cropped image units
    mocap_pts -= np.array([crop_x0, crop_y0])
    
    # DEBUG: Print final coordinates
    if len(mocap_pts) > 0:
        print(f"\nFINAL COORDINATES DEBUG:")
        print(f"  world_bounds: xmin={world_bounds['xmin']:.3f}, xmax={world_bounds['xmax']:.3f}, ymin={world_bounds['ymin']:.3f}, ymax={world_bounds['ymax']:.3f}")
        print(f"  px_per_m: {px_per_m:.3f}")
        print(f"  Dewarped image shape (h,w): {h_dewarped}, {w_dewarped}")
        print(f"  Cropped image shape: {crop_side} x {crop_side}")
        print(f"  Crop offset: ({crop_x0}, {crop_y0})")
        print(f"  Number of mocap points plotted: {len(mocap_pts)}")
        print(f"  Point pixel coordinate ranges: x=[{mocap_pts[:, 0].min():.1f}, {mocap_pts[:, 0].max():.1f}], y=[{mocap_pts[:, 1].min():.1f}, {mocap_pts[:, 1].max():.1f}]")

    # Initialize plot
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(rectified)
    ax.scatter(mocap_pts[:,0], mocap_pts[:,1], s=3)
    ax.set_xlim([0, crop_side])
    ax.set_ylim([crop_side, 0])
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(f'{plots_path}/full_validation_vid_t_{vid_times[vid_t_idx]}_mocap_t_{mocap_t}.png')
    plt.close(fig)

if __name__ == '__main__':
    path_to_vid_folder = '/original/20230329/video/'
    calibration_path = '/intrinsics/arena_board_calibration/calibration_official.yaml'
    path_to_mocap_batch = '/mocap/20230329/csvs/10K_Marching_0049.csv'
    ts_path = '/mocap/20230329/qtm_capture_times.csv'
    video_box_region = (500, 800, 6400, 6700)
    plots_path = '/output/20230329/kp_plots/calibration/'
    offset_s = 1 # -0.64

    validate_full_calibration(path_to_vid_folder, calibration_path, path_to_mocap_batch, ts_path, video_box_region, plot = True, plots_path = plots_path, offset_s = offset_s)