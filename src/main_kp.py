'''_____________________________________________________IMPORTS____________________________________________________________'''

# from ultralytics import YOLO
from data_handling import *
from keypoints import *
from phase import *
from visualize_phase import *
from animate import *
from cluster_analysis import *

import numpy as np
import cv2
from helper_fns import *
import yaml
from locust_arena_calibration.transforms import rectify_image_to_arena, scale_calibration_bundle
from dewarping import validate_dewarp


'''_____________________________________________________PARAMETERS____________________________________________________________'''

# YOLO training parameters
workspace = "locustorientations"
project_name = "locusts_2kp"
data_version = 8
yolo_short_name = "yolo26"
yolo_version = 'yolo26n-pose.pt'
img_sz = 640

# Keypoints path
h5_kp='/keypoints/20230329_processed_complete.hdf5'
n_kp_batches = 5
total_frames = 60466
batch_len = np.ceil(total_frames/n_kp_batches).astype(int)

# Entropy params
n_layers = 6
n_ang0 = 16
r_max = 200
n_focals = 1000
occ_path = f'/keypoints/20230329_{n_layers}_{n_ang0}_{r_max}_{n_focals}.hdf5'

# Arena params (found using define_boundary.py)
arena_center =  np.array([3527.08, 3520.27]) # px, found using define_boundary
arena_radius = 3395.79 # px, found using define_boundary

# Loading parameters
exp_name = '20230329'
batch_idx = 0
subsample = 1

# Visualizing and animating parameters
vid_path = '/original/20230329/20230329.mp4'
img_dir = '/original/20230329/video/'

# Calibration bundle parameters
intrinsics_file = '/intrinsics/arena_board_calibration/homography_official.yaml'

# Undistortion parameters
# calibration_path = '/intrinsics/full_intrinsics_output/calibration.yaml'
calibration_path = '/intrinsics/arena_board_calibration/calibration_official.yaml'
frame_width = 7000
frame_height = 7000

# Saving params
plots_path = '/output/' + exp_name + '/kp_plots/'
'''_____________________________________________________RUN CODE____________________________________________________________'''

if __name__ == "__main__":

    ''' STEP 1: KEYPOINT ACQUISITION '''

    # GENERATE TILES FOR TRAINING KEYPOINTS FROM VIDEO
    # extract_random_tiles(image_dir=img_dir, output_dir="/keypoints/unlabeled_tiles", n_tiles=50, tile_size=320)

    # TRAIN MODEL
    # train_model(workspace, project_name, data_version, yolo_short_name, yolo_version, epochs = 1, img_sz = img_sz)

    # GET KEYPOINTS FOR WHOLE VIDEO
    # slice_folder_to_h5(path_to_model = '/keypoints/best_kp_weights.pt', frames_dir = img_dir, h5_in = '/keypoints/20230329_unprocessed_kps.hdf5', start_idx = 0, stop_idx = 20000, chunk_size = 200, tile_size = 640, img_size = 7000, overlap = 0.3)
   
    # PREPROCESS KEYPOINTS TO GET RID OF DUPLICATES AND FALSE DETECTIONS ACROSS TILES
    # preprocess_kps_fast(h5_in='/keypoints/20230329_unprocessed_complete.hdf5')

    ''' STEP 2: DEWARP KEYPOINTS'''

    # DEWARP KEYPOINTS
    # ds = kp_detections_to_xr(h5_kp, calibration_path, frame_width=frame_width, frame_height=frame_height, start_frame=batch_idx*batch_len, end_frame=(batch_idx+1)*batch_len, subsample=subsample)
    
    ds = load_preprocessed_data(f'/keypoints/dewarped/20230329_preprocessed_complete_dewarped_batch_{batch_idx}_{round(5/subsample, 2)}Hz.hdf5')

    # VALIDATE DEWARPING OF KEYPOINTS AND IMAGES
    validate_dewarp(calibration_path, img_dir, ds, plots_path, rel_idx = 0, frame_width = frame_width, frame_height = frame_height)

    # SAVE DEWARPED DATASET
    # save_ds(ds, f'/keypoints/dewarped/20230329_preprocessed_complete_dewarped_batch_{batch_idx}_{round(5/subsample, 2)}Hz.hdf5', None)


    # ANIMATE KEYPOINTS
    # ds_kp = kp_detections_to_xr(h5_kp)


    # LOOK AT PHASE
    # px_to_cm = 1/(3.7/50)**2 # Turns density values from /px**2 to /cm**2
    # per_m2 = 1/(3.7/50)**2*(100**2) # Turns density values from /px**2 to /m**2
    
    # h5_prep = f'/keypoints/20230329_preprocessed_complete_{round(5/subsample, 2)}Hz.hdf5'
    # h5_prep = f'/keypoints/20230329_preprocessed_complete_batch_{batch_idx}_{round(5/subsample, 2)}Hz.hdf5'
    # ds = load_preprocessed_data(h5_prep)
    # ds = get_local_env(ds, 'metric', 100) # 2 BL
    # ds = get_local_env(ds, 'metric', 200, arena_center, arena_radius) # 4 BL
    # ds = get_local_env(ds, 'voronoi', None, arena_center, arena_radius, density_factor = per_m2)
    # ds = get_local_env(ds, 'metric', 400) # 8 BL
    # ds = get_local_env(ds, 'metric', 500) # 10 BL
    # save_ds(ds, f'/keypoints/20230329_preprocessed_complete_batch_{batch_idx}_{round(5/subsample, 2)}Hz.hdf5', None)
    # plot_phase(ds, 'density_metric_100', 'polarization_metric_100', plots_path, [r'Local density $(/cm^2)$', 'Polarization'], 'Locality: 2 BL', gridsize = 30, x_factor = px_to_cm)
    # plot_phase(ds, 'density_metric_200', 'polarization_metric_200', plots_path, [r'Local density $(/cm^2)$', 'Polarization'], 'Locality: 4 BL', gridsize = 30, x_factor = px_to_cm)
    # plot_phase(ds, 'density_metric_300', 'polarization_metric_300', plots_path, [r'Local density $(/m^2)$', 'Polarization'], 'Locality: 6 BL', gridsize = 30, x_factor = px_to_m)
    # plot_phase(ds, 'density_voronoi_None', 'polarization_voronoi_None', plots_path, [r'Local density $(/m^2)$', 'Polarization'], 'Locality: First shell', gridsize = 30, x_factor = px_to_m)
    # plot_distribution_over_time(ds, 'density_metric_200', r'Local density $(/cm^2)$', plots_path, 'Locality: 4 BL', y_factor = px_to_cm, y_bins = 30)
    # plot_distribution_over_time(ds, 'polarization_metric_300', 'Polarization', plots_path, 'Locality: 6 BL', y_factor = 1)
    # print('HERE')
    # plot_distribution_over_time(ds, 'theta', r'$\theta$', plots_path, '', y_factor = 1, start_frame = 10000, end_frame = 15000, subsample = 5)
    # plot_distribution_over_time(ds, 'theta', r'$\theta$', plots_path, '', y_factor = 1, start_frame = 33000, end_frame = 38000, subsample = 5)
    # plot_distribution_over_time(ds, 'theta', r'$\theta$', plots_path, '', y_factor = 1, start_frame = 50000, end_frame = 55000, subsample = 5)

    # plot_distribution_over_time(ds, 'density_metric_300', r'Local density $(/m^2)$', plots_path, '', y_factor = px_to_m, start_frame = 10000, end_frame = 15000, subsample = 5)
    # plot_distribution_over_time(ds, 'density_metric_300', r'Local density $(/m^2)$', plots_path, '', y_factor = px_to_m, start_frame = 33000, end_frame = 38000, subsample = 5)
    # plot_distribution_over_time(ds, 'density_metric_300', r'Local density $(/m^2)$', plots_path, '', y_factor = px_to_m, start_frame = 50000, end_frame = 55000, subsample = 5)

    # plot_distribution_over_time(ds, 'polarization_metric_300', 'Polarization', plots_path, '', y_factor = 1, start_frame = 10000, end_frame = 15000, subsample = 5)
    # plot_distribution_over_time(ds, 'polarization_metric_300', 'Polarization', plots_path, '', y_factor = 1, start_frame = 33000, end_frame = 38000, subsample = 5)
    # plot_distribution_over_time(ds, 'polarization_metric_300', 'Polarization', plots_path, '', y_factor = 1, start_frame = 50000, end_frame = 55000, subsample = 5)

    
    # interactive_voronoi_overlay(ds, 'density_voronoi_None', plots_path, arena_center, arena_radius, start_frame = 0, end_frame = 100, subsample = 1, cmap = 'viridis')
    # interactive_voronoi_overlay(ds, 'density_voronoi_None', plots_path, arena_center, arena_radius, start_frame = 0, end_frame = 1000, subsample = 20, cmap = 'viridis')
    
    # interactive_voronoi_distributions(ds, plots_path, arena_center, arena_radius, start_frame = 0, end_frame = 100, subsample = 1, density_factor = 1)
    # interactive_voronoi_distributions(ds, plots_path, arena_center, arena_radius, start_frame = 0, end_frame = 1000, subsample = 20, density_factor = 1)

    # plot_voronoi_corr_single_frame(ds, param = 'theta', frame_idx = 0, arena_center = arena_center, arena_radius = arena_radius, output_dir = plots_path, param_type = 'circular', title = 'Theta', subsample = 30)
    # compare_corrs_single_frame(ds, param = 'polarization_voronoi_None', frame_idx = 0, arena_center = arena_center, arena_radius = arena_radius, output_dir = plots_path, param_type = 'scalar', title = 'Polarization (4 metric body lengths)', subsample = 1000, metric_n_bins = 50, distance_factor = px_to_m**(-0.5))
    
    # analyse_clusters_single_frame(ds, frame_idx = 0, arena_center = arena_center, arena_radius = arena_radius, output_dir = plots_path, pol_thresh = 0.6, min_cluster_size = 2, area_factor = px_to_m**(-1))

    # REFLECTION ANALYSIS
    # define_cycle(ds, output_dir = plots_path, density_factor = px_to_m)
    # find_reflections(ds, fps = 5)
    # whole_batch_cluster_analysis(ds, output_dir = plots_path, arena_center= arena_center, arena_radius= arena_radius, fps = 5, start_frame = 0, end_frame = None, subsample = 1,tolerance = 1e-6,
    # pol_thresh = 0.8, min_cluster_size = 2, area_factor = per_m2**(-1), n_workers = 12)
    # interactive_cluster_analysis('/output/20230329/kp_plots/clusters/cluster_data_start_12094_end_24187_pol_thresh_0.8_min_cluster_size_2.h5', max_layers = 10)
    # interactive_cluster_merging('/output/20230329/kp_plots/clusters/cluster_data_start_12094_end_24187_subsample_1_pol_thresh_0.8_min_cluster_size_2.h5')
    # interactive_cluster_structure(ds, input_path = '/output/20230329/kp_plots/clusters/cluster_data_start_12094_end_24187_subsample_1_pol_thresh_0.8_min_cluster_size_2.h5', layer_cutoff = 10)
    
    pass
