'''_____________________________________________________IMPORTS____________________________________________________________'''

import h5py
# from ultralytics import YOLO

from data_handling import *
from clean_tracks import *
from visualize_preprocessed import *
from compute_activity import *
from visualize_activity import *
from compute_neighbours import *
from check_frequencies import *
from visualize_time import *
from image_analysis import *
from keypoints import *
from self_track import *
from entropy import *
from visualize_entropy import *
from phase import *
from visualize_phase import *
from animate import *
'''_____________________________________________________PARAMETERS____________________________________________________________'''

# Training parameters
workspace = "locustorientations"
project_name = "locusts_2kp"
data_version = 8
yolo_short_name = "yolo26"
yolo_version = 'yolo26n-pose.pt'
img_sz = 640

# Keypoints path
h5_kp='/keypoints/20230329_processed_complete.hdf5'
kp_local_env = '/keypoints/20230329_local_env_2.hdf5'


# Entropy params
n_layers = 6
n_ang0 = 16
r_max = 200
n_focals = 1000
occ_path = f'/keypoints/20230329_{n_layers}_{n_ang0}_{r_max}_{n_focals}.hdf5'

# Loading parameters
batch_num = 1
exp_name = '20230329'
num_ids = None # None means all, for loading from TReX outputs

# Activity thresholds for each computed batch (index is batch number)
threshes = [0.3352, 0.3243] # High order diff speed

# Smoothing and interpolating parameters
# speed_dict = {'raw': None, 'moving_avg': {'window_length': 5, 'center': True}, 'moving_med': {'window_length': 5, 'center': True}, 'sg': {'window_length': 5, 'polyorder': 2, 'deriv': 0, 'delta': 1.0}}
# speed_dict = {'raw': None, 'high_ord':{'dt': 1, 'num_iterations': 1, 'order': 4}, 'moving_avg': {'window_length': 3, 'center': True}, 'moving_med': {'window_length': 3, 'center': True}, 
            #   'sg': {'window_length': 3, 'polyorder': 2, 'deriv': 0, 'delta': 1.0}, 'butter': {'dt': 1, 'filter_order': 2, 'cutoff_freq': 0.5}}
speed_dict = {'raw': None, 'high_ord':{'dt': 1, 'num_iterations': 1, 'order': 4}, 'butter': {'dt': 1, 'filter_order': 2, 'cutoff_freq': 0.3}, 'spline': {'degree': 2, 's': 0.5}}
interp_dict= {'max_gap': 1, 'max_dist': 0.5}
fill_gaps = False

params = {'speed_dict': speed_dict, 'interp_dict': interp_dict, 'fill_gaps': fill_gaps}

# Visualizing and animating parameters
vid_path = '/videos/20230329.mp4'
img_dir = '/original/20230329/video/'

# Neighbour computation parameters
# inter_dict = {'metric': [15, 20, 25, 30], 'topo': [1, 3, 7], 'voronoi': [None]}
inter_dict = {'metric': [15, 20], 'topo': [3, 5]}

# Frequency check parameters
fmin = 0.05 # Minimum frequency resolution in Hz for PSD
quant_low = 0.15
quant_high = 0.5

# Saving params
suffix = ''
plots_path = '/output/' + exp_name + '/'
'''_____________________________________________________RUN CODE____________________________________________________________'''

if __name__ == "__main__":

    # GENERATE TILES FOR TRAINING KEYPOINTS FROM VIDEO
    # extract_random_tiles(image_dir=img_dir, output_dir="/keypoints/unlabeled_tiles", n_tiles=50, tile_size=320)

    # TRAIN MODEL
    # train_model(workspace, project_name, data_version, yolo_short_name, yolo_version, epochs = 1, img_sz = img_sz)

    # EVALUATE MODEL AND VISUALIZE KEYPOINTS
    # visualize_results(weights_path='/keypoints/best_kp_weights.pt', img_idx=4, img_dir='./locusts_2kp-8/')

    # TEST ONE IMAGE
    # with h5py.File('/keypoints/temp_test.hdf5','w') as f_out:
    #     model = YOLO('/keypoints/best_kp_weights.pt')
    #     slice_img(model, img_path='/original/20230329/video/65MP01_10Kmarching_01_2023-03-29_10-10-24-124.jpg', h5_out=f_out, frame_num=0, overlap = 0.3)

    # VISUALIZE WHOLE FRAME
    # visualize_frame_results(h5_file='/keypoints/temp_test.hdf5', img_path='/original/20230329/video/65MP01_10Kmarching_01_2023-03-29_10-10-24-124.jpg', frame_num=0)

    # GET KEYPOINTS FOR WHOLE VIDEO
    # slice_folder_to_h5(path_to_model = '/keypoints/best_kp_weights.pt', frames_dir = img_dir, h5_in = '/keypoints/20230329_unprocessed_kps.hdf5', start_idx = 0, stop_idx = 20000, chunk_size = 200, tile_size = 640, img_size = 7000, overlap = 0.3)
   
    # PREPROCESS KEYPOINTS TO GET RID OF DUPLICATES ACROSS TILES
    # preprocess_kps_fast(h5_in='/keypoints/20230329_unprocessed_complete.hdf5')
    # unprocessed_stats(h5_in = '/keypoints/20230329_unprocessed_complete.hdf5', subsample = 10, n_bins = 100, output_dir = plots_path)

    # ANIMATE KEYPOINTS
    ds_kp = detections_h5_to_kp_xr(h5_kp)
    # GET KEYPOINTS FOR WHOLE VIDEO
    animate_zoom_out(ds_kp, img_dir, plots_path, start_frame = 0, end_frame = 250)

    # COMPUTE ENTROPY
    # ds = detections_h5_to_xr_dataset(h5_kp)
    # occupancies, k_layers = compute_occupancy_fixed_dr(ds, n_layers, n_ang0, r_max, n_focals, rotation_null = False, shuffle_null = False)
    # occ_da = xr.DataArray(
    #     occupancies, 
    #     dims=['frame', 'id', 'n_bins'],
    #     coords={
    #         'frame': np.arange(occupancies.shape[0]),
    #         'id': np.arange(occupancies.shape[1]),
    #         'bin': np.arange(occupancies.shape[2])},
    #     name='occupancy_map')

    # occ_ds = xr.Dataset(
    #     data_vars={'occupancy': occ_da},
    #     attrs={
    #         'description': 'Local occupancy bin counts for social bubble analysis',
    #         'units': 'binary_count',
    #         'bin_resolution_cm': 0.5, # Example metadata
    #         'created_at': '2026-03-27'
    #     }
    # )
    # save_ds(occ_ds, occ_path, None)
    # # null_occupancies, _ = compute_occupancy_fixed_dr(ds, n_layers, n_ang0, r_max, n_focals, rotation_null = True, shuffle_null = False)
    # null_occupancies, _ = compute_occupancy_fixed_dr(ds, n_layers, n_ang0, r_max, n_focals, rotation_null = False, shuffle_null = True)
    # n_samples = get_min_valids([occupancies, null_occupancies])
    # # bin_occupation_histogram(occupancies, plots_path, k_layers, r_max, n_focals)
    # entropy_over_time(occupancies, null_occupancies, 'Shuffle', plots_path, k_layers, n_samples, r_max, n_focals)
    # # plot_pca(occupancies, plots_path, k_layers, r_max, n_focals)
    # plot_bin_heatmap(occupancies, plots_path, k_layers, r_max)
    # n_samples = get_min_valids([occupancies])
    # state_hist(occupancies, plots_path, k_layers, n_samples, r_max, n_focals)

    # occ_ds = load_preprocessed_data(occ_path)
    # occupancies = occ_ds.values
    # k_layers = [8, 24, 40]
    # occupants_histogram(occupancies, plots_path, k_layers, r_max, n_focals)
    # plot_density_bin_heatmap(occupancies, plots_path, k_layers, r_max)

    # LOOK AT PHASE
    # ds = load_preprocessed_data(kp_local_env)
    # ds = get_local_env(ds, 'metric', 100) # 2 BL
    # ds = get_local_env(ds, 'metric', 200) # 4 BL
    # ds = get_local_env(ds, 'metric', 300) # 6 BL
    # ds = get_local_env(ds, 'metric', 400) # 8 BL
    # ds = get_local_env(ds, 'metric', 500) # 10 BL
    # save_ds(ds, kp_local_env, None)
    px_to_cm = 1/(3.7/50)**2 # Turns density values into /px**2 to /cm**2
    # plot_phase(ds, 'density_metric_100', 'polarization_metric_100', plots_path, [r'Local density $(/cm^2)$', 'Polarization'], 'Locality: 2 BL', gridsize = 30, x_factor = px_to_cm)
    # plot_phase(ds, 'density_metric_200', 'polarization_metric_200', plots_path, [r'Local density $(/cm^2)$', 'Polarization'], 'Locality: 4 BL', gridsize = 30, x_factor = px_to_cm)
    # plot_phase(ds, 'density_metric_300', 'polarization_metric_300', plots_path, [r'Local density $(/cm^2)$', 'Polarization'], 'Locality: 6 BL', gridsize = 30, x_factor = px_to_cm)
    # plot_distribution_over_time(ds, 'density_metric_200', r'Local density $(/cm^2)$', plots_path, 'Locality: 4 BL', y_factor = px_to_cm, y_bins = 30)
    # plot_distribution_over_time(ds, 'polarization_metric_300', 'Polarization', plots_path, 'Locality: 6 BL', y_factor = 1)
    # plot_distribution_over_time(ds, 'theta', r'$\theta$', plots_path, '', y_factor = 1)

    # TRACK WITH MAYOLO
    # track_kps(detections_h5_path='/keypoints/20230329_processed_kps.hdf5', output_dir='/keypoints/20230329_tracked_kps', start_frame=0, end_frame=8, max_frames_lost=3)

    # CREATE CSV FILE FOR TREX TRACKING
    # detections_h5_to_trex_csv('/keypoints/20230329_processed_kps.hdf5', '/keypoints/20230329_trex_input.csv', end_frame = 3)

    # # LOAD RAW DATA
    # print('Make sure batch number is not excluded in dockerignore.')
    # num_ids, ds_raw = load_trex_data(batch_num, exp_name, 1000) # Last integer specifies how many IDs to include
    # print('TRex data loaded. Number of IDs:', num_ids)

    # # PREPROCESS DATA
    # ds = preprocess_raw(ds_raw, speed_dict, fill_gaps=fill_gaps, interp_dict=interp_dict, center_only=True, radius=960) # Center only to exclude stray detections near borders
    # print('Data pre-processed.')

    # # SAVE PREPROCESSED DATA
    # ds_save_name = f'/output/preprocessed/{exp_name}/batch_{batch_num}/traj_data' + suffix
    # save_ds(ds, ds_save_name)
    # print('Data saved.')

    # LOAD PREPROCESSED DATA
    # ds_load_name = f'/output/preprocessed/{exp_name}/batch_{batch_num}/traj_data' + '.h5'
    # ds = load_preprocessed_data(ds_load_name)
    # print('Pre-processed data loaded.')
    # print(ds.data_vars)
    # print(ds.isel(id=0, frame=slice(0, 300))['x_raw'])
    # print(ds.isel(id=0, frame=slice(0, 300))['x_butter'])

    # COMPUTE MEDIAN PER-LOCUST AREA FROM LOW DENSITY LOCUSTS
    # val_threshold = 20
    # # Modify positions to account for different resolutions
    # factor = 7000/1920
    # ds['x_high_ord'] = ds['x_high_ord'] * factor
    # ds['y_high_ord'] = ds['y_high_ord'] * factor
    # med_locust_area = compute_single_locust_area(image_dir=img_dir, ds=ds, pos_name='high_ord', num_individuals=1000, density_radius=70, val_threshold=val_threshold, exp_name=exp_name, batch_num=batch_num)
    # # print('Median locust area from low density locusts:', med_locust_area)

    # ESTIMATE NUMBER OF LOCUSTS PER FRAME
    # med_locust_area = 1289.5
    # num_locusts = estimate_locust_number(image_dir=img_dir, per_locust_area=med_locust_area, val_threshold=val_threshold, area_threshold=0.7*med_locust_area, radius_inclusion=3500, start_frame = 0, end_frame=1000)
    # # print('Estimated number of locusts per frame:', num_locusts)
    # plt.plot(num_locusts)
    # plt.xlabel('Frame number', fontsize = 17)
    # plt.ylabel('Estimated number of locusts', fontsize = 17)
    # plt.savefig(f'plots/{exp_name}/batch_{batch_num}/preprocess/estimated_num_locusts.png')

    # PLOT TRACKS
    # plot_tracks(ds, ids = [0, 1], end_frame = 300, exp_name = exp_name, batch_num = batch_num)

    # PLOT ORIENTATION AND ANGULAR SPEED OVER TIME FOR A COUPLE OF IDS
    # plot_ang_speed(ds, speed_name = 'spline', exp_name = exp_name, batch_num = batch_num, end_frame = 300)

    # PLOT SPEED HISTOGRAMS
    # plot_speed_hists(ds, speed_names = ['raw', 'high_ord', 'butter'], exp_name = exp_name, batch_num = batch_num, fit_speed = False)
    # print('Plotted speed histograms.')

    # PLOT SMOOTHED COORDINATES

    # plot_smoothed_coords(ds, id = 0, speed_names = ['high_ord', 'butter', 'spline'], t_slice = slice(0, 300), exp_name = exp_name, batch_num = batch_num)
    # print('Plotted smoothed coordinates.')

    # PLOT CORRELATION BETWEEN SPEED AND TRACKLET LENGTH
    # corr_speed_tracklet_length(ds, speed_name='high_ord', exp_name=exp_name, batch_num=batch_num)
    # corr_speed_pos_in_tracklet(ds, speed_name='high_ord', exp_name=exp_name, batch_num=batch_num)

    # PLOT SINGLE HISTOGRAM OF SMOOTHED TRACKLET LENGTHS
    # plot_single_tracklet_lengths(ds, speed_name='spline', exp_name=exp_name, batch_num=batch_num)

    # PLOT MULTIPLE HISTOGRAMS OF TRACK LENGTHS
    # plot_tracklet_lengths_hist(ds_raw, speed_dict, interp_dict, radius=960, exp_name = exp_name, batch_num = batch_num)
    # print('Plotted tracklet length histograms.')

    # PLOT ORIENTATIONS AND ANGULAR SPEED OVER TIME
    # plot_ang_speed(ds, speed_name = 'spline', exp_name = exp_name, batch_num = batch_num, end_frame = 300)
    # print('Plotted orientations and angular speed over time.')

    # PLOT NUMBER OF TRACKLETS OVER TIME
    # plot_num_tracklets_over_time(ds, exp_name = exp_name, batch_num = batch_num)
    # print('Plotted number of tracklets over time.')

    # ANIMATE TRACKLET LENGTHS
    # animate_trajs_coloured(ds, vid_path, exp_name = exp_name, batch_num = batch_num, colours=ds['tracklet_length'], cbar_name='Tracklet length', start_frame=3500, end_frame=4000, interval=50)
    # print('Animated tracklet lengths.')

    # ANIMATE EGOCENTRIC VIEW OF A FOCAL INDIVIDUAL
    # animate_focal_ego(ds, fid=5, video_path=vid_path, smooth_name='spline', exp_name=exp_name, batch_num=batch_num, buffer=50, start_frame=0, end_frame=500, interval=80)

    # COMPUTE NEIGHBOURS
    # create_nbrs_h5(ds, inter_dict, exp_name, batch_num, do_regions = False)

    # VALIDATE ACTIVITY QUANTILES
    # validate_quantiles(ds, f_min=fmin, inactive_quant=quant_low, active_quant=quant_high, exp_name=exp_name, batch_num=batch_num, plot=True)

    # COMPUTES PSDS
    # compute_activity_psd(ds, f_min=0.2, inactive_quant=quant_low, active_quant=quant_high, exp_name=exp_name, batch_num=batch_num, smooth_list = ['high_ord', 'moving_avg', 'moving_med', 'sg', 'butter'])
    # compute_total_psd(ds, fmin, exp_name, batch_num, smooth_list = ['high_ord', 'moving_avg', 'moving_med', 'sg', 'butter'])
    # print('PSDs computed.')

    # LOAD PSD FILE
    # psd_load_name = f'/output/preprocessed/{exp_name}/batch_{batch_num}/psd' + '.h5'
    # psds = load_psds_hdf5(psd_load_name)
    # print('PSD data loaded.')

    # PLOT PSDS
    # plot_psds(psds, f_min = 0.2, exp_name=exp_name, batch_num=batch_num, smooth_names = ['high_ord', 'moving_avg', 'moving_med', 'sg', 'butter'], actives = True, normalize = True, quants = [quant_low, quant_high])

    # PLOT AVERAGED AUTOCORELLATION FOR ALL INDIVIDUALS
    # tau_max = 100
    # speed_name = 'v_butter'
    # valid_trajs = list_long_tracklets(ds[speed_name].where(~np.isnan(ds[speed_name])), min_tracklet_length=5*tau_max)
    # autocorrs, taus = compute_autocorr_tracklets(valid_trajs, tau_max)
    # plot_autocorr(autocorrs, taus, speed_name = speed_name, exp_name=exp_name, batch_num=batch_num)

    # PLOT AVERAGED AUTOCORELLATION BY ACTIVITY
    # speed_name = 'v_butter'
    # autocorrs, taus = compute_activity_autocorr(ds, [quant_low, quant_high], speed_name, tau_max=50)
    # plot_activity_autocorr(autocorrs, taus, [quant_low, quant_high], speed_name = speed_name, exp_name=exp_name, batch_num=batch_num)
    # powers, freq = compute_activity_autocorr_psd(autocorrs, fs=5, f_min=0.5)
    # plot_activity_autocorr_psd(powers, freq, [quant_low, quant_high], speed_name = speed_name, exp_name=exp_name, batch_num=batch_num)

    # ANIMATE NEIGHBOURS
    # nbrs = load_neighbours_hdf5(f'/output/preprocessed/{exp_name}/batch_{batch_num}/nbrs.h5')
    # # print(nbrs)
    # animate_neighbours(ds, nbrs, interaction = 'topo', inter_param = 5, fid = 1, end_frame = 300, video_path = vid_path, speed_name = 'spline', exp_name = exp_name, batch_num = batch_num, buffer = 80)

    # corr_tracklet_length_density(ds, nbrs, exp_name=exp_name, batch_num=batch_num, radius = 20)
    # corr_tracklet_length_centrality(ds, exp_name=exp_name, batch_num=batch_num)

    # MULTI-BATCH ANALYSIS

    # PREPROCESS ALL BATCHES
    # preprocess_save_all_batches(exp_name, num_batches=11, speed_dict=speed_dict, fill_gaps=fill_gaps, interp_dict=interp_dict, center_only=True, radius=960, reprocess=False)
        
    # COUNT DETECTIONS OVER TIME ACROSS ALL FRAMES
    # bb_detections, kp_detections = bb_vs_kp_detections_over_time(bb_h5 = '/bb/h5s/10K_full_2.hdf5', kp_preprocessed_h5 = '/keypoints/20230329_processed_kps.hdf5', plot = True)

    # COUNT TRACKLETS OVER TIME ACROSS ALL BATCHES
    # tracklets_over_time(speed_name='raw', num_batches=11, exp_name=exp_name, num_detections=num_detections)
    pass
