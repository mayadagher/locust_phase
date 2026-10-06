'''_____________________________________________________IMPORTS____________________________________________________________'''

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
from visualize_entropy import *
from phase import *
from visualize_phase import *
from animate import *
from sliders import *

'''_____________________________________________________PARAMETERS____________________________________________________________'''

# Arena params (found using define_boundary.py, after dewarping)
arena_center_px_warped =  np.array([3767.15, 3767.15]) # px, found using define_boundary
arena_radius_px_warped = 3395.79 # px, found using define_boundary
arena_center_m = np.array([0.01, 0])
arena_radius_m = 2.38

# Loading parameters
n_batches = 11
batch_num = 8
fs = 5

# Preprocess parameters
h5_dir = f'/bb/20230329/dewarped/'
smooth_dict = {'spline': {'degree': 2, 's': 0.000002}}
interp_dict = {'max_gap': 1, 'max_dist_m': 0.02}
fill_gaps = True
center_only = True

# Activity thresholds for each computed batch (index is batch number)
threshes = [0.3352, 0.3243] # High order diff speed

# Visualizing and animating parameters
vid_path = '/original/20230329/20230329.mp4'
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
plots_path = '/output/' + exp_name + '/bb_plots/dewarped/'

# Undistortion parameters
calibration_path = '/intrinsics/arena_board_calibration/calibration_official.yaml'
frame_width = 7000
frame_height = 7000

'''_____________________________________________________RUN CODE____________________________________________________________'''

if __name__ == "__main__":

    # LOAD TREX DETECTIONS AND CALIBRATE THEM
    # for i in range(n_batches):
    #     ds = load_trex_data(i, exp_name, calibration_path, frame_height, frame_width, scale_factor = 1920)
    #     save_name = f'/bb/{exp_name}/dewarped/batch_{i}_5.0Hz'
    #     save_ds(ds, save_name, {'dewarped': True})

    # PREPROCESS DEWARPED DATA
    # preprocess_and_save_all_batches(n_batches = n_batches, h5_dir = h5_dir, smooth_dict = smooth_dict, fill_gaps = fill_gaps, interp_dict = interp_dict, 
    #                                 center_only = center_only, arena_center_m = arena_center_m, arena_radius_m = arena_radius_m, fs = fs)

    # LOAD PREPROCESSED DATA
    ds_load_name = f'/bb/20230329/dewarped/batch_{batch_num}_5.0Hz.hdf5'
    ds = load_preprocessed_data(ds_load_name)
    # print('Pre-processed data loaded.')

    # CHECK PREPROCESS JOB
    # plot_smoothed_coords(ds, plots_path, smooth_dict, id = 2, start_frame = 0, end_frame = 150)

    # CHECK SPATIAL DISTRIBUTION OF SPEED
    # plot_voronoi_overlay(ds, 1, arena_center_m, arena_radius_m, values = 'v_spline', plot = True, bb = True)

    interactive_voronoi_overlay(ds, 'v_spline', plots_path, arena_center_m, arena_radius_m, start_frame = 0, end_frame = 100, subsample = 1, cmap = 'magma', bb = True)
    interactive_voronoi_overlay(ds, 'v_spline', plots_path, arena_center_m, arena_radius_m, start_frame = 0, end_frame = 1000, subsample = 20, cmap = 'magma', bb = True)

    # COMPUTE MEDIAN PER-LOCUST AREA FROM LOW DENSITY LOCUSTS
    # val_threshold = 20
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

    # plot_smoothed_coords(ds, output_dir = f'/output/{exp_name}/bb_plots/batch_{batch_num}/preprocess/', id = 0, smooth_names = ['high_ord', 'butter', 'spline'], start_frame = 0, end_frame = 300)
    # print('Plotted smoothed coordinates.')

    # PLOT CORRELATION BETWEEN SPEED AND TRACKLET LENGTH
    # corr_speed_tracklet_length(ds, speed_name='high_ord', exp_name=exp_name, batch_num=batch_num)
    # corr_speed_pos_in_tracklet(ds, speed_name='high_ord', exp_name=exp_name, batch_num=batch_num)

    # PLOT SINGLE HISTOGRAM OF SMOOTHED TRACKLET LENGTHS
    # plot_single_tracklet_lengths(ds, speed_name='spline', exp_name=exp_name, batch_num=batch_num)

    # PLOT MULTIPLE HISTOGRAMS OF TRACK LENGTHS
    # plot_tracklet_lengths_hist(ds_raw, speed_dict, interp_dict, radius=960, exp_name = exp_name, batch_num = batch_num)
    # print('Plotted tracklet length histograms.')

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
        
    # COUNT DETECTIONS OVER TIME ACROSS ALL FRAMES
    # bb_detections, kp_detections = bb_vs_kp_detections_over_time(bb_h5 = '/bb/20230329/detections_warped/h5s/10K_full_2.hdf5', kp_preprocessed_h5 = '/keypoints/20230329/complete_warped/kp_processed_complete.hdf5', plot = True, plots_path = f'/output/{exp_name}/both/')

    # COUNT TRACKLETS OVER TIME ACROSS ALL BATCHES
    # tracklets_over_time(smooth_name='spline', num_batches=n_batches, h5_in=f'/bb/{exp_name}/dewarped/', num_detections=bb_detections, output_dir='/output/' + exp_name + '/bb_plots/dewarped/check_detections_and_tracklets/')
    pass
