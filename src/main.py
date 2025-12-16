'''_____________________________________________________IMPORTS____________________________________________________________'''

from data_handling import *
from clean_tracks import *
from visualize_preprocessed import *
from compute_activity import *
from visualize_activity import *
from compute_neighbours import *
from check_frequencies import *

'''_____________________________________________________PARAMETERS____________________________________________________________'''
# Loading parameters
batch_num = 1
exp_name = '20230329'
num_ids = None # None means all, for loading from TReX outputs

# Activity thresholds for each computed batch (index is batch number)
threshes = [0.3352, 0.3243] # High order diff speed

# Smoothing and interpolating parameters
# speed_dict = {'raw': None, 'moving_avg': {'window_length': 5, 'center': True}, 'moving_med': {'window_length': 5, 'center': True}, 'sg': {'window_length': 5, 'polyorder': 2, 'deriv': 0, 'delta': 1.0}}
speed_dict = {'raw': None, 'high_ord':{'dt': 1, 'num_iterations': 1, 'order': 4}, 'moving_avg': {'window_length': 3, 'center': True}, 'moving_med': {'window_length': 3, 'center': True}, 
              'sg': {'window_length': 3, 'polyorder': 2, 'deriv': 0, 'delta': 1.0}, 'butter': {'dt': 1, 'filter_order': 2, 'cutoff_freq': 0.5}}
interp_dict= {'max_gap': 5, 'max_dist': 10}
fill_gaps = True

# Visualizing and animating parameters
vid_path = './tracking/trex_inputs/20230329.mp4'

# Neighbour computation parameters
# inter_dict = {'metric': [15, 20, 25, 30], 'topo': [1, 3, 7], 'voronoi': [None]}
inter_dict = {'metric': [15, 20], 'topo': [3, 5]}

# Frequency check parameters
fmin = 0.05 # Minimum frequency resolution in Hz for PSD
quant_low = 0.15
quant_high = 0.5

# Saving params
suffix = ''
'''_____________________________________________________RUN CODE____________________________________________________________'''

if __name__ == "__main__":

    # LOAD RAW DATA
    # print('Make sure batch number is not excluded in dockerignore.')
    # num_ids, ds_raw = load_trex_data(batch_num, exp_name, num_ids) # Last integer specifies how many IDs to include
    # print('TRex data loaded. Number of IDs:', num_ids)

    # PREPROCESS DATA
    # ds = preprocess_data(ds_raw, speed_dict, fill_gaps=fill_gaps, interp_dict=interp_dict, center_only=True, radius=960) # Center only to exclude stray detections near borders
    # print('Data pre-processed.')

    # SAVE PREPROCESSED DATA
    # ds_save_name = f'./preprocessed/{exp_name}/batch_{batch_num}/traj_data' + suffix
    # save_ds(ds, ds_save_name)
    # print('Data saved.')

    # LOAD PREPROCESSED DATA
    ds_load_name = f'./preprocessed/{exp_name}/batch_{batch_num}/traj_data' + '.h5'
    ds = load_preprocessed_data(ds_load_name)
    print('Pre-processed data loaded.')
    # print(ds.data_vars)

    # PLOT SPEED HISTOGRAMS
    # plot_speed_hists(ds, speed_names = ['raw', 'high_ord', 'sg'], exp_name = exp_name, batch_num = batch_num, fit_speed = False)
    # print('Plotted speed histograms.')

    # PLOT SMOOTHED COORDINATES
    plot_smoothed_coords(ds, id = 0, speed_names = ['high_ord', 'moving_avg', 'moving_med', 'sg', 'butter'], t_slice = slice(0, 200), exp_name = exp_name, batch_num = batch_num)
    print('Plotted smoothed coordinates.')

    # PLOT HISTOGRAMS OF TRACK LENGTHS
    # plot_tracklet_lengths_hist(ds_raw, speed_dict, interp_dict, radius=800, exp_name = exp_name, batch_num = batch_num)
    # print('Plotted tracklet length histograms.')

    # PLOT ORIENTATIONS AND ANGULAR SPEED OVER TIME
    # plot_ang_speed(ds, t_slice = slice(0, 500), exp_name = exp_name, batch_num = batch_num)
    # print('Plotted orientations and angular speed over time.')

    # PLOT NUMBER OF TRACKLETS OVER TIME
    # plot_num_tracklets_over_time(ds, exp_name = exp_name, batch_num = batch_num)
    # print('Plotted number of tracklets over time.')

    # ANIMATE TRACKLET LENGTHS
    # animate_trajs_coloured(ds, vid_path, exp_name = exp_name, batch_num = batch_num, colours=ds['tracklet_length'], cbar_name='Tracklet length', start_frame=0, end_frame=-1, interval=50)
    # print('Animated tracklet lengths.')

    # COMPUTE NEIGHBOURS
    # create_nbrs_h5(ds, inter_dict, exp_name, batch_num, do_regions = True, num_regions=16)

    # VALIDATE ACTIVITY QUANTILES
    # validate_quantiles(ds, f_min=fmin, inactive_quant=quant_low, active_quant=quant_high, exp_name=exp_name, batch_num=batch_num, plot=True)

    # COMPUTES PSDS
    # compute_activity_psd(ds, f_min=0.5, inactive_quant=quant_low, active_quant=quant_high, exp_name=exp_name, batch_num=batch_num, smooth_list = ['high_ord', 'moving_avg', 'moving_med', 'sg', 'butter'])
    # compute_total_psd(ds, fmin, exp_name, batch_num, smooth_list = ['high_ord', 'moving_avg', 'moving_med', 'sg', 'butter'])
    # print('PSDs computed.')

    # LOAD PSD FILE
    # psd_load_name = f'./preprocessed/{exp_name}/batch_{batch_num}/psd' + '.h5'
    # psds = load_psds_hdf5(psd_load_name)
    # print('PSD data loaded.')

    # PLOT PSDS
    # plot_psds(psds, f_min = 0.5, exp_name=exp_name, batch_num=batch_num, smooth_names = ['high_ord', 'moving_avg', 'moving_med', 'sg', 'butter'], actives = True, normalize = False, quants = [quant_low, quant_high])

    # ANIMATE NEIGHBOURS
    # nbrs = load_neighbours_hdf5(f'./preprocessed/{exp_name}/batch_{batch_num}/nbrs.h5')
    # print(nbrs)
    # animate_neighbours(ds, nbrs, interaction = 'topo', inter_param = 5, fid = 0, end_frame = 500, video_path = vid_path, exp_name = exp_name, batch_num = batch_num, buffer = 80)

    pass
