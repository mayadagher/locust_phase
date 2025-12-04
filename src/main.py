'''_____________________________________________________IMPORTS____________________________________________________________'''

from data_handling import *
from clean_tracks import *
from visualize_preprocessed import *
from compute_activity import *
from visualize_activity import *
from compute_neighbours import *

'''_____________________________________________________PARAMETERS____________________________________________________________'''
# Loading parameters
batch_num = 1
exp_name = '20230329'
num_ids = None # None means all, for loading from TReX outputs

# Activity thresholds for each computed batch (index is batch number)
threshes = [0.3352, 0.3243] # High order diff speed

# Smoothing and interpolating parameters
# speed_dict = {'raw': None, 'moving_avg': {'window_length': 5, 'center': True}, 'moving_med': {'window_length': 5, 'center': True}, 'sg': {'window_length': 5, 'polyorder': 2, 'deriv': 0, 'delta': 1.0}}
speed_dict = {'raw': None, 'high_ord':{'dt': 1, 'num_iterations': 1, 'order': 4}}
interp_dict= {'max_gap': 5, 'max_dist': 10}
fill_gaps = True

# Visualizing and animating parameters
vid_path = './locust_data/trex_inputs/20230329.mp4'

# Saving params
suffix = ''
'''_____________________________________________________RUN CODE____________________________________________________________'''

if __name__ == "__main__":

    # Load data
    print('Make sure batch number is not excluded in dockerignore.')
    num_ids, ds_raw = load_trex_data(batch_num, exp_name, num_ids) # Last integer specifies how many IDs to include
    print('TRex data loaded. Number of IDs:', num_ids)

    # Pre-process data
    ds = preprocess_data(ds_raw, speed_dict, fill_gaps=fill_gaps, interp_dict=interp_dict, center_only=True, radius=960) # Center only to exclude stray detections near borders
    print('Data pre-processed.')

    # Save pre-processed data
    save_data(ds, 'traj_data' + suffix)
    print('Data saved.')

    # Load previously pre-processed data

    # ds = load_preprocessed_data('traj_data' + suffix + '.h5')
    # print('Pre-processed data loaded.')

    # Plot speed histograms
    # plot_speed_hists(ds, speed_names = ['raw', 'high_ord', 'sg'], exp_name = exp_name, batch_num = batch_num, fit_speed = False)
    # print('Plotted speed histograms.')

    # print(np.nanstd(ds['sg_speed'].values))

    # Plot smoothed coordinates
    # plot_smoothed_coords(ds, id = 0, speed_names = ['high_ord', 'moving_med', 'sg'], t_slice = slice(0, 200), exp_name = exp_name, batch_num = batch_num)
    # print('Plotted smoothed coordinates.')

    # Plot histograms of track lengths
    # plot_tracklet_lengths_hist(ds_raw, speed_dict, interp_dict, radius=800, exp_name = exp_name, batch_num = batch_num)
    # print('Plotted tracklet length histograms.')

    # Plot orientations and angular speed over time
    # plot_ang_speed(ds, t_slice = slice(0, 500), exp_name = exp_name, batch_num = batch_num)
    # print('Plotted orientations and angular speed over time.')

    # Plot number of tracklets over time
    # plot_num_tracklets_over_time(ds, exp_name = exp_name, batch_num = batch_num)
    # print('Plotted number of tracklets over time.')

    # Animate tracklet lengths
    # animate_trajs_coloured(ds, vid_path, exp_name = exp_name, batch_num = batch_num, colours=ds['tracklet_length'], cbar_name='Tracklet length', start_frame=0, end_frame=-1, interval=50)
    # print('Animated tracklet lengths.')
    pass
