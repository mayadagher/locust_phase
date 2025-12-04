'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import math
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import seaborn as sns
from scipy.optimize import curve_fit
import scipy.stats as st

from clean_tracks import preprocess_data

'''_____________________________________________________ANIMATION FUNCTIONS____________________________________________________________'''
def animate_trajs_lined(ds, video_path: str, exp_name: str, batch_num: int, buffer = 150, start_frame=0, end_frame=-1, interval=50, trail=10):
    """
    Animate trajectories from an xarray.Dataset over a subsection of a video. buffer specifies the size of the window to use for the animation.
    start_frame and end_frame should be relative to the actual start of the batch (since subsequent batches won't start at 0).
    """

    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Manually find start and end frame in dataset if not specified
    first_frame = int(ds.frame.min())
    last_frame = int(ds.frame.max())
    num_frames = last_frame - first_frame + 1

    if end_frame < 0:
        end_frame = num_frames

    # Check appropriate start and end frame inputs
    assert start_frame < num_frames, f"Start frame is too large for the number of frames for this batch, which is {num_frames}."
    assert end_frame <= num_frames, f"End frame is too large for the number of frames for this batch, which is {num_frames}."
    assert start_frame < end_frame, f"Start frame must be less than end frame."

    # Define frames in animation (consecutive)
    start_frame += first_frame
    end_frame += first_frame
    frames = np.arange(start_frame, end_frame)

    # Define ids
    ids = ds.id.values

    # Subset dataset
    ds_sub = ds.sel(frame = slice(start_frame, end_frame - 1))

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_axis_off()

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first video frame.")
    img_artist = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    lines = {i: ax.plot([], [], '-', lw=1)[0] for i in ids}
    for line in lines.values():
        line.set_zorder(2)
    img_artist.set_zorder(1)

    ax.set_xlim([1920/2 - buffer, 1920/2 + buffer])
    ax.set_ylim([1920/2 - buffer, 1920/2 + buffer])

    def init():
        for line in lines.values():
            line.set_data([], [])
        return [img_artist, *lines.values()]

    def update(idx):
        frame_num = frames[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return [img_artist, *lines.values()]
        img_artist.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        recent = frames[max(0, idx - trail):idx + 1]
        for i in ids:
            ds_recent = ds_sub.sel(id=i).sel(frame=recent)
            lines[i].set_data(ds_recent.x_high_ord, ds_recent.y_high_ord)
        return [img_artist, *lines.values()]

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(f'plots/{exp_name}/batch_{batch_num}/locust_tracks_lined.gif')
    cap.release()
    return

def animate_trajs_coloured(ds, video_path: str, exp_name: str, batch_num: int, colours: xr.DataArray, cbar_name: str, start_frame=0, end_frame=-1, interval=50):
    """
    Scatter points with colours from 'colours' DataArray over video frames. Colours should be per-id, per-frame. Colours elements are assumed to be scalars, not tuples.
    start_frame and end_frame should be relative to the actual start of the batch (since subsequent batches won't start at 0).
    """

    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Manually find start and end frame in dataset if not specified
    first_frame = int(ds.frame.min())
    last_frame = int(ds.frame.max())
    num_frames = last_frame - first_frame + 1

    if end_frame < 0:
        end_frame = num_frames

    # Check appropriate start and end frame inputs
    assert start_frame < num_frames, f"Start frame is too large for the number of frames for this batch, which is {num_frames}."
    assert end_frame <= num_frames, f"End frame is too large for the number of frames for this batch, which is {num_frames}."
    assert start_frame < end_frame, f"Start frame must be less than end frame."

    # Define frames in animation (consecutive)
    start_frame += first_frame
    end_frame += first_frame
    frames = np.arange(start_frame, end_frame)

    # Subset dataset
    ds_sub = ds.sel(frame = slice(start_frame, end_frame - 1))
    colours_sub = colours.sel(frame = slice(start_frame, end_frame - 1))

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_axis_off()

    # Initialize scatter plot artist outside update function
    valid_colours = colours_sub.values[(ds_sub['missing'].values != 1) & ~np.isnan(colours_sub.values)] # Excludes points due to boundaries and due to other third axis specific filtering
    vmax = float(np.nanmax(valid_colours))
    vmin = float(np.nanmin(valid_colours))
    norm = plt.Normalize(vmin, vmax)
    scat = ax.scatter([], [], c=[], cmap = 'viridis', s=0.5, norm=norm)
    scat.set_zorder(2)

    cbar = fig.colorbar(scat, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_ticks(np.linspace(vmin, vmax, 7))
    cbar.set_label(cbar_name)

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first video frame.")
    img_artist = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_artist.set_zorder(1)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return [img_artist, scat]

    def update(idx):
        frame_num = frames[idx]

        # Show frame of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return [img_artist, scat]
        img_artist.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Get positions and tracklet lengths for current frame
        try:
            frame_data = ds_sub.sel(frame=frame_num)
            frame_colours = colours_sub.sel(frame=frame_num).values

            x_vals = frame_data['x_high_ord'].values
            y_vals = frame_data['y_high_ord'].values
            
            # Remove NaN values or those excluded from outside the arena
            valid = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isnan(frame_colours) | frame_data['missing'].values.astype(bool))
            
            # Update scatter plot
            scat.set_offsets(np.c_[x_vals[valid], y_vals[valid]])
            scat.set_array(frame_colours[valid])

        except KeyError:
            # Frame not in dataset
            scat.set_offsets(np.empty((0, 2)))

        if idx % 100 == 0:
            print(f'Processed frame {idx+1}/{len(frames)}')
        
        return [img_artist, scat]

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(f'plots/{exp_name}/batch_{batch_num}/locust_tracks_coloured_{'_'.join(cbar_name.lower().split(' '))}.gif')
    cap.release()
    return

'''_____________________________________________________PLOT FUNCTIONS____________________________________________________________'''
def plot_tracks(ds, t_slice, exp_name: str, batch_num: int):

    plt.figure(figsize=(10, 10))
    for i in ds['id'].values:
        plt.plot(ds['x_raw'].sel(id=i, frame=t_slice), ds['y_raw'].sel(id=i, frame=t_slice), marker='.', label=f'ID {i}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Locust tracks')
    if len(ds['id'].values) <= 10:
        plt.legend()
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/original_tracks.png')

def plot_smoothed_coords(ds, id: int, speed_names: list, t_slice, exp_name: str, batch_num: int): # Takes preprocessed xarray.Dataset as input that has raw and sg computed
    # Look at raw versus smooth x and y data for a single id
    _, axs = plt.subplots(len(speed_names)+2, 1, figsize=(26, 12))
    coords = ['x', 'y']
    # print(ds['v_raw'].isel(id=id, frame=t_slice).values)
    # print(ds['v_sg'].isel(id=id, frame=t_slice).values)

    for i in range(2):
        axs[i].plot(ds['frame'].isel(frame=t_slice), ds[coords[i]].isel(id=id, frame=t_slice), label='Original')
        axs[i].plot(ds['frame'].isel(frame=t_slice), ds[coords[i]+'_high_ord'].isel(id=id, frame=t_slice), label='Smoothed', linestyle='--')
        axs[i].set_xlabel('Frame', fontsize = 17)
        axs[i].set_ylabel(coords[i], fontsize = 17)
        # if i !=2:
        #     axs[i].set_ylim(1200, 1500)

    for i, name in enumerate(speed_names):
        axs[i + 2].plot(ds['frame'].isel(frame=t_slice), ds['v_raw'].isel(id=id, frame=t_slice), label = 'Original')
        axs[i + 2].plot(ds['frame'].isel(frame=t_slice), ds['v_' + name].isel(id=id, frame=t_slice), label = 'Smoothed', linestyle = '--')
        axs[i + 2].set_xlabel('Frame', fontsize = 17)
        axs[i + 2].set_ylabel(name.capitalize(), fontsize = 17)
    axs[-1].legend()

    plt.tight_layout()
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/smoothed_speeds.png')

def fit_mixture_model(data, ax=None, opt = True):
    '''
    Fit two distributions to speed data.

    Parameters
    ----------
    data : array-like
        1D array of speed values (must be non-negative integers or will be rounded)
    ax : matplotlib axis, optional
        Axis to plot on. If None, no plot is created.
    '''

    def exp_pdf(x, r): # func1
        '''PDF of Exponential distribution.'''
        return r * np.exp(-r * x)
    
    def delta_pdf(x, _): # func1
        '''PDF of Delta distribution at zero.'''
        return np.where((x >= bin_edges[0]) & (x < bin_edges[1]), 1.0, 0.0)

    def gamma_pdf(x, alpha, lam, _): # func2
        '''PDF of Gamma distribution.'''
        return (x**(alpha - 1) * np.exp(-x / lam)) / (lam**alpha * math.gamma(alpha))
    
    def lognorm_pdf(x, mu, sigma, _): # func2
        '''PDF of Lognormal distribution.'''
        return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2 * sigma**2))
    
    def weibull_pdf(x, k, lam, _): # func2
        '''PDF of Weibull distribution.'''
        return (k / lam) * (x / lam)**(k - 1) * np.exp(-(x / lam)**k)
    
    def logt_pdf(x, nu, mu, sigma): # func2
        '''PDF of Log-Student's t distribution.'''
        y = np.log(x)
        return (1.0 / x) * st.t.pdf(y, df=nu, loc=mu, scale=sigma)

    def mixture_pdf(x, p11, p21, p22, p23, weight):
        '''PDF of mixture of two distributions.'''
        return weight * norm * func1(x, p11) + (1 - weight) * norm * func2(x, p21, p22, p23)
    
    func1 = exp_pdf
    func2 = logt_pdf

    # Initial parameter guesses
    p0 = np.array([25, 100, 1e-15, 1.2, 0.15])  # r, p21, p22, p23, weight

    # Fit histogram with this gamma mixture pdf
    bins = np.linspace(0, np.max(data), 400)
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_width = np.diff(bin_edges)[0]
    bin_centers = bin_edges[:-1] + bin_width / 2
    norm = 1/(len(bin_centers)*bin_width) # Ensure distribution sums to 1

    # Normalize counts to create a probability distribution for fitting
    normalized_counts = counts / counts.sum()

    if opt:
        try:
            p0, _ = curve_fit(
                mixture_pdf,
                bin_centers,
                normalized_counts,
                p0=p0,
                bounds=((len(p0) - 1)*[0] + [0.14], (len(p0) - 1)*[1000] + [1]),
                maxfev=10000
            )
            print('Fitted parameters:', p0)

        except RuntimeError:
            # Fallback: return initial guess if fitting fails
            print("Warning: Zero-inflated fitting did not converge. Using initial guess.")

    else:
        print('Using initial parameters:', p0)

    # Plot if axis provided
    if ax is not None:
        # Plot histogram
        ax.bar(bin_centers, normalized_counts, width = bin_width, label='Data', color='steelblue')

        # Plot fitted mixture
        x_plot = np.linspace(bin_centers[0], bin_centers[-1], 800)

        pdf_mix = mixture_pdf(x_plot, *p0)
        ax.plot(x_plot, pdf_mix, 'r-', linewidth=2, label='Mixture')

        # ax.set_xscale('log')
        ax.set_xlabel('Speed (rounded)')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.set_title(f'{func1.__name__.replace("_pdf","").capitalize()} + {func2.__name__.replace("_pdf","").capitalize()} mixture fit')

        return

def plot_speed_hists(ds, speed_names: list, threshes: list, exp_name: str, batch_num: int, fit_speed: bool = False): # Takes preprocessed xarray.Dataset as input

    # Look at all speeds for each ID
    _, axs = plt.subplots(len(speed_names), 1, figsize=(26, 12), sharey = True, sharex = True)
    
    # Ensure axs is always a list for consistent indexing
    if len(speed_names) == 1:
        axs = [axs]     

    for i in range(len(speed_names)):
        speed_vals = ds[f'v_{speed_names[i]}'].values.ravel()
        valid_speeds = speed_vals[np.isfinite(speed_vals)]
        
        # Fit mixed distributions if requested
        if fit_speed:
            fit_mixture_model(valid_speeds, ax=axs[i], opt = False)
        else:
            sns.histplot(valid_speeds, ax=axs[i], bins=500)
            axs[i].set_title(f'{speed_names[i].capitalize()} speed')

        axs[i].axvline(x=20, color='k', linestyle='--') # Plotting a line when speed is 20 because that is the cut-off in TRex settings for keeping identity (in px/frame)
        axs[i].axvline(x=threshes[batch_num], color='k', linestyle='--') # Plotting a line at the activity threshold for this batch
    
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Speed')
    plt.ylabel('Frequency')
    plt.tight_layout()

    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/speed_histograms.png')

def plot_tracklet_lengths_hist(ds_raw, speed_dict: dict, interp_dict: dict, radius: float, exp_name: str, batch_num: int, n_bins = 15):

    # Initiate plot
    _, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (8, 5))
    labels = [f'None', f'{round(radius)}']

    # Iterate over no interpolation, with interpolation
    for i in range(2):
        lengths_list = []
        # Iterate over all data, centered
        for j in range(2):
            print(2*i + j + 1)
            ds = preprocess_data(ds_raw, speed_dict, fill_gaps = bool(i), interp_dict = interp_dict, center_only = bool(j), radius = radius)

            # Broadcast id coordinate to the same shape as tracklet_id
            ids = xr.broadcast(ds['id'], ds['tracklet_id'])[0].values.ravel()  # shape (id, frame)

            # Compute tracklet lengths
            tids = ds['tracklet_id'].values.ravel()

            mask = ~np.isnan(tids) # Make mask of non nan tracklet ids
            valid_ids = ids[mask].astype(int)
            valid_tids = tids[mask].astype(int)

            # Count length of tracklets
            counts = np.bincount(np.ravel_multi_index((valid_ids, valid_tids), (int(valid_ids.max()+1), int(valid_tids.max()+1))))
            lengths_list.append(counts)

        max_length = np.max([np.max(lengths) for lengths in lengths_list])
        bins = np.logspace(0, np.log10(max_length), n_bins)

        for k, lengths in enumerate(lengths_list):
            sns.histplot(lengths, ax=ax[i,0], bins=bins, label=labels[k])
            
        counts1, _ = np.histogram(lengths_list[0], bins = bins)
        counts2, _ = np.histogram(lengths_list[1], bins = bins)
        widths = np.diff(bins)
        bar_width = np.median(widths / bins[:-1]) * bins[:-1]  # fraction of local bins
        ax[i,1].bar(bins[:-1], counts1 - counts2, color = 'gray', width = bar_width, align = 'edge')

        ax[i,0].set_ylabel('Count', fontsize = 15)
        ax[i,1].set_ylabel('Excluded', fontsize = 15)

        for k in range(2):
            ax[i,k].set_title('Interpolated' if i == 1 else 'Not interpolated', fontsize = 15)
            if i:
                ax[i,k].set_xlabel('Tracklet length (frames)', fontsize = 13)
            if not k:
                ax[i,k].legend(title = 'Radius')

    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/tracklet_length_hists/rad_{str(int(radius))}.png')

def plot_ang_speed(ds, t_slice, exp_name: str, batch_num: int):
    # Look at angular speeds over time for each ID

    _, axs = plt.subplots(2, 1, figsize=(26, 12), sharey = True)
    column_names = ['vtheta_high_ord', 'vtheta_sg']

    for id in range(2):
        for i in range(len(column_names)):
            fitted_speed = ds[f'{column_names[i]}'].isel(id=id, frame = t_slice)
            axs[i].plot(fitted_speed['frame'], fitted_speed, label=f'ID {id}')

    for i in range(len(column_names)):
        axs[i].set_xlabel('Frame')
        axs[i].set_ylabel(f'{column_names[i].capitalize()}')
        if len(ds['id'].values) < 10:
            axs[i].legend()
    plt.tight_layout()
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/angular_speed_over_time.png')

def plot_num_tracklets_over_time(ds, exp_name: str, batch_num: int):
    # Look at number of tracklets in any given instance to see how many individuals are missing. This assumes all are tracked at one point.

    perc_tracklets = np.mean(ds['missing'] != 1, axis = 0) # Includes interpolated points if they exist

    plt.figure(figsize=(12, 6))
    plt.plot(perc_tracklets['frame'], perc_tracklets)
    plt.xlabel('Frame')
    plt.ylabel('Individuals tracked (%)') # Assumes TReX at one point tracked all individuals
    plt.title('Tracked Individuals Over Time')
    plt.savefig(f'plots/{exp_name}/batch_{batch_num}/num_tracklets_over_time.png')