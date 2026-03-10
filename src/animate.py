import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os

cwd = os.getcwd()
if cwd.endswith('src'):
    from helper_fns import *
else:
    from src.helper_fns import *

'''_____________________________________________________ANIMATION FUNCTIONS____________________________________________________________'''

def animate_trajs_lined(ds: xr.Dataset, video_path: str, smooth_name: str, output_dir: str, buffer:int | None = 150, start_frame:int = 0, end_frame: int | None = None, interval:int =50, trail:int =10):
    """
    Animate trajectories from an xarray.Dataset over a spatial subsection of a video. buffer specifies the size of the window to use for the animation.
    start_frame and end_frame should be relative to the actual start of the batch (since subsequent batches won't start at 0).
    """

    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get appropriate frame slice for this batch
    frames = get_frame_slice(ds, start_frame, end_frame)

    # Define ids
    ids = ds.id.values

    # Subset dataset
    ds_sub = ds.sel(frame = frames)

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_axis_off()

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first video frame.")
    img_artist = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    lines = {i: ax.plot([], [], '-', c = 'white', lw=1)[0] for i in ids}
    for line in lines.values():
        line.set_zorder(2)
    img_artist.set_zorder(1)
    
    # Get video height and width to set spatial limits of video
    if buffer is not None:
        height, width = frame.shape[:2]
        ax.set_xlim([width/2 - buffer, width/2 + buffer])
        ax.set_ylim([height/2 - buffer, height/2 + buffer])
    
    # Initialize smooth_name
    if smooth_name != '':
        smooth_name = '_' + smooth_name

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
            lines[i].set_data(ds_recent['x' + smooth_name], ds_recent['y' + smooth_name])
        return [img_artist, *lines.values()]

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(output_dir + f'tracks_lined{smooth_name}.gif')
    cap.release()
    return

def animate_trajs_coloured(ds, video_path: str, smooth_name: str, output_dir: str, colours: xr.DataArray, cbar_name: str, start_frame:int = 0, end_frame:int | None = None, interval=50):
    """
    Scatter points with colours from 'colours' DataArray over video frames. Colours should be per-id, per-frame. Colours elements are assumed to be scalars, not tuples.
    start_frame and end_frame should be relative to the actual start of the batch (since subsequent batches won't start at 0).
    """

    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get appropriate frame slice for this batch
    frames = get_frame_slice(ds, start_frame, end_frame)

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
    
    # Initialize smooth_name
    if smooth_name != '':
        smooth_name = '_' + smooth_name

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

            x_vals = frame_data['x' + smooth_name].values
            y_vals = frame_data['y' + smooth_name].values
            
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
    ani.save(output_dir + f'tracks_coloured_{'_'.join(cbar_name.lower().split(' '))}.gif')
    cap.release()
    return

def animate_neighbours(ds: xr.DataArray, nbrs, interaction: str, inter_param, fid: int, video_path: str, smooth_name: str, output_dir: str, buffer = 150, start_frame:int = 0, end_frame:int | None = None, interval = 50):

    inter_param = str(inter_param)

    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get appropriate frame slice for this batch
    frames = get_frame_slice(ds, start_frame, end_frame)

    # Subset dataset
    ds_sub = ds.sel(frame = slice(start_frame, end_frame - 1))
    n_ids = len(ds_sub.id)

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_axis_off()
    scat = ax.scatter([], [], c=[], s=20, marker = 'x')
    scat.set_zorder(2) # Scatter is above image

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first video frame.")
    img_artist = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_artist.set_zorder(1)
    
    # Initialize smooth_name
    if smooth_name != '':
        smooth_name = '_' + smooth_name

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

        # Get positions for current frame
        frame_data = ds_sub.sel(frame=frame_num)

        # Find focal position
        fid_pos = np.array([frame_data.sel(id=fid)['x' + smooth_name].values, frame_data.sel(id=fid)['y' + smooth_name].values])

        # Find nbr positions
        offset_idx = int(idx*n_ids + fid)
        nbr_vals = nbrs[interaction][inter_param]['nbrs']['values']
        nbr_offsets = nbrs[interaction][inter_param]['nbrs']['offsets'].astype(int)
        nbr_ids = nbr_vals[nbr_offsets[offset_idx]:nbr_offsets[offset_idx+1]].astype(int)
        # print(nbr_ids)
        nbr_pos = np.array([[frame_data.sel(id=nbr_id)['x' + smooth_name].values, frame_data.sel(id=nbr_id)['y'+ smooth_name].values] for nbr_id in nbr_ids])
        
        # Set new axis limits centered around focal
        try:
            ax.set_xlim(fid_pos[0] - buffer, fid_pos[0] + buffer)
            ax.set_ylim(fid_pos[1] - buffer, fid_pos[1] + buffer)

            # Update scatter plot
            pts = np.vstack([fid_pos.reshape(1, 2), nbr_pos])
            scat.set_offsets(pts)
            colours = ['blue'] + ['red'] * len(nbr_ids)
            scat.set_color(colours)
        except:
            scat.set_offsets([np.nan, np.nan])
            scat.set_color([])

        if idx % 100 == 0:
            print(f'Processed frame {idx+1}/{len(frames)}')
        
        return [img_artist, scat]

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=interval, blit=False, repeat=False)
    ani.save(output_dir + 'nbrs_focal_{fid}_inter_{interaction}_{inter_param}{smooth_name}.gif')
    cap.release()
    return

def animate_focal_ego(ds:xr.DataArray, fid:int, video_path:str, smooth_name:str, output_dir:str, buffer:int = 150, start_frame:int = 0, end_frame:int | None = None, interval:int = 50):
    """ Animate focal individual in its egocentric frame of reference. Useful for verifying orientation calculations."""
    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get appropriate frame slice for this batch
    frames = get_frame_slice(ds, start_frame, end_frame)

    # Subset dataset
    ds_sub = ds.sel(id=fid, frame = frames)

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_axis_off()
    line = ax.plot([], [], '-', color = 'red', lw=1)[0]
    line.set_zorder(2)
    # scat = ax.scatter([], [], c='blue', s=20, marker = 'x')
    # scat.set_zorder(2) # Scatter is above image

    # Initialize video frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first video frame.")
    img_artist = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_artist.set_zorder(1)
    
    # Initialize smooth_name
    if smooth_name != '':
        smooth_name = '_' + smooth_name

    def init():
        # scat.set_offsets(np.empty((0, 2)))
        # return [img_artist, scat]
        line.set_data([], [])
        return [img_artist, line]
    
    def rotate_frame(frame, center, angle_rad):
        angle_deg = np.degrees(angle_rad)

        h, w = frame.shape[:2]

        # Rotation matrix around focal animal
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return rotated

    def update(idx):
        # Set frame number
        frame_num = frames[idx]

        # Get position of focal
        frame_data = ds_sub.sel(frame=frame_num)
        fid_pos = np.array([frame_data['x' + smooth_name].values, frame_data['y' + smooth_name].values])
        detected = np.sum(np.isnan(fid_pos)) == 0

        # Get direction of focal
        theta = frame_data[f'theta_{smooth_name}'].values
        theta = theta if detected else 0
        # theta = 0

        # Show frame of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            # return [img_artist, scat]
            return [img_artist, line]
        
        # Rotate frame to align with focal's heading
        center = (fid_pos[0], fid_pos[1]) if detected else (1920/2, 1920/2)
        # rotated_frame = rotate_frame(frame, center=center, angle_rad= -theta)
        rotated_frame = frame
        img_artist.set_data(cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB))
        
        # Set new axis limits centered around focal
        if detected:
            ax.set_xlim(fid_pos[0] - buffer, fid_pos[0] + buffer)
            ax.set_ylim(fid_pos[1] - buffer, fid_pos[1] + buffer)

            # Update scatter plot
            # scat.set_offsets(fid_pos.reshape(1, 2))
            line.set_data([fid_pos[0], fid_pos[0] + 20 * np.cos(theta)], [fid_pos[1], fid_pos[1] + 20 * np.sin(theta)])
        else:
            ax.set_xlim(0, 1920)
            ax.set_ylim(0, 1920)
            # scat.set_offsets([np.nan, np.nan])
            line.set_data([], [])

        if idx % 100 == 0:
            print(f'Processed frame {idx+1}/{len(frames)}')
        
        # return [img_artist, scat]
        return [img_artist, line]

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=interval, blit=False, repeat=False)
    ani.save(output_dir + 'ego_focal_{fid}{smooth_name}.gif')
    cap.release()
    return