import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
from pathlib import Path
import imageio
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from matplotlib.collections import PolyCollection
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from data_handling import load_cluster_stats_h5
from tqdm import tqdm
import gc
from cluster_analysis import find_reflections



cwd = os.getcwd()
from helper_fns import *

'''_____________________________________________________ANIMATION FUNCTIONS____________________________________________________________'''

def animate_trajs_lined(ds: xr.Dataset, video_path: str, smooth_name: str, output_dir: str, buffer:int | None = 150, start_frame:int = 0, end_frame: int | None = None, interval:int =50, trail:int =10, ds_fps: int = 5, vid_fps: int = 5):
    """
    Animate trajectories from an xarray.Dataset over a spatial subsection of a video. buffer specifies the size of the window to use for the animation.
    start_frame and end_frame should be relative to the actual start of the batch (since subsequent batches won't start at 0).
    """

    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get appropriate frame slice for this batch
    abs_frames, _ = get_frame_slice(ds, start_frame, end_frame)

    # Subset dataset
    ds_sub = ds.sel(frame = abs_frames)

    # Define x and y now for faster access during animation
    suffix = f"_{smooth_name}" if smooth_name else ""
    all_x = ds_sub[f'x{suffix}'].values # (n_ids, n_frames)
    all_y = ds_sub[f'y{suffix}'].values
    n_ids, n_ds_frames = all_x.shape

    # Find duration of animation in ms to determine how many steps are needed
    duration_sec = n_ds_frames / ds_fps
    total_anim_steps = int(duration_sec * (1000 / interval))

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_axis_off()

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first video frame.")
    img_artist = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_artist.set_zorder(1)

    lines = {i: ax.plot([], [], '-', c = 'red', lw=1, zorder = 2)[0] for i in range(n_ids)}
    # for line in lines.values():
    #     line.set_zorder(2)
    
    # Get video height and width to set spatial limits of video
    if buffer:
        height, width = frame.shape[:2]
        ax.set_xlim([width/2 - buffer, width/2 + buffer])
        ax.set_ylim([height/2 - buffer, height/2 + buffer])
    
    # Initialize smooth_name
    if smooth_name != '':
        smooth_name = '_' + smooth_name

    # Check when video updates are needed based on frame rate mismatches
    state = {'last_vid_idx': -1}

    def init():
        for line in lines.values():
            line.set_data([], [])
        return [img_artist, *lines.values()]

    def update(idx):
        # Calculate current time in the animation (s)
        current_time = (idx * interval) / 1000.0

        # Map time to source indices
        ds_idx = int(current_time * ds_fps)
        vid_idx = int(current_time * vid_fps) + int(start_frame*(vid_fps // ds_fps)) # Needs to be offset by the dataset's start_frame relative to time

        # Bounds check
        if ds_idx >= n_ds_frames:
            return [img_artist, *lines.values()]

        # A. Update video (only if the frame has changed)
        if vid_idx != state['last_vid_idx']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
            success, frame = cap.read()
            if success:
                img_artist.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                state['last_vid_idx'] = vid_idx

        # B. Update trails
        start_idx = max(0, ds_idx - trail)
        for i in range(n_ids):
            seg_x = all_x[i, start_idx : ds_idx + 1]
            seg_y = all_y[i, start_idx : ds_idx + 1]
            lines[i].set_data(seg_x, seg_y)

        if idx % 50 == 0:
            print(f'Anim Step {idx} | Time: {current_time:.2f}s | Vid Frame: {vid_idx}')

        return [img_artist, *lines.values()]

    ani = FuncAnimation(fig, update, frames=total_anim_steps, init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(output_dir + f'tracks_lined{smooth_name}.gif', writer = 'pillow', fps = 1000/interval)
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
    abs_frames, _ = get_frame_slice(ds, start_frame, end_frame)

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
        frame_num = abs_frames[idx]

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
            print(f'Processed frame {idx+1}/{len(abs_frames)}')
        
        return [img_artist, scat]

    ani = FuncAnimation(fig, update, frames=len(abs_frames), init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(output_dir + f'tracks_coloured_{"_".join(cbar_name.lower().split(' '))}.gif')
    cap.release()
    return

def animate_neighbours(ds: xr.DataArray, nbrs, interaction: str, inter_param, fid: int, video_path: str, smooth_name: str, output_dir: str, buffer = 150, start_frame:int = 0, end_frame:int | None = None, interval = 50):

    inter_param = str(inter_param)

    # Open video and count frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get appropriate frame slice for this batch
    abs_frames, _ = get_frame_slice(ds, start_frame, end_frame)

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
        frame_num = abs_frames[idx]

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
            print(f'Processed frame {idx+1}/{len(abs_frames)}')
        
        return [img_artist, scat]

    ani = FuncAnimation(fig, update, frames=len(abs_frames), init_func=init, interval=interval, blit=False, repeat=False)
    ani.save(output_dir + 'nbrs_focal_{fid}_inter_{interaction}_{inter_param}{smooth_name}.gif')
    cap.release()
    return

# def animate_focal_ego(ds:xr.DataArray, fid:int, video_path:str, smooth_name:str, output_dir:str, buffer:int = 150, start_frame:int = 0, end_frame:int | None = None, interval:int = 50, vid_frame_rate:int = 5, ds_frame_rate:int = 5):
#     """ Animate focal individual in its egocentric frame of reference. Useful for verifying orientation calculations."""
#     # Open video and count frames
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Cannot open video: {video_path}")

#     # Get appropriate frame slice for this batch
#     abs_frames, _ = get_frame_slice(ds, start_frame, end_frame)

#     # Subset dataset
#     ds_sub = ds.sel(id=fid, frame = abs_frames)

#     # Initialize figure
#     fig, ax = plt.subplots()
#     ax.set_axis_off()
#     line = ax.plot([], [], '-', color = 'red', lw=1)[0]
#     line.set_zorder(2)
#     # scat = ax.scatter([], [], c='blue', s=20, marker = 'x')
#     # scat.set_zorder(2) # Scatter is above image

#     # Initialize video frame
#     ret, frame = cap.read()
#     if not ret:
#         raise ValueError("Could not read first video frame.")
#     img_artist = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     img_artist.set_zorder(1)
    
#     # Initialize smooth_name
#     if smooth_name != '':
#         smooth_name = '_' + smooth_name

#     # Handle video and ds frame rate mismatches by only updating video every nth frame or vice versa
#     frame_mod = 1
#     ds_mod = 1
#     if vid_frame_rate != ds_frame_rate:
#         if vid_frame_rate < ds_frame_rate: # Update video frame less often than ds frame
#             frame_mod = int(np.ceil(ds_frame_rate / vid_frame_rate))

#         else: # Update video frame more often than ds frame
#             ds_mod = int(np.ceil(vid_frame_rate / ds_frame_rate))

#     def init():
#         # scat.set_offsets(np.empty((0, 2)))
#         # return [img_artist, scat]
#         line.set_data([], [])
#         return [img_artist, line]
    
#     def rotate_frame(frame, center, angle_rad):
#         angle_deg = np.degrees(angle_rad)

#         h, w = frame.shape[:2]

#         # Rotation matrix around focal animal
#         M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

#         rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#         return rotated

#     def update(idx):
        
#         # Set frame number for ds, accounting for potential frame rate mismatches
#         ds_frame_idx = np.floor(idx/ds_mod).astype(int)

#         # Get position of focal
#         frame_data = ds_sub.isel(frame=ds_frame_idx)
#         # fid_pos = np.array([frame_data['x' + smooth_name].values, frame_data['y' + smooth_name].values])
#         fid_pos = np.array([frame_data['x'].values, frame_data['y'].values])
#         detected = np.sum(np.isnan(fid_pos)) == 0

#         # Get direction of focal
#         theta = frame_data['theta' + smooth_name].values
#         theta = theta if detected else 0
#         # theta = 0

#         # Show frame of video
#         # cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
#         if not idx % frame_mod:
#             ret, frame = cap.read()

#             if ret:
#                 # Rotate frame to align with focal's heading
#                 center = (fid_pos[0], fid_pos[1]) if detected else (cap.shape[1]//2, cap.shape[0]//2)
#                 # rotated_frame = rotate_frame(frame, center=center, angle_rad= -theta)
#                 rotated_frame = frame
#                 img_artist.set_data(cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB))

#                 return [img_artist, line]
        
#         # Set new axis limits centered around focal
#         if detected:
#             ax.set_xlim(fid_pos[0] - buffer, fid_pos[0] + buffer)
#             ax.set_ylim(fid_pos[1] - buffer, fid_pos[1] + buffer)

#             # Update scatter plot
#             line_length = 20
#             line.set_data([fid_pos[0], fid_pos[0] + line_length * np.cos(theta)], [fid_pos[1], fid_pos[1] + line_length * np.sin(theta)])
#         else:
#             ax.set_xlim(0, cap.shape[1])
#             ax.set_ylim(0, cap.shape[0])
#             # scat.set_offsets([np.nan, np.nan])
#             line.set_data([], [])

#         if idx % 100 == 0:
#             print(f'Processed frame {idx+1}/{len(frames)}')
        
#         # return [img_artist, scat]
#         return [img_artist, line]

#     num_updates = ds_mod*len(frames)
#     ani = FuncAnimation(fig, update, frames=num_updates, init_func=init, interval=interval, blit=False, repeat=False)
#     ani.save(output_dir + f'ego_focal_{fid}{smooth_name}.gif')
#     cap.release()
#     return

def animate_orientations_fast(ds:xr.DataArray, smooth_name:str, output_dir:str, start_frame:int = 0, end_frame:int | None = None, interval:int = 50, trail:int =  10):
    
    # Get slice
    abs_frames, _ = get_frame_slice(ds, start_frame, end_frame)
    ds_sub = ds.sel(frame=abs_frames)
    
    suffix = f"_{smooth_name}" if smooth_name else ""
    
    # Pull data into np.arrays for fast access during animation
    all_x = ds_sub[f'x{suffix}'].values # (n_ids, n_frames)
    all_y = ds_sub[f'y{suffix}'].values
    all_theta = ds_sub[f'theta{suffix}'].values
    
    n_ids, n_frames = all_x.shape
    
    fig, ax = plt.subplots()
    ax.set_axis_off()

    # Set limits once based on the whole dataset
    ax.set_xlim(np.nanmin(all_x), np.nanmax(all_x))
    ax.set_ylim(np.nanmin(all_y), np.nanmax(all_y))

    # Initialize artists
    lines = [ax.plot([], [], '-', c='black', lw=1, alpha=0.6)[0] for _ in range(n_ids)]
    oris = [ax.plot([], [], '-', c='red', lw=2)[0] for _ in range(n_ids)]
    line_len = 20

    def init():
        return lines + oris

    def update(idx):
        # Determine the trail window using simple NumPy slicing
        start_idx = max(0, idx - trail)
        
        for i in range(n_ids):
            # Update trail lines
            seg_x = all_x[i, start_idx : idx + 1]
            seg_y = all_y[i, start_idx : idx + 1]
            lines[i].set_data(seg_x, seg_y)
            
            # Update orientation vectors (current frame only)
            curr_x = all_x[i, idx]
            
            if not np.isnan(curr_x and idx):
                curr_y = all_y[i, idx]
                curr_t = all_theta[i, idx]
                oris[i].set_data([curr_x, curr_x + line_len * np.cos(curr_t)], [curr_y, curr_y + line_len * np.sin(curr_t)])
            else:
                oris[i].set_data([], [])

        if idx % 100 == 0:
            print(f'Frame {idx}/{n_frames}')
            
        return lines + oris

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(f"{output_dir}/orientations{suffix}.gif", writer='pillow')
    plt.close(fig)

def animate_state(ds:xr.DataArray, x_var:str, y_var:str, output_dir:str, start_frame:int = 0, end_frame:int | None = None, interval:int = 50, trail:int =  10):
    # Get slice
    abs_frames, _ = get_frame_slice(ds, start_frame, end_frame)
    ds_sub = ds.sel(frame=abs_frames)
    
    # Pull data into np.arrays for fast access during animation
    all_x = ds_sub[x_var].values # (n_ids, n_frames)
    all_y = ds_sub[y_var].values
    
    n_ids, n_frames = all_x.shape
    
    fig, ax = plt.subplots()
    ax.set_axis_off()

    # Set limits once based on the whole dataset
    ax.set_xlim(np.nanmin(all_x), np.nanmax(all_x))
    ax.set_ylim(np.nanmin(all_y), np.nanmax(all_y))
    ax.set_xlabel(x_var, fontsize = 17)
    ax.set_ylabel(y_var, fontsize = 17)

    # Initialize artists
    lines = [ax.plot([], [], '-', c='black', lw=1, alpha=0.6)[0] for _ in range(n_ids)]

    def init():
        return lines

    def update(idx):
        # Determine the trail window using simple NumPy slicing
        start_idx = max(0, idx - trail)
        
        for i in range(n_ids):
            # Update trail lines
            seg_x = all_x[i, start_idx : idx + 1]
            seg_y = all_y[i, start_idx : idx + 1]
            lines[i].set_data(seg_x, seg_y)

        if idx % 100 == 0:
            print(f'Frame {idx}/{n_frames}')
            
        return lines

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=interval, blit=True, repeat=False)
    ani.save(output_dir + f'states_{x_var}_{y_var}.gif', writer='pillow')
    plt.close(fig)

def animate_zoom_out(ds_kp:xr.Dataset, video_path:str, output_dir:str, start_frame:int = 0, end_frame:int | None = None,
                     start_zoom:float=0.2, center:tuple=(3500, 3500), full_size:int=7000, fps:int = 5):
    """
    video_path: path to your .mp4 or .avi
    points_by_frame: list of (N, 2) arrays
    lines_by_frame: list of lists containing (point1_idx, point2_idx)
    start_zoom: 0.1 is 10% of the image (very zoomed in), 1.0 is full frame.
    """
    head_x, head_y = ds_kp['head_x'].values, ds_kp['head_y'].values
    tail_x, tail_y = ds_kp['tail_x'].values, ds_kp['tail_y'].values
    mask = (~np.isnan(head_x)) & (~np.isnan(head_y)) & (~np.isnan(tail_x)) & (~np.isnan(tail_y)) # (n_frames, max_ids)
    mask = mask & (np.hypot(head_x - center[0], head_y - center[1]) < full_size/2) & (np.hypot(tail_x - center[0], tail_y - center[1]) < full_size/2)
    
    heads_by_frame = []
    tails_by_frame = []

    if end_frame:
        max_f = min(head_x.shape[0], end_frame)
    else:
        max_f = head_x.shape[0]

    for f in range(start_frame, max_f):
        heads = np.array([head_x[f, mask[f]].T, head_y[f, mask[f]].T])
        tails = np.array([tail_x[f, mask[f]].T, tail_y[f, mask[f]].T])
        heads_by_frame.append(heads)
        tails_by_frame.append(tails)

    frames = np.arange(start_frame, max_f)
    total_frames = len(frames)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off') # Remove axes for that clean look
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # Get sorted list of images
    img_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.jpeg'))])

    # Initialize with the first image
    first_img = cv2.imread(os.path.join(video_path, img_files[0]))
    first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    im = ax.imshow(first_img)
    im.set_zorder(1)

    # Initialize plot elements
    scat_h = ax.scatter([], [], c='cyan', s=5, edgecolors='white', linewidths=0.5)
    scat_t = ax.scatter([], [], c='orange', s=5, edgecolors='white', linewidths=0.5)
    scat_h.set_zorder(3)
    scat_t.set_zorder(3)
    lines = [ax.plot([], [], color='white', alpha=0.5, lw=1)[0] for _ in range(head_x.shape[1])]
    for line in lines:
        line.set_zorder(2)

    def update(frame_idx):
        img_path = os.path.join(video_path, img_files[frame_idx])

        frame = cv2.imread(img_path)
        if frame is None:
            print("Frame not read.")
            return [im, scat_h, scat_t] + lines
        
        # 1. Update Image and Data
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im.set_data(frame_rgb)
        
        # Update Scatters (Positions)
        hs = heads_by_frame[frame_idx].T
        ts = tails_by_frame[frame_idx].T
        scat_h.set_offsets(hs)
        scat_t.set_offsets(ts)
        
        # 2. Calculate Zoom (Linear interpolation)
        # Current scale goes from start_zoom to 1.0
        current_scale = start_zoom + (1.0 - start_zoom) * (frame_idx / total_frames)
        half_width = (full_size * current_scale) / 2
        
        ax.set_xlim(center[0] - half_width, center[0] + half_width)
        ax.set_ylim(center[1] + half_width, center[1] - half_width) # Invert Y for image coords

        # Reset all lines to be invisible first (to handle flickering/disappearing IDs)
        for l in lines:
            l.set_data([], [])

        # Update only the lines that exist for this frame
        for i, (h, t) in enumerate(zip(hs, ts)):
            # Extract x and y for just THIS head-tail pair
            hx, hy = h[0], h[1]
            tx, ty = t[0], t[1]
            
            lines[i].set_data([hx, tx], [hy, ty])

        return [im, scat_h, scat_t] + lines


    # Create Animation
    ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)
    
    ani.save(output_dir + 'kps_zoom_out_complete.gif', writer='pillow', fps=fps)
    plt.close()

def animate_phase(ds_kp:xr.Dataset, output_dir:str, frames_dir:str, x_var:str, y_var:str, labels:list[str], x_factor:float = 1, y_factor:float = 1, start_frame:int = 0, end_frame:int | None = None, vid_length:int = 300, fps:int = 5, subsample:int = 1, gridsize:int = 25, vmax:int = 125):
    ''' Animate phase space by plotting hexbin heatmaps of x_var vs y_var across frames. Subsampling can be applied to speed up animation.'''

    # --- Load frame file paths ---
    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
    if len(frame_paths) == 0:
        raise ValueError(f"No JPEGs found in {frames_dir}")
    n_video_frames = len(frame_paths)

    # Get rescaled valid values for x and y
    valid_mask = (~np.isnan(ds_kp[x_var].values)) & (~np.isnan(ds_kp[y_var].values)) & (ds_kp[x_var].values < np.nanquantile(ds_kp[x_var].values, 0.999))
    x = ds_kp[x_var].values*x_factor # Rescale if necessary, (n_frames, n_max_ids)
    y = ds_kp[y_var].values*y_factor # Rescale if necessary, (n_frames, n_max_ids)

    # Get relative and absolute frames
    abs_frames, ds_indcs = get_frame_slice(ds_kp, start_frame, end_frame, subsample)

    # Ensure frames are within available JPEG range
    abs_frames = abs_frames[abs_frames < n_video_frames]
    ds_indcs = ds_indcs[abs_frames < n_video_frames]
    
    # Get valid indices for each frame included
    valid_indices = {f: np.where(valid_mask[f])[0] for f in ds_indcs}

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Load first image
    first_img = cv2.imread(str(frame_paths[abs_frames[0]]))
    if first_img is None:
        raise ValueError(f"Could not read {frame_paths[abs_frames[0]]}")
    img = ax[0].imshow(cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB))
    
    # Initialize the hexbin heatmap
    hb = ax[1].hexbin(x[ds_indcs[0], valid_mask[ds_indcs[0]]], y[ds_indcs[0], valid_mask[ds_indcs[0]]], gridsize=gridsize, cmap='viridis', mincnt=1, vmin = 1, vmax = vmax) #, bins='log')
    
    # Add Colorbar
    cb = plt.colorbar(hb, ax=ax[1])
    cb.set_label('Occupancy', size=12)
    
    # Labeling the 2D space
    ax[1].set_xlabel(labels[0], fontsize=17)
    ax[1].set_ylabel(labels[1], fontsize=17)

    plt.grid(True, linestyle='--', alpha=0.3)

    # Define general index to get proper absolute frame and ds index
    idcs = np.arange(0, len(ds_indcs))

    def update(idx):
        nonlocal hb

        hb.remove()

        # Load JPEG
        frame_path = frame_paths[abs_frames[idx]] # Grab correct frame based on initial subsample of dataset (frame_factor)
        img_frame = cv2.imread(str(frame_path))
        if img_frame is None:
            print('Skipped.')
            return [img, hb]
        img.set_data(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
        
        # Find valid x and y values
        i = ds_indcs[idx] # Current ds index
        x_vals = x[i, valid_indices[ds_indcs[i]]]
        y_vals = y[i, valid_indices[ds_indcs[i]]]
        hb = ax[1].hexbin(x_vals, y_vals, gridsize=gridsize, cmap='viridis', mincnt=1, vmin = 1, vmax = vmax)

        if (idx + 1) % (round(len(idcs)/20)) == 0:
            print(f'Processed frame {idx + 1}/{len(idcs)}')

        plt.suptitle(f'Frame {abs_frames[idx]}', fontsize=17)

        return [img, hb]
    
    # Create Animation
    ani = FuncAnimation(fig, update, frames=idcs, interval=vid_length/fps, blit=False)
    
    ani.save(output_dir + f'vid_phase_{x_var}_{y_var}_start_{abs_frames[0]}_end_{abs_frames[-1]}_subsample_{round(np.diff(abs_frames)[0])}.gif', writer='pillow', fps=fps)
    plt.close()

def jpeg_sequence_to_mp4(frames_dir, output_path, start_frame=0, end_frame=None, step=1, fps=5):
    
    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))

    if end_frame is None:
        end_frame = len(frame_paths)

    selected = frame_paths[start_frame:end_frame:step]

    with imageio.get_writer(output_path + '20230329_clip.mp4', mode='I', fps=fps) as writer:
        for i, p in enumerate(selected):
            img = imageio.imread(p)  # loads ONE frame only
            writer.append_data(img)  # immediately writes it

            if i % 100 == 0:
                print(f"Processed {i}/{len(selected)}")

    print("Done.")

def animate_param_overlay(ds_kp:xr.Dataset, output_dir:str, frames_dir:str, param:str, mincnt: int, locality:str, title:str, batch_num:int, param_factor:float = 1, start_frame:int = 0, end_frame:int | None = None, vid_length:int = 300, fps:int = 5, subsample:int = 1, frame_factor:int = 1, gridsize:int = 35):
    ''' Animate phase space by plotting hexbin heatmaps of param over the video. Locality (used to update file name) represents range used for quantifying param. Subsampling can be applied to speed up animation. frame_factor is used to display the actual frame number in the title when subsampling (e.g. if subsample=10 and frame_factor=1, title will show frames 0, 10, 20 etc. If frame_factor=10, title will show frames 0, 100, 200 etc.).
    This is because the ds may be already subsampled (e.g. every 10th frame) and we want the title to reflect the actual frame number in the original video, not the index in the subsampled dataset.'''

    # --- Load frame file paths ---
    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
    if len(frame_paths) == 0:
        raise ValueError(f"No JPEGs found in {frames_dir}")
    n_video_frames = len(frame_paths)

    # Get x, y, param, and valid mask
    x = ds_kp['centroid_x'].values # (n_frames, n_max_ids)
    y = ds_kp['centroid_y'].values # (n_frames, n_max_ids)
    valid_mask = (~np.isnan(ds_kp[param].values)) & (~np.isnan(x)) & (~np.isnan(y))
    ds_frames = ds_kp.coords['frame'].values
    p = ds_kp[param].values*param_factor # Rescale if necessary, (n_frames, n_max_ids)
    vmax = np.nanmax(p)

    # Get frames
    if not end_frame:
        end_frame = x.shape[0]
    frames = np.arange(start_frame, end_frame, subsample)

    # Ensure frames are within available JPEG range
    frames = frames[frames < n_video_frames]
    
    # Get valid indices for each frame included
    valid_indices = {f: np.where(valid_mask[f - ds_frames[0]])[0] for f in frames}

    fig = plt.figure(figsize=(7, 7))

    # Load first image
    first_img = cv2.imread(str(frame_paths[frames[0]]))
    if first_img is None:
        raise ValueError(f"Could not read {frame_paths[frames[0]]}")
    img = plt.imshow(cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB))
    img.set_zorder(1)
    
    # Initialize the hexbin heatmap
    hb = plt.hexbin(x[0, valid_mask[0]], y[0, valid_mask[0]], C=p[0, valid_mask[0]], reduce_C_function=np.mean, gridsize=gridsize, cmap='viridis', mincnt=mincnt, vmin = 0, vmax = vmax, alpha = 0.3) #, bins='log')
    hb.set_zorder(2)
    
    # Add Colorbar
    cb = plt.colorbar(hb)
    cb.set_label(title, size=12)

    plt.grid(True, linestyle='--', alpha=0.3)

    def update(frame):
        nonlocal hb

        hb.remove()

        # Load JPEG
        frame_path = frame_paths[frame_factor*frame] # Grab correct frame based on initial subsample of dataset (frame_factor)
        img_frame = cv2.imread(str(frame_path))
        if img_frame is None:
            print('Skipped.')
            return [img, hb]
        img.set_data(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
        
        idx = valid_indices[frame] # Indices of valid points for this frame
        x_vals = x[frame - ds_frames[0], idx]
        y_vals = y[frame - ds_frames[0], idx]
        p_vals = p[frame - ds_frames[0], idx]
        hb = plt.hexbin(x_vals, y_vals, C=p_vals, reduce_C_function=np.mean, gridsize=gridsize, cmap='viridis', mincnt=mincnt, vmin = 0, vmax = vmax, alpha = 0.3)

        if frame % int(subsample*500) == 0:
            print(f'Processed frame {frame_factor*(frame+1)}/{frame_factor*end_frame}')

        plt.suptitle(f'Frame {frame_factor*frame}', fontsize=17)

        return [img, hb]
    
    # Create Animation
    ani = FuncAnimation(fig, update, frames=frames, interval=vid_length/fps, blit=False)
    
    ani.save(output_dir + f'overlay_{title}_locality_{locality}_batch_{batch_num}_{round(subsample/fps, 1)}Hz.gif', writer='pillow', fps=fps)
    plt.close()


def plot_voronoi_overlay(pos_valid_t:np.ndarray, arena_center: np.ndarray, arena_radius: float, values:np.ndarray, ax=None, cmap="viridis", alpha=0.5):

    # Compute Voronoi tessellation
    vor = Voronoi(pos_valid_t)
    
    if ax is None:
        fig, ax = plt.subplots()

    verts = []
    colours = []
    areas = []
    excluded_indcs = []

    # Iterate over each point
    for point_idx, val in enumerate(values):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]

        # Exclude regions with no points
        if len(region) == 0:
            continue

        # Clip and order vertices
        vertices = vor.vertices[np.array(region)[(np.array(region) != -1).astype(bool)].astype(int)]
        poly = Polygon(clip_voronoi_region(vertices, arena_center, arena_radius, 0.05))

        if poly.is_empty:
            excluded_indcs.append(point_idx)
            continue

        # Append properties to lists (ordered)
        verts.append(np.array(poly.exterior.coords))
        colours.append(val)
        areas.append(poly.area)          

    coll = PolyCollection(verts, array=np.array(colours), cmap=cmap, edgecolor="k", linewidth=0.3, alpha=alpha)

    ax.add_collection(coll)
    ax.autoscale()
    ax.set_xlim([0, 7000])
    ax.set_ylim([0, 7000])

    # Compute neighbour relationships from Voronoi ridges
    nbrs = {i: set() for i in range(len(pos_valid_t))}
    for i1, i2 in vor.ridge_points:
        nbrs[i1].add(i2)
        nbrs[i2].add(i1)
    indcs = [sorted(list(v)) for v in nbrs.values()]
    included = np.ones(len(nbrs)).astype('bool')
    included[np.array(excluded_indcs)] = False
    num_nbrs = np.array([len(nbrs) for nbrs in indcs])[included]

    return ax, coll, areas, colours, num_nbrs

def interactive_voronoi_overlay(ds:xr.Dataset, param:str, output_dir:str, arena_center: np.ndarray, arena_radius: float, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1, n_bins:int = 10, cmap:str = 'viridis', fs:float = 5):

    # Define position array to save time from accessing ds
    positions = np.stack([ds['centroid_x'], ds['centroid_y']]) # (2, n_frames, max_ids)
    z = ds[param].values # (n_frames, max_ids)

    # Set up colour bins
    colourscale = pc.sample_colorscale(cmap, n_bins)
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    # if z_max > 10*np.nanstd(z) + np.nanmean(z): # If maximum is an outlier, replace with more reasonable maximum
    #     z_max = 5*np.nanstd(z) + np.nanmean(z)
    bins = np.linspace(z_min, z_max, n_bins + 1)

    # Filter out detections outside of arena
    dist_from_center = np.sqrt((positions[0,:,:] - arena_center[0])**2 + (positions[1,:,:] - arena_center[1])**2) # (n_frames, max_ids)
    outside_arena_mask = dist_from_center > arena_radius

    # Define valid mask
    valid_mask = (~np.isnan(positions).any(axis = 0)) & (~np.isnan(z)) & (~outside_arena_mask) # (n_frames, max_ids)

    # Get frames
    abs_frames, ds_idcs = get_frame_slice(ds, start_frame, end_frame, subsample)

    fig = go.Figure()

    # Track how many traces belong to each frame
    traces_per_frame = []
    areas = []

    for f in ds_idcs:
        # Filter valid positions and z values
        valid_positions_t = positions[:, f, valid_mask[f]]  # (2, n_ids)
        valid_z_t = z[f, valid_mask[f]]

        if valid_positions_t.shape[1] < 3:
            traces_per_frame.append(0)
            continue

        # Compute voronoi tessellation
        vor = Voronoi(valid_positions_t.T)

        bin_xs = [[] for _ in range(n_bins)]
        bin_ys = [[] for _ in range(n_bins)]

        # Iterate over each point
        for point_idx, z_val in enumerate(valid_z_t):
            region_idx = vor.point_region[point_idx]
            region = vor.regions[region_idx]

            # Exclude regions with no points
            if len(region) == 0:
                continue

            # Clip and order vertices
            vertices = vor.vertices[np.array(region)[(np.array(region) != -1).astype(bool)].astype(int)]
            poly = Polygon(clip_voronoi_region(vertices, arena_center, arena_radius, 0.05))

            if poly.is_empty:
                continue

            areas.append(poly.area)

            bin_idx = min(np.searchsorted(bins, z_val) - 1, n_bins - 1)
            pos = np.array(poly.exterior.coords)
            bin_xs[bin_idx].extend(pos[:, 0].tolist() + [None])
            bin_ys[bin_idx].extend(pos[:, 1].tolist() + [None])

        for b in range(n_bins):
            fig.add_trace(go.Scatter(x=bin_xs[b], y=bin_ys[b], fill='toself', mode='lines', fillcolor=colourscale[b], line=dict(width=0.5, color='rgba(0,0,0,0.3)'), visible=False, showlegend=False))
            
        traces_per_frame.append(n_bins)

    # Add arena outline
    theta = np.linspace(0, 2*np.pi, 300)
    fig.add_trace(go.Scatter(x=arena_center[0] + arena_radius*np.cos(theta), y=arena_center[1] + arena_radius*np.sin(theta), mode='lines', line=dict(color='black'), showlegend=False))
    circle_trace_idx = len(fig.data) - 1

    # Add dummy scatter points for colour bar
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(colorscale=cmap, cmin=z_min, cmax=z_max, color=[z_min], colorbar=dict(title=dict(text=param, side='right'),
                tickvals=np.linspace(z_min, z_max, 6).tolist(), ticktext=[f'{v:.2f}' for v in np.linspace(z_min, z_max, 6)], thickness=20, len=0.75), showscale=True), showlegend=False, visible=True))
    colourbar_trace_idx = len(fig.data) - 1

    # Build slider steps — each step makes exactly its frame's traces visible
    slider_steps = []
    total_traces = sum(traces_per_frame) + 2 # Add 2 for circle and colour bar trace
    cumulative = 0

    for i, count in enumerate(traces_per_frame):
        visibility = [False] * total_traces
        for j in range(cumulative, cumulative + count):
            visibility[j] = True
        visibility[circle_trace_idx] = True  # always show circle
        visibility[colourbar_trace_idx] = True

        slider_steps.append(dict(
            method='restyle',
            args=[{'visible': visibility}],
            label=str(abs_frames[i]),  # display actual frame number
        ))
        cumulative += count

    # Make first frame visible by default
    if traces_per_frame[0] > 0:
        for i in range(traces_per_frame[0]):
            fig.data[i].visible = True

    # Add slider and fix axes
    fig.update_layout(sliders=[dict(active=0, steps=slider_steps, currentvalue=dict(prefix='Frame: ', visible=True), pad=dict(t=50))],
                      xaxis=dict(range=[arena_center[0] - arena_radius * 1.1, arena_center[0] + arena_radius * 1.1], constrain='domain'),
                      yaxis=dict(range=[arena_center[1] - arena_radius * 1.1, arena_center[1] + arena_radius * 1.1],
                      scaleanchor='x', scaleratio=1, constrain='domain'),
                      title='Voronoi Tessellation Over Time')
    

    output_path = f"{output_dir}voronoi_sliders/voronoi_slider_{param}_{abs_frames[0]}_{abs_frames[-1]}_fs_{fs/round(np.diff(abs_frames)[0])}.html"
    fig.write_html(output_path)
    print(f"Saved to {output_path}.")

    return

def interactive_voronoi_distributions(ds:xr.Dataset, output_dir:str, arena_center:np.ndarray, arena_radius:float, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1, n_bins: list[int] = [40, None, 30], fs:float = 5, density_factor:float = 1):

    # Define position array to save time from accessing ds
    positions = np.stack([ds['centroid_x'], ds['centroid_y']]) # (2, n_frames, max_ids)
    densities = ds['density_voronoi_None'].values*density_factor # (n_frames, max_ids)

    # Filter out detections outside of arena
    dist_from_center = np.sqrt((positions[0,:,:] - arena_center[0])**2 + (positions[1,:,:] - arena_center[1])**2) # (n_frames, max_ids)
    outside_arena_mask = dist_from_center > arena_radius

    # Define valid mask
    valid_mask = (~np.isnan(positions).any(axis = 0)) & (~np.isnan(densities)) & (~outside_arena_mask) # (n_frames, max_ids)

    # Get frames
    abs_frames, ds_idcs = get_frame_slice(ds, start_frame, end_frame, subsample)

    # Name parameters
    titles = ['Voronoi areas', 'Voronoi neighbours', 'Voronoi densities']
    xlabels = ['Area (㎡)', 'Number of neighbours (n)', 'Density (n/㎡)']
    n_params = len(titles)

    fig = make_subplots(rows=1, cols= 3, subplot_titles=titles, horizontal_spacing=0.08, vertical_spacing=0.12)

    all_areas = []
    all_nbr_counts = []

    max_area = 0
    max_nbrs = 0
    max_density = np.quantile(densities[valid_mask], 0.999) # Some crazy outliers need to be excluded

    # Iterate over each frame to collect area and nbr count data
    for ds_idx in ds_idcs:

        # Filter valid positions and z values
        valid_positions_t = positions[:, ds_idx, valid_mask[ds_idx]].T # (n_ids, 2)

        if valid_positions_t.shape[0] < 3:
            continue

        # Compute voronoi tessellation
        vor = Voronoi(valid_positions_t)

        # Now we don't care about the order of the polygons so we can compute the polygons in one line
        polys = [Polygon(clip_voronoi_region(vor.vertices[np.array(region)[(np.array(region) != -1).astype(bool)].astype(int)], arena_center, arena_radius)) for region in vor.regions]

        # Get areas of each polygon
        areas = [poly.area/density_factor for poly in polys]

        # Compute neighbour relationships from Voronoi ridges
        nbrs = {i: set() for i in range(len(valid_positions_t))}
        for i1, i2 in vor.ridge_points:
            nbrs[i1].add(i2)
            nbrs[i2].add(i1)
        indcs = [sorted(list(v)) for v in nbrs.values()]
        num_nbrs = np.array([len(nbrs) for nbrs in indcs])

        # Append data to lists
        all_areas.append(areas)
        all_nbr_counts.append(num_nbrs)

        # Update maxima
        if np.max(areas) > max_area:
            max_area = np.max(areas)
        if np.max(num_nbrs) > max_nbrs:
            max_nbrs = np.max(num_nbrs)

    # Add ALL traces upfront (one per param per frame), only the first frame visible
    maxes = [max_area, max_nbrs, max_density]
    maxes_counts = [0, 0, 0]
    
    if not n_bins[1]: # If nbr bins not specified
        n_bins[1] = max_nbrs

    for f_idx, (frame_num, ds_idx) in enumerate(zip(abs_frames, ds_idcs)):
        # Iterate over params to get each histogram
        params = [all_areas[f_idx], all_nbr_counts[f_idx], densities[ds_idx, valid_mask[ds_idx]]]

        for i, param in enumerate(params):
            if i == 1:
                bin_edges = np.arange(0, max_nbrs + 1)
                counts, _ = np.histogram(param, bins = bin_edges, range = (0, maxes[i]))
            else:
                counts, bin_edges = np.histogram(param, bins=n_bins[i], range=(0, maxes[i]))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            if max(counts) > maxes_counts[i]:
                maxes_counts[i] = max(counts)

            fig.add_trace(go.Bar(x=bin_centers, y=counts,
                                 name=titles[i],
                                 showlegend=False,
                                 visible=(f_idx == 0),  # only first frame visible initially
                                 marker_color='steelblue',
                                 marker_line_width=0), row=1, col=i + 1)

    # Each slider step sets visibility: only the n_params traces for that frame are True
    total_traces = positions.shape[1] * n_params
    steps = []
    for f_idx, frame_num in enumerate(abs_frames):
        visibility = [False] * total_traces
        for i in range(n_params):
            visibility[f_idx * n_params + i] = True

        steps.append({
            'method': 'restyle',
            'label': str(frame_num),
            'args': [{'visible': visibility}],
        })

    sliders = [{
        'active': 0,
        'currentvalue': {'prefix': 'Frame: ', 'visible': True, 'xanchor': 'center'},
        'pad': {'t': 50},
        'steps': steps,
    }]

    fig.update_layout(
        sliders=sliders,
        bargap=0.02,
        title_text='Distributions by Frame')

    for i in range(n_params):
        fig.update_xaxes(range=[0, maxes[i]], row=1, col=i + 1, title_text = xlabels[i])
        fig.update_yaxes(range=[0, maxes_counts[i]], row=1, col=i + 1, title_text = 'Counts' if not i else '')

    output_path = os.path.join(output_dir, f'voronoi_sliders/vor_distributions_{abs_frames[0]}_{abs_frames[-1]}_fs_{fs/round(np.diff(abs_frames)[0])}.html')
    fig.write_html(output_path)
    print(f'Saved to {output_path}')
    return fig

def interactive_cluster_analysis(input_path: str, min_obs:int = 5, max_layers:int | None = None, n_bins:int = 21):

    # Load data
    data = load_cluster_stats_h5(input_path)
    
    # Relative frames in integers
    rel_frames = np.arange(-1*round((len(data) - 1)/2), round((len(data) - 1)/2) + 1)

    # Initialize extrema dictionaries and parameter strings
    all_params = data['0'].keys()
    max_x = {p: 0 for p in all_params}
    max_x['medPols'], max_x['meanThetas'], max_x['p_by_layer'], max_x['p_from_edge'] = 1, np.pi, 1, 1
    min_x = {p: 0 for p in all_params}
    min_x['ns'], min_x['areas'], min_x['meanThetas'] = 2, np.inf, -np.pi

    # Iterate over frames to collect maxima
    for rel_idx in rel_frames:
        
        for param in ['ns', 'areas', 'varPols', 'medDs', 'varDs', 'varThetas', 'd_by_layer', 'd_from_edge']:

            # Update maximum overall value
            vals = data[str(rel_idx)][param]

            if param == 'areas':
                min_val = np.nanquantile(vals, 0.01)
                if min_val < min_x[param]:
                    min_x[param] = min_val

            if type(vals) == dict: # If layers
                max_val = np.nanquantile(vals['data'], 0.999) # Ignoring crazy outliers
            else:
                max_val = np.nanquantile(vals, 0.999)

            if max_val > max_x[param]:
                max_x[param] = float(max_val)

        del vals

    titles = ['N distribution', 'Area distribution', 'Mean θ distribution', 'Area vs N', 'Med. pol & density vs N', 'Var. pol & density vs N', 'θ vs N', 'Pol vs density',
              'Polarization by layer (centre)', 'Polarization by layer (edge)', 'Density by layer (centre)', 'Density by layer (edge)']

    # Initialize figure
    fig = make_subplots(rows=3, cols=4, subplot_titles=titles, horizontal_spacing=0.16, vertical_spacing=0.12,
                        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                               [{"secondary_y": True}, {"secondary_y": True}, {"type": "scatter"}, {"type": "scatter"}],
                               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]])

    # Histograms
    hist_params = ['ns', 'areas', 'meanThetas']
    max_counts = {p: 0 for p in hist_params}

    # Scatterplots (Areas vs N, Pols/dens vs N, Std pol/den vs N, theta vs N, pol vs den)
    scatter_x_params = ['ns', 'ns', 'ns', 'ns', 'medDs']
    scatter_y_params = ['areas', ['medPols', 'medDs'], ['varPols', 'varDs'], 'meanThetas', 'medPols']
    scatter_rows = [1, 2, 2, 2, 2]
    scatter_cols = [4, 1, 2, 3, 4]

    # Layer plots
    layer_params = ['p_by_layer', 'p_from_edge', 'd_by_layer', 'd_from_edge']

    # Dictionary for axis labels
    label_dict = {'ns': 'N', 'areas': 'Area (㎡)', 'medPols': 'Med. polarization', 'varPols': 'Var. polarization', 'medDs': 'Med. density (n/㎡)',
                  'varDs': 'Var. density (n/㎡)', 'meanThetas': 'Avg. θ (rad)', 'varThetas': 'Var. θ (rad)', 'p_by_layer': 'Med. polarization',
                  'p_from_edge': 'Med. polarization', 'd_by_layer': 'Med. density (n/㎡)', 'd_from_edge': 'Med. density (n/㎡)'}

    def plot_layers(csr_dict:dict[str: np.ndarray[float]], min_obs:int, max_layers:int | None, rel_idx:int):

        # Unpack vals and idcs arrays
        vals = csr_dict['data']
        idcs = csr_dict['indptr']

        # Convert csr to (n_clusters, n_layers) matrix
        layers = np.full((len(idcs) - 1, np.max(np.diff(idcs))), np.nan, dtype=np.float32)

        for i in range(len(idcs[:-1])):
            layers[i,:(idcs[i+1] - idcs[i])] = vals[idcs[i]:idcs[i+1]]

        # Find appropriate cut-off of layers using number of observations or hard cut-off
        if max_layers is None:
            # Count number of finite observations
            n_obs = np.sum(np.isfinite(layers), axis = 0)
            cutoff = np.where(n_obs < min_obs)[0][0]

        else:
            cutoff = max_layers
            
        # Generate None separated xs and ys lists
        xs = []
        ys = []
        for i in range(len(idcs[:-1])):
            n = min((idcs[i+1] - idcs[i]), cutoff)
            xs.extend(range(0, n))
            ys.extend(vals[idcs[i]:(idcs[i] + n)].tolist())

            # None separator - breaks the line between clusters
            xs.append(None)
            ys.append(None)

        del idcs, vals

        indivs_trace = go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='rgba(55,138,221,0.2)', width=0.8),
                                  showlegend=False, hoverinfo='skip', connectgaps=False, # Ensures no bridging across Nones
                                  visible=(rel_idx == 0)) 
        
        # Compute mean line
        means_x = np.arange(0, cutoff)
        means_y = np.nanmean(layers[:,:cutoff], axis = 0)

        mean_trace = go.Scatter(x=means_x, y=means_y, mode='lines', line=dict(color='#185FA5', width=2),
                                showlegend=False, connectgaps=False, visible=(rel_idx == 0))

        return indivs_trace, mean_trace

    # Iterate over frames to plot and update max_counts for histograms
    for rel_idx in tqdm(rel_frames):
        # ---FIRST ROW---

        # Histograms (N, area, thetas)
        for i, param in enumerate(hist_params):

            if param == 'meanThetas':
                bin_edges = np.linspace(min_x[param], max_x[param], n_bins + 1)
            else:
                bin_edges = np.logspace(np.log10(max(min_x[param], 1e-6)), np.log10(max_x[param]), n_bins + 1)
            counts, _ = np.histogram(data[str(rel_idx)][param], bins = bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Update max_counts
            if np.max(counts) > max_counts[param]:
                max_counts[param] = np.max(counts)

            # Add trace
            fig.add_trace(go.Bar(x=bin_centers, y=counts,                            
                                 showlegend=False,
                                 visible=(rel_idx == 0),  # only t = 0 visible initially
                                 marker_color='steelblue',
                                 marker_line_width=0), row=1, col=i + 1)
        
        # ---(END OF FIRST AND) SECOND ROW---

        for i, x_param in enumerate(scatter_x_params):
            if type(scatter_y_params[i]) == list:
                for j in range(2):
                    fig.add_trace(go.Scatter(x=data[str(rel_idx)][x_param], y=data[str(rel_idx)][scatter_y_params[i][j]], mode='markers', 
                                             marker=dict(color=['steelblue', 'coral'][j], size=5), showlegend=False, visible=(rel_idx == 0)), scatter_rows[i], scatter_cols[i], bool(j))
            else:
                fig.add_trace(go.Scatter(x=data[str(rel_idx)][x_param], y=data[str(rel_idx)][scatter_y_params[i]], mode='markers', 
                                         marker=dict(color='steelblue', size=5), showlegend=False, visible=(rel_idx == 0)), scatter_rows[i], scatter_cols[i], False)
                            
        # ---THIRD ROW---

        # Line plots (pols vs layer (center), pols vs layer (edge), dens vs layer (center), dens vs layer (edge))
        for i in range(4):
            idvs_trace, mean_trace = plot_layers(data[str(rel_idx)][layer_params[i]], min_obs, max_layers, rel_idx)
            fig.add_trace(idvs_trace, 3, i+1)
            fig.add_trace(mean_trace, 3, i+1)
        del idvs_trace, mean_trace

        data[str(rel_idx)] = None  # Reduce memory as we go
        gc.collect()

    # Build slider steps
    traces_per_frame = len(fig.data) // len(rel_frames)
    assert len(fig.data) % len(rel_frames) == 0, f"Trace count {len(fig.data)} not divisible by {len(rel_frames)} frames"

    steps = []
    for i, rel in enumerate(rel_frames):
        # Create 
        visible_mask = np.zeros(traces_per_frame * len(rel_frames)).astype(bool)

        start = i * traces_per_frame
        end = start + traces_per_frame
        visible_mask[start:end] = True

        steps.append(dict(method='restyle', args=[{'visible':visible_mask.tolist()}],
                         label=f't={rel:+d}' if rel != 0 else 't=0'))
        
    # Update figure
    fig.update_layout(sliders=[dict(active=round((len(data) - 1)/2), steps=steps,
                                   currentvalue=dict(prefix='Relative frame: ', font=dict(size=13)),
                                   pad=dict(t=40, b=10))],
                      height=900, template='plotly_white', margin=dict(l=50, r=30, t=80, b=80))

    # Histograms
    for i in range(3):
        x_scale = ['log', 'log', 'linear'][i]
        print([np.log10(max(min_x[hist_params[i]], 1e-6)), np.log10(max_x[hist_params[i]])])
        fig.update_xaxes(title_text = label_dict[hist_params[i]], range = [min_x[hist_params[i]], max_x[hist_params[i]]]
                                                                           if x_scale == 'linear'
                                                                           else [np.log10(max(min_x[hist_params[i]], 1e-6)), np.log10(max_x[hist_params[i]])], 
                         row = 1, col = i + 1, type = x_scale)
        y_scale = ['log', 'log', 'linear'][i]
        fig.update_yaxes(title_text = 'Counts', range = [0, max_counts[hist_params[i]]]
                                                         if y_scale == 'linear'
                                                         else [0.9, np.log10(max_counts[hist_params[i]])], 
                         row = 1, col = i + 1, type = y_scale)

    # Scatterplots
    for i in range(5):
        x_scale = ['log', 'log', 'log', 'log', 'linear'][i]
        fig.update_xaxes(title_text = label_dict[scatter_x_params[i]], range = [min_x[scatter_x_params[i]], max_x[scatter_x_params[i]]]
                                                                                if x_scale == 'linear'
                                                                                else [np.log10(max(min_x[scatter_x_params[i]], 1e-6)), np.log10(max_x[scatter_x_params[i]])], 
                         row = scatter_rows[i], col = scatter_cols[i], type = x_scale)

        y_scale = ['log', 'linear', 'linear', 'linear', 'linear'][i]
        if type(scatter_y_params[i]) == list:
            for j in range(2):
                fig.update_yaxes(title_text = label_dict[scatter_y_params[i][j]], range = [min_x[scatter_y_params[i][j]], max_x[scatter_y_params[i][j]]] 
                                                                                           if y_scale == 'linear' 
                                                                                           else [np.log10(max(min_x[scatter_y_params[i][j]], 1e-6)), np.log10(max_x[scatter_y_params[i][j]])], 
                                 row = scatter_rows[i], col = scatter_cols[i], secondary_y = bool(j), type = y_scale)
        else:
            fig.update_yaxes(title_text = label_dict[scatter_y_params[i]], range = [min_x[scatter_y_params[i]], max_x[scatter_y_params[i]]] 
                                                                                    if y_scale == 'linear' 
                                                                                    else [np.log10(max(min_x[scatter_y_params[i]], 1e-6)), np.log10(max_x[scatter_y_params[i]])], 
                             row = scatter_rows[i], col = scatter_cols[i], type = y_scale)
        
    # Layer plots
    for i in range(4):
        fig.update_xaxes(title_text = ['Layer (from center)', 'Layer (from edge)', 'Layer (from center)', 'Layer (from edge)'][i], row = 3, col = i + 1)
        fig.update_yaxes(title_text = label_dict[layer_params[i]], row = 3, col = i + 1, range = [min_x[layer_params[i]], max_x[layer_params[i]]])

    output_path = '.'.join(input_path.split('.')[:-1]) + '.html'
    fig.write_html(output_path)
    print(f'Saved to {output_path}')
    return fig

def interactive_cluster_merging(input_path: str):

    data = load_cluster_stats_h5(input_path)

    # Collect total clustered individuals and total number of clusters per absolute frame
    abs_frames = [int(key) for key in data.keys()]
    n_clustered = []
    n_clusters = []

    for abs in data.keys():
        ns = data[abs]['ns']
        n_clustered.append(np.sum(ns))
        n_clusters.append(len(ns))

    # Create figure
    fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": False}]])

    # Col 1: time series
    fig.add_trace(go.Scatter(x=abs_frames, y=n_clustered, line=dict(color='steelblue'), name="Clustered individuals"), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=abs_frames, y=n_clusters, line=dict(color='coral'), name="Clusters"), secondary_y=True, row=1, col=1)

    # Col 2: n_clustered vs n_clusters scatter
    fig.add_trace(go.Scatter(x=n_clustered, y=n_clusters, mode='markers', marker=dict(color=abs_frames, colorscale='Viridis', colorbar=dict(title='Absolute frame', orientation='h', x=0.72, y=-0.1, xanchor='center', yanchor='top',
                                                                                                                                            len=0.45, thickness=20), showscale=True), name="Single frame"), secondary_y=False, row=1, col=2)

    # Add range slider to col 1 x-axis only
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type='linear'),
                    yaxis=dict(anchor="x", autorange=True, mirror=True, showline=True, side="left", tickmode="auto", ticks="", type="linear", zeroline=False),
                    yaxis2=dict(anchor="x", autorange=True, mirror=True, showline=True, side="right", tickmode="auto", ticks="", type="linear", zeroline=False))

    # Update labels — use col to target the right axis
    fig.update_xaxes(title_text='Frame', row=1, col=1)
    fig.update_xaxes(title_text='Total number of clustered individuals', row=1, col=2)
    fig.update_yaxes(title_text='Total number of clustered individuals', secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text='Total number of clusters', secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text='Total number of clusters', row=1, col=2)

    output_path = '.'.join(input_path.split('.')[:-1]) + '_merging.html'
    fig.write_html(output_path)
    print(f'Saved to {output_path}')
    return fig

def interactive_cluster_structure(ds:xr.Dataset, input_path:str, layer_cutoff:int| None = None, fps:int = 5, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1):

    # Load data
    data = load_cluster_stats_h5(input_path)

    # Define stat names that will be aggregated and initialize storage dictionary
    stat_names = ['p_by_layer', 'p_from_edge', 'd_by_layer', 'd_from_edge']

    # Store data according to max_layer value and relative frame - first by max_layer, then by stat name
    stats_by_max_layer:dict[int, dict[str, list]] = {}

    # Iterate over absolute frames
    for j, key in enumerate(data.keys()):

        # Iterate over stat type
        for stat in stat_names:

            # Get observations for this absolute frame
            csr_dict = data[key][stat]

            # Unpack vals and idcs arrays
            vals = csr_dict['data']
            idcs = csr_dict['indptr']

            # Add lists to dictionary
            for i in range(len(idcs) - 1):
                max_layer = idcs[i+1] - idcs[i]
                stats_by_max_layer.setdefault(int(max_layer), {}).setdefault(stat, []).append(vals[idcs[i]:idcs[i+1]])

    def plot_layers(stats_by_max_layer:dict[int, dict[str, list]], max_layer:int, stat:str, cutoff:int | None = None):

        # Generate None separated xs and ys lists
        xs = []
        ys = []
        for row in stats_by_max_layer[max_layer][stat]:
            y = row[:cutoff]
            xs.extend(range(len(y)))
            ys.extend(y)

            # None separator - breaks the line between clusters
            xs.append(None)
            ys.append(None)

        # Create plotly trace for individual cluster curves
        indivs_trace = go.Scatter(x=xs, y=ys, mode='lines', line=dict(color="#FFF0B8", width=0.8), showlegend=False, hoverinfo='skip', connectgaps=False, visible=(max_layer == 5))

        # Take median of all cluster curves as a function of layer
        
        
        median_y = np.nanmedian(stats_by_max_layer[max_layer][stat], axis = 0)[:cutoff]
        median_x = np.arange(len(median_y))

        # Create plotly trace for median curve
        median_trace = go.Scatter(x=median_x, y=median_y, mode='lines', line=dict(color="#1172D2", width=2), showlegend=False, connectgaps=False, visible=(max_layer == 5))

        return indivs_trace, median_trace, np.nanmax(stats_by_max_layer[max_layer][stat])
    
    # Initialize figure and variables to store extrema
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.16, vertical_spacing=0.12)
    max_x = layer_cutoff if layer_cutoff else np.max(list(stats_by_max_layer.keys()))
    max_y = {stat: [1, 1, 0, 0] for stat in stat_names}
    all_max_ds = []
    
    # Iterate over unique n values and plot (including adding slider steps)
    unique_max_layers = np.unique(list(stats_by_max_layer.keys()))
    steps = []
    traces_per_frame = 2*len(stat_names) # 2: Individuals, median

    for j, max_layer in enumerate(unique_max_layers):

        # Iterate over different stats
        for i, stat in enumerate(stat_names):

            # Get traces
            indivs, median, maxy = plot_layers(stats_by_max_layer, max_layer, stat, layer_cutoff)

            # Update extrema
            if stat == 'd_by_layer': # Don't need to do it for d_from_edge, since values are the same
                all_max_ds.append(maxy)

            # Add traces to figure
            fig.add_trace(indivs, row= (i // 2) + 1, col= (i % 2) +1)
            fig.add_trace(median, row= (i // 2) + 1, col= (i % 2) +1)
    
        # Create a visibility mask
        visible_mask = np.zeros(traces_per_frame * len(unique_max_layers)).astype(bool)
        start = j*traces_per_frame
        end = start + traces_per_frame
        visible_mask[start:end] = True

        steps.append(dict(method='restyle', args=[{'visible':visible_mask.tolist()}], label=f'l = {max_layer}'))

    # Update maximum density values using quantiles (to exclude outliers)
    for stat in ['d_by_layer', 'd_from_edge']:
        max_y[stat] = np.quantile(all_max_ds, 0.9)

    # Update figure
    fig.update_layout(sliders=[dict(active= 5, steps=steps, currentvalue=dict(prefix='Layer radius: ', font=dict(size=13)), pad=dict(t=40, b=10))], height=900, template='plotly_white', margin=dict(l=50, r=30, t=80, b=80))
    
    
    # Update axes of figure
    row_labels = ['Median polarization', 'Median density']
    col_labels = ['Voronoi layer (from center)', 'Voronoi layer (from edge)']
    for i in range(4):
        fig.update_xaxes(title_text=col_labels[i%2] if i > 1 else None, range=[0, max_x], row=(i//2)+1, col=(i%2)+1)
        fig.update_yaxes(title_text=row_labels[i//2] if not (i%2) else None, range=[0, max_y[stat_names[i]]], row=(i//2)+1, col=(i%2)+1)

    # Save figure
    output_path = '.'.join(input_path.split('.')[:-1]) + '_structure.html'
    fig.write_html(output_path)
    print(f'Saved to {output_path}')
    return fig
            


    
    # def idk():
    #     # Load data
    #     data = load_cluster_stats_h5(input_path)

    #     # Get event frames
    #     _, abs_event_frames, _, _ = find_reflections(ds, fps, start_frame, end_frame, subsample)

    #     # Find absolute frames in data
    #     abs_frames = [int(key) for key in data.keys()]

    #     # For each abs_frame, find the nearest event frame and compute offset
    #     offsets = abs_event_frames[np.argmin(np.abs(abs_frames[:, None] - abs_event_frames[None, :]), axis=1)]
    #     rel_frames = abs_frames - offsets
        


    