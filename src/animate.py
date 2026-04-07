import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os

cwd = os.getcwd()
if cwd.endswith('project'):
    from helper_fns import *
else:
    from src.helper_fns import *

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
    frames = get_frame_slice(ds, start_frame, end_frame)

    # Subset dataset
    ds_sub = ds.sel(frame = frames)

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

def animate_focal_ego(ds:xr.DataArray, fid:int, video_path:str, smooth_name:str, output_dir:str, buffer:int = 150, start_frame:int = 0, end_frame:int | None = None, interval:int = 50, vid_frame_rate:int = 5, ds_frame_rate:int = 5):
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

    # Handle video and ds frame rate mismatches by only updating video every nth frame or vice versa
    frame_mod = 1
    ds_mod = 1
    if vid_frame_rate != ds_frame_rate:
        if vid_frame_rate < ds_frame_rate: # Update video frame less often than ds frame
            frame_mod = int(np.ceil(ds_frame_rate / vid_frame_rate))

        else: # Update video frame more often than ds frame
            ds_mod = int(np.ceil(vid_frame_rate / ds_frame_rate))

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
        
        # Set frame number for ds, accounting for potential frame rate mismatches
        ds_frame_idx = np.floor(idx/ds_mod).astype(int)

        # Get position of focal
        frame_data = ds_sub.isel(frame=ds_frame_idx)
        # fid_pos = np.array([frame_data['x' + smooth_name].values, frame_data['y' + smooth_name].values])
        fid_pos = np.array([frame_data['x'].values, frame_data['y'].values])
        detected = np.sum(np.isnan(fid_pos)) == 0

        # Get direction of focal
        theta = frame_data['theta' + smooth_name].values
        theta = theta if detected else 0
        # theta = 0

        # Show frame of video
        # cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        if not idx % frame_mod:
            ret, frame = cap.read()

            if ret:
                # Rotate frame to align with focal's heading
                center = (fid_pos[0], fid_pos[1]) if detected else (cap.shape[1]//2, cap.shape[0]//2)
                # rotated_frame = rotate_frame(frame, center=center, angle_rad= -theta)
                rotated_frame = frame
                img_artist.set_data(cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB))

                return [img_artist, line]
        
        # Set new axis limits centered around focal
        if detected:
            ax.set_xlim(fid_pos[0] - buffer, fid_pos[0] + buffer)
            ax.set_ylim(fid_pos[1] - buffer, fid_pos[1] + buffer)

            # Update scatter plot
            line_length = 20
            line.set_data([fid_pos[0], fid_pos[0] + line_length * np.cos(theta)], [fid_pos[1], fid_pos[1] + line_length * np.sin(theta)])
        else:
            ax.set_xlim(0, cap.shape[1])
            ax.set_ylim(0, cap.shape[0])
            # scat.set_offsets([np.nan, np.nan])
            line.set_data([], [])

        if idx % 100 == 0:
            print(f'Processed frame {idx+1}/{len(frames)}')
        
        # return [img_artist, scat]
        return [img_artist, line]

    num_updates = ds_mod*len(frames)
    ani = FuncAnimation(fig, update, frames=num_updates, init_func=init, interval=interval, blit=False, repeat=False)
    ani.save(output_dir + f'ego_focal_{fid}{smooth_name}.gif')
    cap.release()
    return

def animate_orientations_fast(ds:xr.DataArray, smooth_name:str, output_dir:str, start_frame:int = 0, end_frame:int | None = None, interval:int = 50, trail:int =  10):
    
    # Get slice
    frames_range = get_frame_slice(ds, start_frame, end_frame)
    ds_sub = ds.sel(frame=frames_range)
    
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
    frames_range = get_frame_slice(ds, start_frame, end_frame)
    ds_sub = ds.sel(frame=frames_range)
    
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