'''_____________________________________________________IMPORTS____________________________________________________________'''

from networkx import edges
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
import pynumdiff
from scipy.stats import circvar, binned_statistic, binned_statistic_2d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

from data_handling import load_preprocessed_data, save_ds, load_cluster_ids
from clean_tracks import compute_speed, compute_ang_speed, smooth_nantolerant
from phase import get_local_env, local_density_gradient
from visualize_preprocessed import plot_smoothed_coords, plot_speed_hists, plot_ang_speed
from compute_activity import bootstrap_threshold
from cluster_analysis import compute_whole_batch_clusters, find_reflections
from helper_fns import get_frame_slice

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def find_largest_cluster(clusters:dict):
    '''Find largest cluster at each absolute time frame.
    clusters has CSR format:
        abs_frame
            ids
            layers
            idx_ptr
    '''

    abs_frames = np.asarray(sorted(clusters.keys()))
    largest_idcs = np.full((len(abs_frames), 2), np.nan) # (num_frames, start/end)

    for i, frame in enumerate(abs_frames):
        ptr = clusters[frame]['idx_ptr']
        max_idx = np.argmax(np.diff(ptr))
        largest_idcs[i,:] = [ptr[max_idx], ptr[max_idx + 1]]

    return largest_idcs.astype(int), abs_frames

def find_band_speed(ds:xr.Dataset, clusters:dict, output_dir:str, x_var:str = 'x_raw', y_var:str = 'y_raw', v_var:str = 'v_high_ord', fps:float = 5):
    '''Find the speed to the centroid of the largest cluster.
    clusters has CSR format:
            abs_frame
                ids
                layers
                idx_ptr
    '''

    # Pre-load data arrays
    positions = np.array([ds[x_var].transpose('id','frame').values, ds[y_var].transpose('id','frame').values]) # (x/y, max_id, n_frames)
    thetas = ds['theta'].transpose('id', 'frame').values
    vs = ds[v_var].transpose('id','frame').values # (max_id, n_frames)

    # Find indices of the largest cluster in each frame
    largest_idcs, abs_frames = find_largest_cluster(clusters)

    def circular_mean(angles):
        """Circular mean of angles in radians; ignores non-finite values."""
        angles = np.asarray(angles, dtype=float)
        angles = angles[np.isfinite(angles)]

        if angles.size == 0:
            return np.nan

        return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

    # Create mask for reflections
    rel_reflections, abs_reflections, period, sides = find_reflections(ds, fps)
    refl_mask = np.min(np.abs(abs_frames[:,None] - abs_reflections), axis = 1) < (period/4)

    # Find the positions and velocities of the ids in the largest cluster in each frame
    n_ids = []
    rel_pos = []
    vels = []
    theta_list = []
    centroids = np.full((2, len(abs_frames)), np.nan)
    mean_thetas = np.full(len(abs_frames), np.nan)
    var_thetas = np.full(len(abs_frames), np.nan)
    for i, frame in enumerate(abs_frames):

        # Get ids and positions of locusts in the band
        ids = clusters[frame]['ids'][largest_idcs[i,0]:largest_idcs[i,1]]
        n_ids.append(len(ids))
        pos_t = positions[:, ids, i] # (x/y, n_ids)

        # Store positions and velocities of locusts in the band
        if not refl_mask[i]: # Mask-out reflections for speed plots
            theta_list.extend(thetas[ids, i])
            vels.extend(vs[ids, i]) # (n_ids,)

        # Compute centroid of the band
        centroids[:,i] = np.nanmean(pos_t, axis = 1)

        # Get circular mean orientation of locusts in the band
        mean_theta = circular_mean(thetas[ids, i])
        mean_thetas[i] = mean_theta
        var_thetas[i] = circvar(thetas[ids, i], nan_policy='omit')

        # Get sign-fixed relative postions of locusts in the band
        if not refl_mask[i]: # Mask-out reflections for speed computations
            if not (-np.pi/2 <= mean_theta < np.pi/2): # Band is moving left
                rel_pos.extend(-1*(pos_t[0,:] - np.nanmin(pos_t[0,:])))
            else: # Band is moving right
                rel_pos.extend(pos_t[0,:] - np.nanmax(pos_t[0,:]))

    # Mask out reflections for speed plots
    centroids[:,refl_mask] = np.nan
    centroids[:,refl_mask] = np.nan

    window = 15  # odd number of frames; try 11–15
    polyorder = 2

    centroids_for_smoothing = centroids.copy()
    centroids_for_smoothing[:, refl_mask] = np.nan

    x_cluster = smooth_nantolerant(
        centroids_for_smoothing[0],
        lambda segment, **_: savgol_filter(
            segment, window_length=window, polyorder=polyorder, mode="interp"
        ),
        {},
        min_tracklet_length=window
    )

    y_cluster = smooth_nantolerant(
        centroids_for_smoothing[1],
        lambda segment, **_: savgol_filter(
            segment, window_length=window, polyorder=polyorder, mode="interp"
        ),
        {},
        min_tracklet_length=window
    )

    dt = np.diff(abs_frames) / fps
    vx_cluster = np.diff(x_cluster) / dt
    vy_cluster = np.diff(y_cluster) / dt

    # # Compute band speed
    # high_ord_params = {'dt': 1, 'num_iterations': 1, 'order': 4}
    # min_tracklet_length = high_ord_params['order'] + 1
    # x_cluster, vx_cluster = smooth_nantolerant(centroids[0,:], pynumdiff.finite_difference.finitediff, high_ord_params, min_tracklet_length)
    # y_cluster, vy_cluster = smooth_nantolerant(centroids[1,:], pynumdiff.finite_difference.finitediff, high_ord_params, min_tracklet_length)
    # vx_cluster *= fps
    # vy_cluster *= fps


    fig, ax = plt.subplots(2, 3, figsize = (10, 8))

    time = abs_frames / fps

    band_vx = np.abs(vx_cluster[np.isfinite(vx_cluster)])
    band_vy = np.abs(vy_cluster[np.isfinite(vy_cluster)])

    bins = np.linspace(0, max(np.quantile(band_vx, 0.99), np.quantile(band_vy, 0.99)), 20)

    ax[0][0].hist(
        band_vx,
        bins=bins,
        density=True,
        linewidth=2,
        alpha = 0.5,
        histtype='step',
        color="C0",
        label="x",
    )
    ax[0][0].hist(
        band_vy,
        bins=bins,
        density=True,
        linewidth=2,
        alpha = 0.5,
        histtype='step',
        color="C1",
        label="y",
    )

    ax[0][0].set_xlabel("Band speed (m/s)")
    ax[0][0].set_ylabel("Probability density")
    ax[0][0].set_title("Band speed components")
    ax[0][0].legend()

    ax[0][1].plot(time, x_cluster, label = 'x')
    ax[0][1].plot(time, y_cluster, label = 'y')
    ax[0][1].set_xlabel('Time (s)')
    ax[0][1].set_ylabel('Band centroid (m)')
    ax[0][1].legend()

    # Band speed magnitude at each frame
    band_speed = np.hypot(vx_cluster, vy_cluster)

    # Individual speed quantiles for the largest cluster at each frame
    q50 = np.full(len(band_speed), np.nan)
    q95 = np.full(len(band_speed), np.nan)

    for i, frame in enumerate(abs_frames):
        if not i:
            continue
        if not refl_mask[i]:

            ids = clusters[frame]["ids"][largest_idcs[i, 0]:largest_idcs[i, 1]]
            individual_speeds = vs[ids, i]
            individual_speeds = individual_speeds[np.isfinite(individual_speeds)]

            if individual_speeds.size:
                q50[i-1], q95[i-1] = np.quantile(individual_speeds, [0.50, 0.95])

    valid = (
        np.isfinite(band_speed)
        & np.isfinite(q50)
        & np.isfinite(q95)
        & ~refl_mask[1:]
    )

    ax[0][2].scatter(band_speed[valid], q50[valid], s=10, alpha=0.35, label="50th percentile")
    ax[0][2].scatter(band_speed[valid], q95[valid], s=10, alpha=0.35, label="95th percentile")
    ax[0][2].set_xlim([0, np.nanquantile(band_speed, 0.99)])
    ax[0][2].set_xlabel("Band speed (m/s)")
    ax[0][2].set_ylabel("Individual locust speed (m/s)")
    ax[0][2].legend()
    ax[0][2].set_aspect('equal')

    # INDIVIDUALS
    rel_pos = np.asarray(rel_pos)
    vels = np.asarray(vels)
    theta_list = np.asarray(theta_list)

    vx_indiv = np.abs(vels * np.cos(theta_list))
    vy_indiv = np.abs(vels * np.sin(theta_list))
    
    valid_x = np.isfinite(rel_pos) & np.isfinite(vx_indiv)
    valid_y = np.isfinite(rel_pos) & np.isfinite(vy_indiv)

    bins = np.linspace(0, max(np.nanquantile(vx_indiv, 0.99), np.nanquantile(vy_indiv, 0.99)), 20)

    ax[1][0].hist(
            vx_indiv[valid_x],
            bins=bins,
            density=True,
            linewidth=2,
            alpha = 0.5,
            histtype='step',
            color="C0",
            label="x",
        )
    ax[1][0].hist(
        vy_indiv[valid_y],
        bins=bins,
        density=True,
        linewidth=2,
        alpha = 0.5,
        histtype='step',
        color="C1",
        label="y",
    )
    
    ax[1][0].set_xlabel("Indv. speed (m/s)")
    ax[1][0].set_ylabel("Probability density")
    ax[1][0].set_title("Indv. speed components")
    ax[1][0].legend()

    if not (valid_x.any() or valid_y.any()):
        raise ValueError("No finite relative-position/velocity observations to plot.")

    # Shared bin edges ensure both panels cover the same ranges.
    x_min = min(np.min(rel_pos[valid_x]), np.min(rel_pos[valid_y]))
    x_max = max(np.max(rel_pos[valid_x]), np.max(rel_pos[valid_y]))
    y_min = min(np.min(vx_indiv[valid_x]), np.min(vy_indiv[valid_y]))
    y_max = max(np.quantile(vx_indiv[valid_x], 0.99), np.quantile(vy_indiv[valid_y], 0.99))

    n_bins = 30
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)

    # Compute both histograms first so their colors use the same count scale.
    hx, _, _ = np.histogram2d(
        rel_pos[valid_x], vx_indiv[valid_x], bins=(x_edges, y_edges)
    )
    hy, _, _ = np.histogram2d(
        rel_pos[valid_y], vy_indiv[valid_y], bins=(x_edges, y_edges)
    )

    vmax = max(hx.max(), hy.max())
    if vmax == 0:
        vmax = 1

    mesh_x = ax[1][1].hist2d(
        rel_pos[valid_x], vx_indiv[valid_x],
        bins=(x_edges, y_edges),
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )[3]

    mesh_y = ax[1][2].hist2d(
        rel_pos[valid_y], vy_indiv[valid_y],
        bins=(x_edges, y_edges),
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )[3]

    # Conditional central intervals for each velocity component.
    intervals = {"50%": (0.25, 0.75)}

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2

    for ax_i, y_values, valid, component in [
        (1, vx_indiv, valid_x, "x"),
        (2, vy_indiv, valid_y, "y"),
    ]:
        for label, (q_low, q_high) in intervals.items():
            lows = np.full(n_bins, np.nan)
            highs = np.full(n_bins, np.nan)

            for b in range(n_bins):
                in_bin = (
                    valid
                    & (rel_pos >= x_edges[b])
                    & (rel_pos < x_edges[b + 1])
                )
                values = y_values[in_bin]

                if values.size:
                    lows[b], highs[b] = np.quantile(values, [q_low, q_high])

            ax[1][ax_i].plot(
                x_centers, lows, color="white", linestyle="--", alpha=0.9
            )
            ax[1][ax_i].plot(
                x_centers, highs, color="white", linestyle="--", alpha=0.9,
                label=f"{label} interval",
            )

        ax[1][ax_i].set_xlabel("Rel. x distance from wavefront (m)")
        ax[1][ax_i].set_ylabel(f"Indv. {component} speed (m/s)")
        ax[1][ax_i].legend(loc = 'upper right')

    fig.tight_layout(rect=[0, 0, 0.92, 1])  # reserve space on the right

    cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.32])
    fig.colorbar(
        mesh_y,
        cax=cbar_ax,
        label="Number of observations per bin",
    )

    plt.savefig(f'{output_dir}clusters/band_speed_{abs_frames[0]}_{abs_frames[-1]}.png')

    return

def band_by_density(ds:xr.Dataset, output_dir:str, x_var:str = 'x_raw', y_var:str = 'y_raw', v_var:str = 'v_high_ord', arena_center:np.ndarray = np.array([0.01, 0]), quantiles:list[float] = [0.1, 0.5, 0.9], 
                    high_ord_params:dict = {'dt': 1, 'num_iterations': 1, 'order': 4}, sg_params:dict = {'window_length':15, 'polyorder':2, 'mode':"interp"}, n_peak_bins: int = 100, n_rel_bins: int = 20, n_arena_bins: int = 20, 
                    peak_smoothing_width_m: float = 0.01, min_counts = 15, v_thresh:float = 0.004, fps:float = 5, shown_side:str = 'right', n_boot:int = 1000, block_len:int = 25):
    '''Using Dor's method to define the band and investigate faster-than-locust density wave.'''

    assert shown_side in ['left', 'right'], "shown_side must be either 'left' or 'right'."

    # Load absolute frames of the dataset
    abs_frames = ds.frame.values.astype(int)
    rel_frames = np.arange(len(ds.frame.values))

    # Pre-load data arrays
    positions = np.array([ds[x_var].transpose('id','frame').values, ds[y_var].transpose('id','frame').values]) # (x/y, max_id, n_frames)
    vs = ds[v_var].transpose('id','frame').values # (max_id, n_frames)
    thetas = ds['theta'].transpose('id', 'frame').values # (max_id, n_frames)

    # Compute dt
    dt = ds['frame'].diff('frame') / fps

    # Create mask for reflections
    rel_reflections, _, period, sides = find_reflections(ds, fps)
    refl_mask = np.min(np.abs(rel_frames[:, None] - rel_reflections), axis = 1) < (period/4) # Frames too close to a reflection

    # Flip positions and x velocities so that left and right marches are aligned
    if sides[0] == shown_side:
        flip_initial_segment = False
    else:
        flip_initial_segment = True

    segment_edges = np.r_[0, rel_reflections, len(rel_frames)]
    for segment_i, (start, stop) in enumerate(zip(segment_edges[:-1], segment_edges[1:])):

        # Determine if this segment should be flipped
        flip_this_segment = flip_initial_segment ^ bool(segment_i % 2) # a ^ b is True if a != b and False otherwise

        if flip_this_segment:
            positions[0, :, start:stop] = (2 * arena_center[0] - positions[0, :, start:stop])

    # Compute vx
    vx = np.full_like(vs, np.nan) # (n_ids, frame)
    for i, row in enumerate(positions[0,:,:]):
        _, vx_temp = smooth_nantolerant(row, pynumdiff.finite_difference.finitediff, high_ord_params, min_tracklet_length=high_ord_params['order'] + 1, two_outputs = True)
        vx[i,:] = vx_temp / np.append([1/fps], dt) # Add extra dt value since finitediff infers the first value (n_ids, frames)

    # Define x coordinate bins for computing the peak
    x_peak_bins = np.linspace(np.nanmin(positions[0]), np.nanmax(positions[0]), n_peak_bins + 1) # Fine, fixed grid for estimating the density peak
    x_peak_centers = (x_peak_bins[:-1] + x_peak_bins[1:]) / 2
    peak_bin_width = x_peak_bins[1] - x_peak_bins[0]

    # Keep the smoothing width constant in physical space as resolution changes
    peak_sigma_bins = peak_smoothing_width_m / peak_bin_width

    # Initialize distribution matrix
    band_dists = np.full((n_peak_bins, len(rel_frames)), np.nan)
    band_smooth = np.full_like(band_dists, np.nan)
    band_center_idcs = np.full(len(rel_frames), -1, dtype=int)
    band_peaks = np.full(len(rel_frames), np.nan)

    # Iterate over included frames and compute position distributions for each frame
    for i in range(len(rel_frames)):

        # Save position distribution
        band_dists[:,i], _ = np.histogram(positions[0,:,i], x_peak_bins, density = True)

        # Save smoothed distribution
        band_smooth[:, i] = gaussian_filter1d(band_dists[:,i], sigma=peak_sigma_bins, mode="nearest")

        # Find peak from smoothed distribution
        band_center_idcs[i] = np.nanargmax(band_smooth[:, i])
        band_peaks[i] = x_peak_centers[band_center_idcs[i]]


    # Compute relative positions
    rel_x = positions[0,:,:] - band_peaks # (n_ids, frames)
    del positions

    # Max coarser relative bins for statistics
    rel_x_bins = np.linspace(np.nanmin(rel_x), np.nanmax(rel_x), n_rel_bins + 1)
    rel_x_centers = (rel_x_bins[:-1] + rel_x_bins[1:])/2

    # Define bootstrapping function for binned values
    def bootstrap_band_speed_binned(x_data, y_data, n_arena_bins, refl_mask, n_boot, block_len):
        """
        Bootstrap median band speed by arena-x bin.

        Contiguous visits to each bin are split into time blocks and resampled
        within that bin, preserving short-range temporal dependence.
        block_len defines the maximum length of contiguous visits sampled.

        Returns
        -------
        median_by_x : ndarray, shape (n_bins,)
        ci_low, ci_high : ndarray, shape (n_bins,)
            2.5th and 97.5th percentiles of the bootstrap medians.
        counts : ndarray, shape (n_bins,)
            Number of valid frame observations per bin.
        n_visits : ndarray, shape (n_bins,)
            Number of contiguous visits per bin.
        """

        # Make coarser arena bins for statistics
        arena_bins = np.linspace(np.nanmin(x_data), np.nanmax(x_data), n_arena_bins + 1)
        arena_centers = (arena_bins[:-1] + arena_bins[1:])/2

        # Generate random seed
        rng = np.random.default_rng(0)

        # Exclude frames that are too close to a reflection
        valid = np.isfinite(x_data) & np.isfinite(y_data) & (~refl_mask)

        # Bin IDs are 0-based; include the rightmost edge in the last bin.
        bin_id = np.digitize(x_data, arena_bins) - 1
        bin_id[x_data == arena_bins[-1]] = n_arena_bins - 1
        valid &= (bin_id >= 0) & (bin_id < n_arena_bins)

        # Instantiate data storage
        median_by_x = np.full(n_arena_bins, np.nan)
        ci_low = np.full(n_arena_bins, np.nan)
        ci_high = np.full(n_arena_bins, np.nan)
        counts = np.zeros(n_arena_bins, dtype=int)
        n_visits = np.zeros(n_arena_bins, dtype=int)

        # Iterate over bins
        for b in range(n_arena_bins):

            # Mask for sample in bin that are valid
            in_bin = valid & (bin_id == b)
            frame_indices = np.flatnonzero(in_bin)
            counts[b] = len(frame_indices)

            if counts[b] == 0:
                continue

            median_by_x[b] = np.median(y_data[frame_indices])

            # Separate visits wherever frames are not consecutive.
            run_splits = np.flatnonzero(np.diff(frame_indices) > 1) + 1
            visits = np.split(frame_indices, run_splits)
            n_visits[b] = len(visits)

            # Split each visit into consecutive blocks; retain any short final block.
            blocks = [visit[start:start + block_len] for visit in visits for start in range(0, len(visit), block_len)]

            boot_medians = np.full(n_boot, np.nan)

            # Iterate over number of bootstraps
            for r in range(n_boot):
                sampled_indices = []
                sampled_count = 0

                while sampled_count < counts[b]:
                    # Randomly choose block with replacement
                    chosen_block = blocks[rng.integers(len(blocks))]
                    sampled_indices.extend(chosen_block)
                    sampled_count += len(chosen_block)

                # Truncate data to have same number of frames as original dataset
                sampled_indices = np.asarray(sampled_indices[:counts[b]], dtype=int)
                boot_medians[r] = np.median(y_data[sampled_indices])

            ci_low[b], ci_high[b] = np.quantile(boot_medians, [0.025, 0.975])

        # Plot only bins with enough original observations
        valid_bins = (counts > min_counts) & np.isfinite(median_by_x)

        return median_by_x, ci_low, ci_high, counts, n_visits, valid_bins, arena_bins, arena_centers

    # Initiate figure
    fig, ax = plt.subplots(3, 3, figsize = (10, 10))

    # PLOT 0,0: Centered median probability density of band

    # Align all rows of density distributions so that their peaks are at the same index
    middle_index = len(x_peak_centers) // 2
    for i, peak_idx in enumerate(band_center_idcs):
        shift = middle_index - peak_idx
        band_dists[:, i] = np.roll(band_dists[:, i], shift)

    ax[0][0].plot(x_peak_centers, np.nanmedian(band_dists[:, ~refl_mask], axis=1)) # Reflections excluded
    ax[0][0].set_xlabel(r"$x' = x - x_p$ (m)")
    ax[0][0].set_ylabel('Median probability density')

    # PLOT 0,1: Median band speed over arena x

    # Compute instantaneous speed of band after smoothing peak positions
    sg_params_band = sg_params.copy()
    sg_params_band['window_length'] = 75
    band_peaks_smoothed = smooth_nantolerant(band_peaks, savgol_filter, sg_params_band, sg_params_band['window_length']) 
    vx_band = np.full(len(rel_frames), np.nan) # (frames, )
    vx_band[1:] = np.diff(band_peaks_smoothed) / dt # (frames,)

    # Compute band speed by position in arena
    vx_by_x, ci_low, ci_high, counts, n_visits, valid_bins, arena_bins, arena_centers = bootstrap_band_speed_binned(band_peaks_smoothed, vx_band, n_arena_bins, refl_mask, n_boot, block_len)

    ax[0][1].plot(arena_centers[valid_bins], vx_by_x[valid_bins]) # Reflections included
    ax[0][1].fill_between(arena_centers[valid_bins], ci_low[valid_bins], ci_high[valid_bins], alpha = 0.3)
    ax[0][1].set_xlabel('Arena x (m)')
    ax[0][1].set_ylabel(r'Median $\vec{v_p}$ (m/s)')

    # PLOT 0,2: Circular variance of locust orientations over relative x

    # Initialize array to store circular variance of thetas in each relative x bin
    orientation_vars = np.full((n_rel_bins, len(rel_frames)), np.nan)

    def theta_var(thetas) -> float:
        """ Circular variance of an array of angles in radians. Returns a value in [0, 1], where 0 = all angles identical, 1 = maximally dispersed."""
        return 1 - np.abs(np.mean(np.exp(1j * thetas)))

    # Iterate over time frames and compute circular variance of thetas in each relative x bin
    for i in range(len(rel_frames)):

        # Create mask for valid rel_x and thetas (also excluding reflection events)
        valid_theta = np.isfinite(rel_x[:,i]) & np.isfinite(thetas[:,i]) & (~refl_mask[i]) # Exclude reflection events

        if not np.any(valid_theta):
            continue

        # Bin thetas by relative x
        binned_thetas, _, _ = binned_statistic(rel_x[valid_theta,i], thetas[valid_theta,i], statistic = theta_var, bins = rel_x_bins)

        # Store circular variance of thetas in each relative x bin
        orientation_vars[:, i] = binned_thetas

    ax[0][2].plot(rel_x_centers, np.nanmedian(orientation_vars, axis=1)) # Reflections excluded
    ax[0][2].set_xlabel(r"$x' = x - x_p$ (m)")
    ax[0][2].set_ylabel('Median orientation variance')
    ax[0][2].set_ylim([0, 1])
    ax[0][2].axvline(x=0, color = 'k', linestyle = '--')

    # PLOT 1,0: Individual speeds over relative x

    # Create mask for relative x and v (also excluding reflection events)
    valid_vx = np.isfinite(rel_x) & np.isfinite(vx) & (~refl_mask[None,:]) # Exclude reflection events

    # Compute quantiles of indv_v_by_rel_x      
    for i, q in enumerate(quantiles):
        indv_v_by_rel_x, _, _ = binned_statistic(rel_x[valid_vx], vx[valid_vx], statistic = lambda y: np.quantile(y, q), bins = rel_x_bins)
        ax[1][0].plot(rel_x_centers, indv_v_by_rel_x, label = f'q{int(100*q)}') # Reflections excluded

    ax[1][0].set_xlabel(r"$x' = x - x_p$ (m)")
    ax[1][0].set_ylabel(r'$\vec{v_x}$ (m/s)')
    ax[1][0].legend()
    ax[1][0].axvline(x=0, color = 'k', linestyle = '--')

    # PLOT 1,1: Individual speeds relative to peak speed over relative x

    # Compute quantiles of rel_v_by_rel_x
    valid_rel_vx = np.isfinite(rel_x) & np.isfinite(vx) & np.isfinite(vx_band)[None, :] & (~refl_mask[None,:]) # Exclude reflection events
    for i, q in enumerate(quantiles):
        rel_v_by_rel_x, _, _ = binned_statistic(rel_x[valid_rel_vx], (vx - vx_band[None,:])[valid_rel_vx], statistic = lambda y: np.quantile(y, q), bins = rel_x_bins)
        ax[1][1].plot(rel_x_centers, rel_v_by_rel_x, label = f'q{int(100*q)}') # Reflections excluded

    ax[1][1].set_xlabel(r"$x' = x - x_p$ (m)")
    ax[1][1].set_ylabel(r"$\vec{v_x'} = \vec{v_x} - \vec{v_p}$ (m/s)")
    ax[1][1].legend()
    ax[1][1].axvline(x=0, color = 'k', linestyle = '--')

    # PLOT 1,2: Acceleration of locusts over relative x
        
    # Compute accelerations of individuals
    accels_x = np.full_like(vx, np.nan) # (n_ids, frame)
    for i, row in enumerate(vx):
        accels_x[i,1:] = np.diff(smooth_nantolerant(row, savgol_filter, sg_params, min_tracklet_length=sg_params['window_length'])) / dt # (n_ids, frames - 1)

    # Create mask of valid rel_x and accels (also excluding reflection events)
    valid_ax = np.isfinite(rel_x) & np.isfinite(accels_x) & (~refl_mask[None,:]) # Exclude reflection events

    # Compute quantiles of indv_a_by_rel_x
    for i, q in enumerate(quantiles):
        indv_a_by_rel_x, _, _ = binned_statistic(rel_x[valid_ax], accels_x[valid_ax], statistic = lambda y: np.quantile(y, q), bins = rel_x_bins)
        ax[1][2].plot(rel_x_centers, indv_a_by_rel_x, label = f'q{int(100*q)}') # Reflections excluded

    ax[1][2].set_xlabel(r"$x' = x - x_p$ (m)")
    ax[1][2].set_ylabel(r'$\vec{a_x}$ (m/s^2)')
    ax[1][2].legend()
    ax[1][2].axvline(x=0, color = 'k', linestyle = '--')

    # PLOT 2,0: Percent moving over relative x
    
    # Create mask for relative x and v (also excluding reflection events)
    valid_v = np.isfinite(rel_x) & np.isfinite(vs) & (~refl_mask[None,:]) # Exclude reflection events

    # Compute quantiles of moving
    moving_by_rel_x, _, _ = binned_statistic(rel_x[valid_v], vs[valid_v] > v_thresh, statistic = 'mean', bins = rel_x_bins)

    # Delete velocities for space
    del vs

    ax[2][0].plot(rel_x_centers, 100*moving_by_rel_x) # Reflections excluded
    ax[2][0].set_xlabel(r"$x' = x - x_p$ (m)")
    ax[2][0].set_ylabel('Moving locusts (%)')
    ax[2][0].axvline(x=0, color = 'k', linestyle = '--')

    # PLOT 2,1: Percent of locusts faster than the peak speed over relative x
    
    # Compute percentage of v - v_p > 0 binned by x
    faster_by_rel_x, _, _ = binned_statistic(rel_x[valid_vx], (vx - vx_band[None,:])[valid_vx] > 0, statistic = 'mean', bins = rel_x_bins)

    ax[2][1].plot(rel_x_centers, 100*faster_by_rel_x)
    ax[2][1].set_xlabel(r"$x' = x - x_p$ (m)")
    ax[2][1].set_ylabel(r"$\vec{v_x'} = \vec{v_x} - \vec{v_p} > 0$ (%)")
    ax[2][1].axvline(x=0, color = 'k', linestyle = '--')

    # PLOT 2,2: Percent of locusts moving against the band over relative x
    if sides[0] == 'right':
        against_by_rel_x, _, _ = binned_statistic(rel_x[valid_vx], vx[valid_vx] < 0, statistic = 'mean', bins = rel_x_bins)
    else:
        against_by_rel_x, _, _ = binned_statistic(rel_x[valid_vx], vx[valid_vx] > 0, statistic = 'mean', bins = rel_x_bins)

    ax[2][2].plot(rel_x_centers, 100*against_by_rel_x)
    ax[2][2].set_xlabel(r"$x' = x - x_p$ (m)")
    ax[2][2].set_ylabel("Locusts moving against flow (%)")
    ax[2][2].axvline(x=0, color = 'k', linestyle = '--')

    plt.suptitle('Band marching ' + shown_side)
    plt.tight_layout()
    plt.savefig(f'{output_dir}inspect_wavefront/band_speed_{abs_frames[0]}_{abs_frames[-1]}.png')

    # Validate peak speed calculation
    fig, ax = plt.subplots(2)

    band_peaks_smoothed[refl_mask] = np.nan
    ax[0].plot(abs_frames, band_peaks_smoothed)
    ax[0].set_xlabel('Absolute frame')
    ax[0].set_ylabel('Band peak x position (m)')

    vx_band[refl_mask] = np.nan
    ax[1].plot(abs_frames, vx_band)
    ax[1].set_xlabel('Absolute frame')
    ax[1].set_ylabel(r'$v_p$ (m/s)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}inspect_wavefront/validate_band_speed_{abs_frames[0]}_{abs_frames[-1]}.png')

def plot_speed_density_gradient(
    ds: xr.Dataset,
    *,
    output_dir: str | Path | None = None,
    x_var: str = "x_raw",
    v_var: str = "v_high_ord",
    d_var: str = "density_metric_0.15",
    grad_x_var: str = "grad_density_x_metric_0.15",
    grad_x_vars: list[str] | None = None,
    fps: float = 5,
    n_bins: int = 20,
    v_thresh: float = 0.004,
    reflection_exclusion_fraction: float = 0.25,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    high_ord_params: dict | None = None,
    figsize: tuple[float, float] = (15, 9),
):
    """
    Plot speed and signed x velocity against local density and its x gradient.

    Requires the existing project helpers:
        find_reflections(ds, fps)
        smooth_nantolerant(...)

    grad_x_vars can list multiple gradient variables to compare, for example
    gradient estimates calculated at different radii or smoothing scales.
    """
    

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    if high_ord_params is None:
        high_ord_params = {"dt": 1, "num_iterations": 1, "order": 4}

    if grad_x_vars is None:
        grad_x_vars = [grad_x_var]
    elif grad_x_var not in grad_x_vars:
        grad_x_vars = [grad_x_var, *grad_x_vars]

    # Put all variables in the same (id, frame) order.
    speed = ds[v_var].transpose("id", "frame").values
    density = ds[d_var].transpose("id", "frame").values
    x = ds[x_var].transpose("id", "frame").values
    frame_values = ds["frame"].values.astype(float)

    if len(frame_values) != speed.shape[1]:
        raise ValueError("Frame coordinate and data arrays have different lengths.")

    # Reflection helper returns reflection locations in relative frame indices.
    rel_reflections, _, period, _ = find_reflections(ds, fps)
    rel_frames = np.arange(speed.shape[1])
    near_reflection = np.zeros(speed.shape[1], dtype=bool)

    if len(rel_reflections):
        near_reflection = (
            np.min(np.abs(rel_frames[:, None] - np.asarray(rel_reflections)), axis=1)
            < period * reflection_exclusion_fraction
        )
    away_from_reflection = ~near_reflection

    # Compute vx
    vx = np.full_like(speed, np.nan) # (n_ids, frame)
    for i, row in enumerate(x):
        _, vx_temp = smooth_nantolerant(row, pynumdiff.finite_difference.finitediff, high_ord_params, min_tracklet_length=high_ord_params['order'] + 1, two_outputs = True)
        vx[i,:] = vx_temp * fps

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    ax_density, ax_vx, ax_direction, ax_fraction, ax_counts, ax_distribution = (
        axes.ravel()
    )

    # Panel 1: speed versus density, with occupancy and quantiles.
    valid_sd = (
        np.isfinite(speed)
        & np.isfinite(density)
        & away_from_reflection[None, :]
    )
    if valid_sd.any():
        d_hi = np.nanquantile(density[valid_sd], 0.99)
        v_hi = np.nanquantile(speed[valid_sd], 0.99)
        d_edges = np.linspace(0, d_hi, n_bins + 1)
        v_edges = np.linspace(0, v_hi, n_bins + 1)
        d_centers = (d_edges[:-1] + d_edges[1:]) / 2

        hist = ax_density.hist2d(
            density[valid_sd],
            speed[valid_sd],
            bins=(d_edges, v_edges),
            cmap="viridis",
            norm=LogNorm(),
        )[3]

        for q in quantiles:
            curve = np.full(n_bins, np.nan)
            for j in range(n_bins):
                in_bin = (
                    valid_sd
                    & (density >= d_edges[j])
                    & (
                        (density < d_edges[j + 1])
                        if j < n_bins - 1
                        else (density <= d_edges[j + 1])
                    )
                )
                if in_bin.any():
                    curve[j] = np.nanquantile(speed[in_bin], q)
            ax_density.plot(d_centers, curve, label=f"q{q:.0%}")

        moving = valid_sd & (speed > v_thresh)
        moving_median = np.full(n_bins, np.nan)
        for j in range(n_bins):
            in_bin = (
                moving
                & (density >= d_edges[j])
                & (
                    (density < d_edges[j + 1])
                    if j < n_bins - 1
                    else (density <= d_edges[j + 1])
                )
            )
            if in_bin.any():
                moving_median[j] = np.nanmedian(speed[in_bin])

        ax_density.plot(
            d_centers,
            moving_median,
            color="white",
            linestyle="--",
            linewidth=2,
            label="Moving median",
        )
        fig.colorbar(hist, ax=ax_density, label="Observations (log scale)")
    else:
        ax_density.text(0.5, 0.5, "No valid observations", ha="center")

    ax_density.set(
        xlabel=f"{d_var} (density)",
        ylabel=f"{v_var} (speed)",
        title="Speed versus density",
    )
    ax_density.legend(fontsize=8)

    # Use one common gradient for panels 2–5; extra scales appear in panel 6.
    grad_arrays = {
        name: ds[name].transpose("id", "frame").values for name in grad_x_vars
    }
    grad = grad_arrays[grad_x_var]

    valid = (
        np.isfinite(grad)
        & np.isfinite(vx)
        & np.isfinite(speed)
        & away_from_reflection[None, :]
    )
    moving_x = valid & (np.abs(vx) > v_thresh)
    positive_x = valid & (vx > v_thresh)
    negative_x = valid & (vx < -v_thresh)

    if valid.any():
        g_lo, g_hi = np.nanquantile(grad[valid], [0.01, 0.99])
        if not np.isfinite(g_lo + g_hi) or g_lo == g_hi:
            g_lo, g_hi = np.nanmin(grad[valid]), np.nanmax(grad[valid])
        g_edges = np.linspace(g_lo, g_hi, n_bins + 1)
        g_centers = (g_edges[:-1] + g_edges[1:]) / 2

        vx_lo, vx_hi = np.nanquantile(vx[valid], [0.01, 0.99])
        vx_edges = np.linspace(vx_lo, vx_hi, n_bins + 1)

        # Panel 2: signed velocity distribution and conditional quantiles.
        h2 = ax_vx.hist2d(
            grad[valid],
            vx[valid],
            bins=(g_edges, vx_edges),
            cmap="viridis",
            norm=LogNorm(),
        )[3]

        for q in quantiles:
            curve = np.full(n_bins, np.nan)
            for j in range(n_bins):
                in_bin = valid & (grad >= g_edges[j]) & (grad < g_edges[j + 1])
                if in_bin.any():
                    curve[j] = np.nanquantile(vx[in_bin], q)
            ax_vx.plot(g_centers, curve, label=f"q{q:.0%}")

        moving_median = np.full(n_bins, np.nan)
        for j in range(n_bins):
            in_bin = (
                moving_x
                & (grad >= g_edges[j])
                & (
                    (grad < g_edges[j + 1])
                    if j < n_bins - 1
                    else (grad <= g_edges[j + 1])
                )
            )
            if in_bin.any():
                moving_median[j] = np.nanmedian(speed[in_bin])

        ax_vx.plot(
            g_centers,
            moving_median,
            color="white",
            linestyle="--",
            linewidth=2,
            label="Moving median",
        )

        fig.colorbar(h2, ax=ax_vx, label="Observations (log scale)")
        ax_vx.set(
            xlabel=f"{grad_x_var} (density gradient)",
            ylabel="$v_x$ (m/s)",
            title="$v_x$ distribution by density gradient",
        )
        ax_vx.legend(fontsize=8)

        # Panel 3: direction-specific fractions among all valid observations.
        def binned_fraction(mask):
            numerator = np.histogram(grad[mask], bins=g_edges)[0].astype(float)
            denominator = np.histogram(grad[valid], bins=g_edges)[0].astype(float)
            return np.divide(
                numerator,
                denominator,
                out=np.full(n_bins, np.nan),
                where=denominator > 0,
            )

        ax_direction.plot(
            g_centers, binned_fraction(positive_x), label="$v_x > v_{thresh}$"
        )
        ax_direction.plot(
            g_centers, binned_fraction(negative_x), label="$v_x < -v_{thresh}$"
        )
        ax_direction.plot(
            g_centers,
            binned_fraction(valid & (np.abs(vx) <= v_thresh)),
            label="Below threshold",
        )
        ax_direction.set(
            xlabel=f"{grad_x_var} (density gradient)",
            ylabel="Fraction of observations",
            ylim=(0, 1),
            title="Direction and stopping by gradient",
        )
        ax_direction.legend(fontsize=8)

        # Panel 4: fraction moving along x, regardless of direction.
        ax_fraction.plot(
            g_centers,
            binned_fraction(moving_x),
            color="black",
            marker="o",
            markersize=3,
        )
        ax_fraction.set(
            xlabel=f"{grad_x_var} (density gradient)",
            ylabel="Fraction with $|v_x| > v_{thresh}$",
            ylim=(0, 1),
            title="Probability of x movement",
        )

        # Panel 5: counts per bin, so sparse tails are visible.
        total_counts = np.histogram(grad[valid], bins=g_edges)[0]
        moving_counts = np.histogram(grad[moving_x], bins=g_edges)[0]
        ax_counts.bar(
            g_centers,
            total_counts,
            width=np.diff(g_edges),
            color="0.75",
            label="All valid",
        )
        ax_counts.bar(
            g_centers,
            moving_counts,
            width=np.diff(g_edges),
            color="tab:blue",
            alpha=0.8,
            label="Moving in x",
        )
        ax_counts.set(
            xlabel=f"{grad_x_var} (density gradient)",
            ylabel="Observations",
            title="Observations per gradient bin",
        )
        ax_counts.legend(fontsize=8)

        # Panel 6: compare gradient scales, using moving/stopped distributions.
        for name, values in grad_arrays.items():
            state_valid = (
                np.isfinite(values)
                & np.isfinite(vx)
                & away_from_reflection[None, :]
            )
            state_moving = state_valid & (np.abs(vx) > v_thresh)
            if state_moving.any():
                lo, hi = np.nanquantile(values[state_valid], [0.01, 0.99])
                edges = np.linspace(lo, hi, n_bins + 1)
                ax_distribution.hist(
                    values[state_moving],
                    bins=edges,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    label=f"{name}: moving",
                )
                ax_distribution.hist(
                    values[state_valid & ~state_moving],
                    bins=edges,
                    density=True,
                    histtype="step",
                    linestyle="--",
                    label=f"{name}: below threshold",
                )

        ax_distribution.set(
            xlabel="Density gradient estimate",
            ylabel="Probability density",
            title="Gradient values by movement state",
        )
        ax_distribution.legend(fontsize=7)

        # Avoid implying that sparsely populated tail bins are well estimated.
        for axis in (ax_vx, ax_direction, ax_fraction, ax_counts):
            axis.set_xlim(g_lo, g_hi)
    else:
        for axis in (ax_vx, ax_direction, ax_fraction, ax_counts, ax_distribution):
            axis.text(0.5, 0.5, "No valid observations", ha="center", va="center")
            axis.set_axis_off()

    fig.suptitle(
        f"Speed, density, and x-gradient relationships "
        f"(frames {int(frame_values[0])}–{int(frame_values[-1])})",
        fontsize=14,
    )

    if output_dir is not None:
        output_path = Path(output_dir) / "inspect_wavefront"
        output_path.mkdir(parents=True, exist_ok=True)
        filename = (
            f"speed_density_gradient_{d_var}_{grad_x_var}_"
            f"{int(frame_values[0])}_{int(frame_values[-1])}.png"
        )
        fig.savefig(output_path / filename, dpi=200, bbox_inches="tight")

    return fig, axes

def plot_speed_density_gradient(
    ds: xr.Dataset,
    *,
    output_dir: str | None = None,
    x_var: str = "x_raw",
    y_var: str = "y_raw",
    v_var: str = "v_high_ord",
    d_var: str = "density_metric_0.25",
    grad_x_var: str = "grad_density_x_metric_0.25",
    grad_y_var: str = "grad_density_y_metric_0.25",
    n_bins: int = 15,
    v_thresh: float = 0.004,
    fps: float = 5,
    high_ord_params: dict | None = None,
    reflection_exclusion_fraction: float = 0.25,
    figsize: tuple[float, float] = (16, 8),
):
    """Plot speed/density and movement/density-gradient relationships.

    Reflection-adjacent frames are excluded from all panels. The x-axis bins
    span the 1st–99th percentile of the relevant variable, as in the earlier
    plotting function.

    Requires project helpers:
        find_reflections(...)
        smooth_nantolerant(...)
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    if high_ord_params is None:
        high_ord_params = {
            "dt": 1,
            "num_iterations": 1,
            "order": 4,
        }

    # Load arrays in a consistent order: (id, frame).
    speed = ds[v_var].transpose("id", "frame").values
    density = ds[d_var].transpose("id", "frame").values
    x = ds[x_var].transpose("id", "frame").values
    y = ds[y_var].transpose("id", "frame").values
    grad_x = ds[grad_x_var].transpose("id", "frame").values
    grad_y = ds[grad_y_var].transpose("id", "frame").values
    abs_frames = ds.frame.values.astype(int)

    print('1')

    # Exclude frames within the chosen interval around a reflection.
    rel_reflections, _, period, _ = find_reflections(ds, fps)
    rel_frames = np.arange(len(abs_frames))

    if len(rel_reflections):
        near_reflection = (
            np.min(
                np.abs(rel_frames[:, None] - np.asarray(rel_reflections)),
                axis=1,
            )
            < period * reflection_exclusion_fraction
        )
    else:
        near_reflection = np.zeros(len(abs_frames), dtype=bool)

    keep_frame = ~near_reflection

    # Compute vx and vy following the existing pynumdiff calculation.
    dt = ds["frame"].diff("frame").values / fps
    params = dict(high_ord_params)

    vx = np.full_like(x, np.nan, dtype=float)
    vy = np.full_like(y, np.nan, dtype=float)

    for i, (x_row, y_row) in enumerate(zip(x, y)):
        _, vx_temp = smooth_nantolerant(
            x_row,
            pynumdiff.finite_difference.finitediff,
            params,
            min_tracklet_length=params["order"] + 1,
            two_outputs=True,
        )
        _, vy_temp = smooth_nantolerant(
            y_row,
            pynumdiff.finite_difference.finitediff,
            params,
            min_tracklet_length=params["order"] + 1,
            two_outputs=True,
        )

        frame_dt = np.append([1 / fps], dt)
        vx[i, :] = vx_temp / frame_dt
        vy[i, :] = vy_temp / frame_dt

    print('2')

    # Gradient magnitude and velocity projected onto the gradient direction.
    grad_mag = np.hypot(grad_x, grad_y)
    print('2.01')
    v_parallel = np.divide(
        vx * grad_x + vy * grad_y,
        grad_mag,
        out=np.full_like(grad_mag, np.nan, dtype=float),
        where=grad_mag > 0,
    )
    print('2.02')

    # Shared helpers for robust percentile limits and binned summaries.
    def percentile_edges(values, mask, bins):
        selected = values[mask & np.isfinite(values)]
        if selected.size == 0:
            return None
        lo, hi = np.nanquantile(selected, [0.01, 0.99])
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if lo == hi:
            pad = max(abs(lo) * 1e-6, 1e-9)
            lo, hi = lo - pad, hi + pad
        return np.linspace(lo, hi, bins + 1)

    def binned_median(x_values, y_values, mask, edges):
        valid = mask & np.isfinite(x_values) & np.isfinite(y_values)

        x = x_values[valid]
        y = y_values[valid]

        medians, _, _ = binned_statistic(
            x,
            y,
            statistic="median",
            bins=edges,
        )
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, medians

    def moving_fraction(x_values, moving_mask, valid_mask, edges):
        total = np.histogram(x_values[valid_mask], bins=edges)[0]
        moving = np.histogram(x_values[moving_mask], bins=edges)[0]
        fraction = np.divide(
            moving,
            total,
            out=np.full(total.shape, np.nan, dtype=float),
            where=total > 0,
        )
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, fraction

    def add_histogram_panels(ax_hist, ax_pdf, values, moving_mask, valid_mask,
                             edges, xlabel, title_prefix):
        all_values = values[valid_mask]
        moving_values = values[moving_mask]

        ax_hist.hist(
            all_values,
            bins=edges,
            alpha=0.6,
            label="All valid observations",
        )
        ax_hist.hist(
            moving_values,
            bins=edges,
            alpha=0.6,
            label="Moving locusts",
        )
        ax_hist.set(
            xlabel=xlabel,
            ylabel="Observations",
            title=f"{title_prefix}: histogram",
        )
        ax_hist.legend(fontsize=8)

        if all_values.size:
            ax_pdf.hist(
                all_values,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=2,
                label="All valid observations",
            )
        if moving_values.size:
            ax_pdf.hist(
                moving_values,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=2,
                label="Moving locusts",
            )
        ax_pdf.set(
            xlabel=xlabel,
            ylabel="Probability density",
            title=f"{title_prefix}: probability density",
        )
        ax_pdf.legend(fontsize=8)

    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)

    # Top row: density. 
    valid_density = (
        np.isfinite(density)
        & np.isfinite(speed)
        & keep_frame[None, :]
    )
    print('2.04')
    moving_speed = valid_density & (speed > v_thresh)
    print('2.05')
    density_edges = percentile_edges(density, valid_density, n_bins)
    print('2.06')
    if density_edges is not None:
        print('2.1')
        density_centers, all_speed_median = binned_median(
            density, speed, valid_density, density_edges
        )
        print('2.2')
        _, moving_speed_median = binned_median(
            density, speed, moving_speed, density_edges
        )

        axes[0, 0].plot(
            density_centers, all_speed_median, label="All locusts"
        )
        axes[0, 0].plot(
            density_centers, moving_speed_median, label="Moving locusts"
        )
        axes[0, 0].set(
            xlabel=rf"{d_var}",
            ylabel="Speed (m/s)",
            title="Speed by density",
        )
        axes[0, 0].legend()

        print('2.3')
        centers, fraction = moving_fraction(
            density, moving_speed, valid_density, density_edges
        )
        print('2.4')
        axes[0, 1].plot(centers, fraction, marker="o", markersize=3)
        axes[0, 1].set(
            xlabel=rf"{d_var}",
            ylabel="Moving locusts (%)",
            ylim=(0, 1),
            title="Movement probability by density",
        )

        add_histogram_panels(
            axes[0, 2],
            axes[0, 3],
            density,
            moving_speed,
            valid_density,
            density_edges,
            xlabel=rf"{d_var}",
            title_prefix="Density",
        )
    else:
        for ax in axes[0]:
            ax.text(0.5, 0.5, "No valid observations", ha="center")

    print('3')

    # Bottom row: density-gradient magnitude and projected velocity.
    valid_gradient = (
        np.isfinite(grad_mag)
        & np.isfinite(v_parallel)
        & np.isfinite(speed)
        & keep_frame[None, :]
    )
    moving_gradient = valid_gradient & (speed > v_thresh)

    grad_edges = percentile_edges(grad_mag, valid_gradient, n_bins)

    if grad_edges is not None:
        grad_centers = (grad_edges[:-1] + grad_edges[1:]) / 2
        parallel_edges = percentile_edges(
            v_parallel, valid_gradient, n_bins
        )

        # Count observations in each (gradient magnitude, v_parallel) bin.
        counts, grad_edges, vpar_edges = np.histogram2d(
            grad_mag[valid_gradient & (speed > v_thresh)],
            v_parallel[valid_gradient & (speed > v_thresh)],
            bins=(grad_edges, parallel_edges),
        )

        # Normalize each gradient-magnitude column to sum to 1.
        column_totals = counts.sum(axis=1, keepdims=True)
        probability = np.divide(
            counts,
            column_totals,
            out=np.full_like(counts, np.nan, dtype=float),
            where=column_totals > 0,
        )

        heat = axes[1, 0].pcolormesh(
            grad_edges,
            vpar_edges,
            probability.T,
            cmap="viridis",
            shading="auto",
            vmin=0,
            vmax=np.nanmax(probability),
        )
        fig.colorbar(heat, ax=axes[1, 0], label=r"$P(v_\parallel \mid |\nabla\rho|\ \mathrm{bin})$")

        axes[1, 0].set(
            xlabel=r"$|\nabla \rho|$",
            ylabel=r"$v_{\parallel}$ (m/s)",
            title="Alignment (moving) by gradient magnitude",
        )

        centers, fraction = moving_fraction(
            grad_mag, moving_gradient, valid_gradient, grad_edges
        )
        axes[1, 1].plot(centers, fraction, marker="o", markersize=3)
        axes[1, 1].set(
            xlabel=r"$|\nabla \rho|$",
            ylabel="Moving locusts (%)",
            ylim=(0, 1),
            title="Movement probability by gradient magnitude",
        )

        add_histogram_panels(
            axes[1, 2],
            axes[1, 3],
            grad_mag,
            moving_gradient,
            valid_gradient,
            grad_edges,
            xlabel=r"$|\nabla \rho|$",
            title_prefix="Gradient magnitude",
        )
    else:
        for ax in axes[1]:
            ax.text(0.5, 0.5, "No valid observations", ha="center")

    print('4')
    fig.suptitle(
        f"Speed and density relationships, frames "
        f"{abs_frames[0]}–{abs_frames[-1]}",
        fontsize=15,
    )

    if output_dir is not None:
        output_path = Path(output_dir) / "inspect_wavefront"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path
            / f"speed_density_gradient_2_{v_var}_{d_var}_{grad_x_var}_{grad_y_var}_{abs_frames[0]}_{abs_frames[-1]}.png",
            dpi=200,
            bbox_inches="tight",
        )

    return fig, axes


if __name__ == '__main__':
    ds_name = f'/bb_kp/20230329/5900_11900_5.0Hz'
    ds = load_preprocessed_data(ds_name + '.hdf5')
    smooth_dict = {'raw': None, 'high_ord':{'dt': 1, 'num_iterations': 1, 'order': 4}}#, 'high_ord_ang':{'dt': 1, 'num_iterations': 1, 'order': 4}}#, 'spline_ang': {'degree': 2, 's': 0.001}}

    # print(ds.data_vars)
    # ds = ds.drop_vars([var for var in ds.data_vars if var not in ['x_raw', 'y_raw', 'theta']])

    # for var in ds.data_vars:
    #     ds[var] = ds[var].transpose('id', 'frame')

    # print(ds.data_vars)

    # ds = get_local_env(ds, 'metric', 0.1, np.array([0.01,0]), 2.38)
    param_dict = smooth_dict | {'metric':0.1, 'arena_center_x': 0.01, 'arena_center_y': 0.0, 'arena_radius':2.38, 'grad_radius':0.1}
    # grad = local_density_gradient(ds, radius = 0.1)
    # ds['grad_density_x_metric_0.1'] = (('id', 'frame'), grad[0,:,:])
    # ds['grad_density_y_metric_0.1'] = (('id', 'frame'), grad[1,:,:])

    # ds = compute_speed(ds, smooth_dict, fs = 5)
    # ds = compute_ang_speed(ds, smooth_dict, fs = 5)
    # save_ds(ds, '/bb_kp/20230329/5900_11900_5.0Hz', {})
    # plot_smoothed_coords(ds, '/output/20230329/both/', smooth_dict, 0)
    # plot_ang_speed(ds, '/output/20230329/both/', 'high_ord', 0, 300, fs = 5)
    v_thresh = 0.00422991
    # plot_speed_hists(ds, '/output/20230329/both/', ['raw', 'high_ord'], v_thresh = v_thresh)

    # Compute v threshold
    # v_thresh, boot = bootstrap_threshold((ds['v_high_ord'].values)[np.isfinite(ds['v_high_ord'])].ravel())
    # print(v_thresh, boot)

    # Compute cluster stats
    # print(ds.coords)
    # compute_whole_batch_clusters(ds, '/output/20230329/both/', v_thresh = v_thresh)

    # clusters = load_cluster_ids('/output/20230329/both/clusters/5900_11900_5.0Hz.h5')
    # find_band_speed(ds, clusters, '/output/20230329/both/')    
    # band_by_density(ds, '/output/20230329/both/', v_thresh = v_thresh)
    # speed_by_density(ds, '/output/20230329/both/', v_thresh = v_thresh)
    plot_speed_density_gradient(ds, output_dir = '/output/20230329/both/', v_thresh = v_thresh)

    # Find centroid of biggest cluster at all frames
    # Compute speed of biggest cluster at all frames
    # Compute speed of ids in cluster relative to distance from wavefront