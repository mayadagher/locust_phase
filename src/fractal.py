'''_____________________________________________________IMPORTS____________________________________________________________'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from data_handling import *

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def fit_square_in_arena(arena_center_m:np.ndarray, arena_radius_m:float):
    '''Return corners of biggest square that fits in the arena. This square can have any orientation but for ease will be the one aligned with the x and y axis.'''

    magnitude = arena_radius_m/np.sqrt(2)
    vecs = [np.array([-1, -1]), np.array([-1, 1]), np.array([1, -1]), np.array([1, 1])]

    return np.array([arena_center_m + magnitude*vec for vec in vecs]) # x0y0, x0y1, x1y0, x1y1

def compute_bin_occupancy(corners:np.ndarray, Linv:int, positions:np.ndarray):
    '''Return occupancies of each bin, defined by the bin length L, which is a fraction of the total region length (or more precisely, the inverse, which is number of boxes per side).'''

    # Extract only valid positions
    positions = positions[np.all(np.isfinite(positions), axis = 1)]

    # Determine counts in all boxes definted by corners and Linv
    bin_edges_x = np.linspace(corners[0][0], corners[2][0], Linv + 1)
    bin_edges_y = np.linspace(corners[0][1], corners[1][1], Linv + 1)
    counts, _, _ = np.histogram2d(positions[:,0], positions[:,1], bins = [bin_edges_x, bin_edges_y])
    
    return counts.astype(int)

def minkowski_bouligand_dim(corners:np.ndarray, positions:np.ndarray, max_Linv_exp:int = 15):
    '''Compute the Minkowski-Bouligand dimension by looking at the ratio of the logarithms of number of occupied boxes and box length for a single frame.
    Since the positions are discrete, counting is stopped when the number of occupied bins is equal to the number of individuals in the whole rectangular region.'''

    # Create array of Linv values that will be explored
    Linvs = (2**np.arange(max_Linv_exp + 1)).astype(int)

    # Iterate over Linv values and count how many bins are actually occupied
    Ns = np.full(len(Linvs), np.nan)

    max_reached = False
    max_N = np.nan # Will be given a value before it's actually used
    for i, Linv in enumerate(Linvs):

        # Compute number of occupied bins using 2D histogram
        counts = compute_bin_occupancy(corners, Linv, positions)

        # Find maximum number of individuals in the square
        if Linv == 1:
            max_N = counts[0][0]
            N = 1
        # Stop using 2D histogram if N is the same as Max)N
        else:
            N = np.sum(counts > 0)
            if N == max_N:
                max_reached = True

        Ns[i] = N

        # N will always stay the same because it is discrete, therfore the ratio becomes meaningless
        if max_reached:
            break


    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.log(Linvs), np.log(Ns))
    ax[1].plot(np.log(Linvs), np.log(Ns)/np.log(Linvs))
    plt.savefig('minkowski_trial.png')

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def correlation_dimension(
    positions,
    n_radii=30,
    fit_range=(0.01, 0.2),
    ax=None,
):
    """
    Estimate the correlation dimension of a 2D point cloud.

    Parameters
    ----------
    positions : np.ndarray, shape (n_ids, 2)
        x/y positions.
    n_radii : int
        Number of spatial scales to evaluate.
    fit_range : tuple
        Range of correlation-sum values C(r) used for the fit.
        Excludes the small-r discreteness regime and large-r finite-size regime.
    ax : matplotlib axis, optional
        Axis on which to plot the correlation sum.

    Returns
    -------
    v : float
        Estimated correlation dimension.
    """

    xy = np.asarray(positions, dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]

    n = len(xy)

    if n < 3:
        return np.nan

    tree = cKDTree(xy)

    # Characteristic system size.
    extent = np.ptp(xy, axis=0)
    L = np.linalg.norm(extent)

    if L == 0:
        return 0.0

    # Approximate typical inter-individual spacing.
    nn_dist, _ = tree.query(xy, k=2)
    r_min = np.median(nn_dist[:, 1])

    # Stay below the system-size regime.
    r_max = 0.5 * L

    if r_min <= 0 or r_min >= r_max:
        return np.nan

    radii = np.geomspace(r_min, r_max, n_radii)

    # count_neighbors includes self-pairs and counts both i,j and j,i.
    counts = np.asarray(
        tree.count_neighbors(tree, radii),
        dtype=float,
    )

    pairs = (counts - n) / 2

    # Normalize by total number of unique pairs.
    C = pairs / (n * (n - 1) / 2)

    valid = (
        (C >= fit_range[0]) &
        (C <= fit_range[1]) &
        (C > 0)
    )

    if valid.sum() < 2:
        return np.nan

    log_r = np.log(radii)
    log_C = np.log(C)

    v, intercept = np.polyfit(
        log_r[valid],
        log_C[valid],
        1,
    )

    # Plot.
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    ax.plot(log_r, log_C, "o-", ms=4, label="Correlation sum")

    ax.plot(
        log_r[valid],
        v * log_r[valid] + intercept,
        lw=2,
        label=fr"Fit: $v={v:.2f}$",
    )

    ax.set_xlabel(r"$\log r$")
    ax.set_ylabel(r"$\log C(r)$")
    ax.set_title("Correlation dimension")
    ax.legend()
    plt.savefig('corr_dim_trial.png')
    print(v)
    return v

arena_center_m = np.array([0.01, 0])
arena_radius_m = 2.38
corners = fit_square_in_arena(arena_center_m, arena_radius_m)

batch_idx = 1
subsample = 1
h5_prep = f'/keypoints/dewarped/20230329_preprocessed_complete_dewarped_batch_{batch_idx}_{round(5/subsample, 2)}Hz.hdf5'
ds = load_preprocessed_data(h5_prep)
positions = np.array([ds.centroid_x.values, ds.centroid_y.values])[:,200,:].T # max_n_ids, 2

# minkowski_bouligand_dim(corners, positions)
correlation_dimension(positions)