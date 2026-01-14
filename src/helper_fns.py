'''_____________________________________________________IMPORTS____________________________________________________________'''


import numpy as np
import xarray as xr

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def run_lengths(arr, val = None):
    """
    Return lengths of either:
    a) all consecutive non-NaN runs within rows.
    b) all consecutive runs of a specific value within rows.
    Runs do NOT continue across rows.
    """
    arr = np.asarray(arr)

    if val is None:
        valid = ~np.isnan(arr)
    else:
        valid = (arr == val)

    # Pad with False at both ends along frame axis
    padded = np.pad(valid, ((0, 0), (1, 1)), constant_values=False)

    # Find run start and end indices
    diff = np.diff(padded.astype(int), axis=1)

    starts = diff == 1
    ends   = diff == -1

    # Indices of starts and ends
    start_idx = np.where(starts)
    end_idx   = np.where(ends)

    # Run lengths = end - start
    lengths = end_idx[1] - start_idx[1]

    return lengths

def run_lengths_labeled(arr):
    """
    Count lengths of consecutive identical non-nan values. It is assumed that the last label of one row cannot be the first label of the next row.
    """
    x = arr.values.ravel()
    x = x[~np.isnan(x)]

    # If empty:
    if x.size == 0:
        raise ValueError('Array is empty.')

    # Find run boundaries
    changes = np.diff(x) != 0
    idx = np.concatenate(([0], np.where(changes)[0] + 1, [len(x)]))

    # Run lengths:
    lengths = np.diff(idx).astype(int)

    return lengths

def autocorr_fft(x, tau_max: int = int(1e6), normalize=True, demean=True):

    """
    Fast autocorrelation using FFT.
    
    Parameters
    ----------
    x : 1D array
    normalize : bool
        If True, normalize so R(0) = 1
    demean : bool
        Subtract mean before computing
    
    Returns
    -------
    acf : 1D array
        Autocorrelation for lags >= 0
    """
    x = np.asarray(x)
    n = len(x)

    if demean:
        x = x - np.mean(x)

    # Zero-pad to avoid circular correlation
    nfft = 1 << (2*n - 1).bit_length()
    fx = np.fft.fft(x, n=nfft)
    acf = np.fft.ifft(fx * np.conj(fx)).real
    acf = acf[:n]

    if normalize:
        if np.std(x) != 0:
            acf /= acf[0]
        
    return acf[:tau_max]

def list_long_tracklets(da: xr.DataArray, min_tracklet_length: int = 1):
    '''Takes a DataArray of shape (id, frame) and returns a list of valid tracklets (continuous segments of data that are sufficiently long).'''

    # Loop over ids
    all_valid_trajs = []
    for i in range(da.sizes["id"]):
        vals = da.isel(id=i).values
        frames = da["frame"].values

        # Mask out invalid values, e.g. gaps
        mask = ~np.isnan(vals)

        if not np.any(mask):
            continue

        valid_vals = vals[mask]
        valid_frames = frames[mask]

        # Find continuous frame segments
        breaks = np.where(np.diff(valid_frames) != 1)[0] + 1
        segments = np.split(np.arange(len(valid_frames)), breaks)

        # Add to list of all segments if long enough
        for seg in segments:
            if len(seg) >= min_tracklet_length:
                all_valid_trajs.append(valid_vals[seg])

    print('Number of eligible tracklets:', len(all_valid_trajs), flush=True)

    return all_valid_trajs