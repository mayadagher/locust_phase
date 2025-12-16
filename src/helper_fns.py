'''_____________________________________________________IMPORTS____________________________________________________________'''


import numpy as np

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