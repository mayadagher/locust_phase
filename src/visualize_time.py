'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_handling import load_preprocessed_data
from helper_fns import *

'''_____________________________________________________PLOTTING FUNCTIONS____________________________________________________________'''

def bb_vs_kp_detections_over_time(bb_h5:str, kp_preprocessed_h5:str, plot:bool=False, plots_path:str | None = None):
    bb_detections = []
    with h5py.File(bb_h5, 'r') as f:
        for i in range(1, len(f.keys()) + 1):
            frame = f[f'coords_{i}']
            bb_detections.append(frame.shape[1])

    kp_detections = []
    with h5py.File(kp_preprocessed_h5, 'r') as f:
        for i in range(len(f.keys())):
            num_detections = len(f[f'f{i}/head']) # Number of heads and tails always equal
            kp_detections.append(num_detections)
    
    if plot:
        plt.plot(bb_detections, label = 'BB')
        plt.plot(kp_detections, label = 'Keypoint')
        plt.legend()
        plt.xlabel('Frame', fontsize=17)
        plt.ylabel('Number of detections', fontsize=17)
        plt.savefig(f'{plots_path}detections_over_time.png')

    return bb_detections, kp_detections

def tracklets_over_time(smooth_name:str, num_batches:int, h5_in:str, num_detections:list, output_dir:str):
    """
    Calculate the number of tracklets over all batches based on the specified smooth variable. Batches have some overlap in time.
    
    Parameters:
    smooth_name (str): The name of the smooth variable to use for determining tracklets.
    num_batches (int): The number of batches to process.
    """
    counts_list = []
    for batch_i in tqdm(range(num_batches), 'Counting tracklets over time across batches'):
        
        # LOAD PREPROCESSED DATA
        ds_load_name = f'{h5_in}batch_{batch_i}_5.0Hz.hdf5'
        ds = load_preprocessed_data(ds_load_name)
        
        # Determine valid frames where smooth is not NaN
        valid = ds[f'x_{smooth_name}'].notnull() # Using x coordinate to avoid issues of differentiation causing NaNs in smooth
        num_tracklets = valid.sum(dim='id')

        counts_list.append(num_tracklets)
    
    # Merge counts from overlaps in batches to make continuous DataArray
    merged = xr.concat(counts_list, dim='frame', join='override')
    merged = merged.drop_duplicates(dim='frame', keep='first')

    # Plot counts from all batches and detections
    _, ax = plt.subplots(2, 1, figsize=(12, 10), sharex = True)

    ax[0].plot(merged['frame'], merged, color = 'r', label = 'TRex tracklets')
    ax[0].plot(np.arange(len(num_detections)), num_detections, color='b', alpha=0.5, label='BB detections')
    ax[0].set_ylabel('Number of tracklets/detections', fontsize=14)
    ax[0].legend(fontsize=14)
    
    ax[1].plot(np.arange(len(num_detections)), num_detections - merged, color='k', alpha=0.5)
    ax[1].set_ylabel('Residuals', fontsize=14)
    ax[1].set_xlabel('Frame', fontsize=14)
    
    plt.savefig(f'{output_dir}tracklets_over_time_{smooth_name}.png')

