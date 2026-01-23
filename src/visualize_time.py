'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import h5py
import matplotlib.pyplot as plt

from data_handling import load_preprocessed_data
from helper_fns import *

'''_____________________________________________________PLOTTING FUNCTIONS____________________________________________________________'''

def detections_over_time(path_h5:str):
    num_detections = []
    with h5py.File(path_h5, 'r') as f:
        for i in range(1, len(f.keys()) + 1):
            frame = f[f'coords_{i}']
            num_detections.append(frame.shape[1])
    
    _, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].plot(num_detections)
    ax[1].plot(100*np.array(num_detections)/np.max(num_detections))
    ax[1].set_xlabel('Frame', fontsize=17)
    ax[0].set_ylabel('Number of detections', fontsize=17)
    ax[1].set_ylabel('Percent detected (%)', fontsize=17)
    plt.savefig('./plots/20230329/all/detections_over_time.png')

    return num_detections

def tracklets_over_time(speed_name:str, num_batches:int, exp_name:str, num_detections:list):
    """
    Calculate the number of tracklets over all batches based on the specified speed variable. Batches have some overlap in time.
    
    Parameters:
    speed_name (str): The name of the speed variable to use for determining tracklets.
    num_batches (int): The number of batches to process.
    """
    counts_list = []
    for batch_i in range(num_batches):
        # LOAD PREPROCESSED DATA
        ds_load_name = f'/output/preprocessed/{exp_name}/batch_{batch_i}/traj_data.h5'
        ds = load_preprocessed_data(ds_load_name)
        print(f'Batch {batch_i} loaded.')
        
        # Determine valid frames where speed is not NaN
        valid = ds[f'x_{speed_name}'].notnull() # Using x coordinate to avoid issues of differentiation causing NaNs in speed
        num_tracklets = valid.sum(dim='id')

        counts_list.append(num_tracklets)
    
    # Merge counts from overlaps in batches to make continuous DataArray
    merged = xr.concat(counts_list, dim='frame', join='override')
    merged = merged.drop_duplicates(dim='frame', keep='first')

    # Plot counts from all batches and detections
    _, ax = plt.subplots(2, 1, figsize=(12, 10), sharex = True)

    ax[0].plot(merged['frame'], merged, color = 'r', label = 'TRex tracklets')
    ax[0].plot(np.arange(len(num_detections)), num_detections, color='b', alpha=0.5, label='Detections')
    ax[0].set_ylabel('Number of tracklets/detections', fontsize=14)
    ax[0].legend(fontsize=14)
    
    ax[1].plot(np.arange(len(num_detections)), num_detections - merged, color='k', alpha=0.5)
    ax[1].set_ylabel('Residuals', fontsize=14)
    ax[1].set_xlabel('Frame', fontsize=14)
    
    plt.savefig(f'./plots/{exp_name}/all/tracklets_over_time_{speed_name}.png')

