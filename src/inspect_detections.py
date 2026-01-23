import numpy as np
import h5py
import matplotlib.pyplot as plt

def detections_over_time(path_h5: str):
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

path_h5 = './tracking/h5s/10K_full_2.hdf5'

detections_over_time(path_h5)