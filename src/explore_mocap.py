import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_to_mocap_batch = '/mocap/20230329/csvs/10K_Marching_0049.csv'
mocap_df = pd.read_csv(path_to_mocap_batch)

ids, counts = np.unique(mocap_df['particle_id'], return_counts = True)

fig, _ = plt.subplots()

bins = np.linspace(0, 5, 51)
bin_centers = (bins[:-1] + bins[1:])/2
cnts, _ = np.histogram(counts, bins = 10**bins)
plt.plot(bin_centers, np.cumsum(cnts)[-1] - np.cumsum(cnts))
plt.xlabel('Minimum log (10) MoCap lengths')
plt.ylabel('Cumulative sum')
plt.savefig('mocap_lengths.png')
plt.close(fig)