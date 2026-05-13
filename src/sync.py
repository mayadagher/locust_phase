'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


from data_handling import *

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

# Load mocap data
path = '/mocap/20230329/tsvs/10K_Marching_0060.tsv'

with open(path, 'r') as f:
    for i, line in enumerate(f, 1):
        print(f'Line {i}: {line.strip()}')
        if i > 20:
            break

# df = pd.read_csv(path, sep='\t')
# print(df.head())
