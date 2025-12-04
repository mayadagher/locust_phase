'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import os

'''_____________________________________________________LOAD AND SAVE FUNCTIONS____________________________________________________________'''
def load_trex_data(batch_num, file_name, load_num_ids=None):
    """
    Load TReX .npz data into an xarray.Dataset.
    
    Dimensions: id × frame
    Coordinates: 'id', 'frame'
    Data variables: x_raw, y_raw, (speed, id_prob, num_pixels,) missing
    """

    assert load_num_ids is None or load_num_ids > 0, "load_num_ids must be a positive integer."

    data_dir = f'./locust_data/trex_outputs/batch_{batch_num}/data/'
    num_ids = len(os.listdir(data_dir))
    ids = np.arange(num_ids) if load_num_ids is None else np.arange(load_num_ids)

    datasets = []
    for i in ids:
        path = os.path.join(data_dir, f"{file_name}_id{i}.npz")
        with np.load(path) as d:
            frames = d['frame']
            ds = xr.Dataset(
                {
                    'x_raw': (['frame'], d['X#wcentroid']),
                    'y_raw': (['frame'], d['Y#wcentroid']),
                    'missing': (['frame'], d['missing']),
                },
                coords={'frame': frames, 'id': i},
            )
            datasets.append(ds)

    # Concatenate along 'id' dimension
    full_ds = xr.concat(datasets, dim='id', join = 'outer')
    full_ds = full_ds.where(np.isfinite(full_ds), np.nan)
    
    return len(ids), full_ds

def load_preprocessed_data(load_name): # Load pre-processed data from h5s
    """
    Load an xarray Dataset from an HDF5 file.
    """
    ds = xr.open_dataset(load_name, engine="h5netcdf")
    return ds.load()

def save_data(ds, save_name): # Save pre-processed data to h5s
    for var in ds.data_vars:
        if ds[var].dtype == np.float64: # Losing some precision here, but saves a lot of space
            ds[var] = ds[var].astype(np.float32)
    ds.close()  # Ensure any open files are closed before saving
    encoding = {var: {'compression': 'gzip', 'compression_opts': 4} for var in ds.data_vars}
    ds.to_netcdf(f'{save_name}.h5', engine="h5netcdf", encoding=encoding)