'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import os
import h5py

'''_____________________________________________________LOAD AND SAVE FUNCTIONS____________________________________________________________'''
def load_trex_data(batch_num, file_name, load_num_ids=None):
    """
    Load TReX .npz data into an xarray.Dataset.
    
    Dimensions: id × frame
    Coordinates: 'id', 'frame'
    Data variables: x_raw, y_raw, (speed, id_prob, num_pixels,) missing
    """

    assert load_num_ids is None or load_num_ids > 0, "load_num_ids must be a positive integer."

    data_dir = f'./tracking/trex_outputs/batch_{batch_num}/data/'
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

def save_ds(ds, save_name): # Save pre-processed data to h5s
    for var in ds.data_vars:
        if ds[var].dtype == np.float64: # Losing some precision here, but saves a lot of space
            ds[var] = ds[var].astype(np.float32)
    ds.close()  # Ensure any open files are closed before saving
    encoding = {var: {'compression': 'gzip', 'compression_opts': 4} for var in ds.data_vars}
    ds.to_netcdf(f'{save_name}.h5', engine="h5netcdf", encoding=encoding)

def save_neighbours_hdf5(h5_path: str, interaction: str, param, values: list, offsets: list, data_name: str, compression="gzip"):
    """
    Saves data in CSR format to:
        /interaction/param/(nbrs/regions)/values
        /interaction/param/(nbrs/regions)/offsets
    in existing .h5 file.
    """

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    group_path = f"{interaction}/{param}/{data_name}"

    with h5py.File(h5_path, "a") as f:

        g = f.create_group(group_path)

        # Use CSR format to store neighbours
        g.create_dataset("values", data=values, compression=compression, chunks=True)
        g.create_dataset("offsets", data=offsets, compression=compression, chunks=True)

def load_neighbours_hdf5(h5_path):
    if os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            out = {}
            for interaction in f.keys(): # metric/topo/etc.
                out[interaction] = {}
                for param in f[interaction].keys(): # radii/k/etc.
                    out[interaction][param] = {}
                    for name in f[interaction][param].keys(): # nbrs/regions
                        grp = f[interaction][param][name]
                        out[interaction][param][name] = {"values": grp["values"][:], "offsets": grp["offsets"][:]}
            return out
    else:
        print(f'File {h5_path} does not exist, returning empty dictionary.')
        return {}

def save_psd_hdf5(h5_path: str, fmin: str, activity: str, coordinate: str, smooth: str, means: list, vars: list, freqs: list, quantile = None):
    """
    Save PSD data to an HDF5 file.
    activity: 'active' or 'inactive' or 'all'
    coordinate: 'x' or 'y'
    quantity: 'mean' or 'var'
    quantile: '[low, high]' or None
    """

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    if activity in ['active', 'inactive']:
        group_path = f"{fmin}/{activity}/{quantile}/{coordinate}/{smooth}"
    elif activity == 'all':
        group_path = f"{fmin}/{activity}/{coordinate}/{smooth}"
    else:
        raise ValueError("Invalid activity parameter.")

    with h5py.File(h5_path, "a") as f:
        g = f.create_group(group_path)

        # Save mean/var/freq vector
        g.create_dataset('mean', data=means, compression="gzip")
        g.create_dataset('var', data=vars, compression="gzip")
        g.create_dataset("freq", data=freqs, compression="gzip")
        print(means, vars, freqs)

def load_psds_hdf5(h5_path):
    if os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            out = {}
            for fmin in f.keys():
                out[fmin] = {}
                for activity in f[fmin].keys():
                    out[fmin][activity] = {}

                    if activity in ['active', 'inactive']: # Handle different quantiles for active/inactive
                        for quantile in f[fmin][activity].keys():
                            out[fmin][activity][quantile] = {}
                            for coordinate in f[fmin][activity][quantile].keys():
                                out[fmin][activity][quantile][coordinate] = {}
                                for smooth in f[fmin][activity][quantile][coordinate].keys():
                                    grp = f[fmin][activity][quantile][coordinate][smooth]
                                    out[fmin][activity][quantile][coordinate][smooth] = {"mean": grp["mean"][:], "var": grp["var"][:], "freq": grp["freq"][:]}

                    else:
                        for coordinate in f[fmin][activity].keys():
                            out[fmin][activity][coordinate] = {}
                            for smooth in f[fmin][activity][coordinate].keys():
                                grp = f[fmin][activity][coordinate][smooth]
                                out[fmin][activity][coordinate][smooth] = {"mean": grp["mean"][:], "var": grp["var"][:], "freq": grp["freq"][:]}
            return out
    else:
        print(f'File {h5_path} does not exist, returning empty dictionary.')
        return {}