'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import os
import h5py
import json
from pathlib import Path
from tqdm import tqdm
import csv

'''_____________________________________________________LOAD AND SAVE FUNCTIONS____________________________________________________________'''
def load_trex_data(batch_num:int, file_name:str, load_num_ids:int | None = None):
    """
    Load TReX .npz data into an xarray.Dataset.
    
    Dimensions: id × frame
    Coordinates: 'id', 'frame'
    Data variables: x_raw, y_raw, (speed, id_prob, num_pixels,) missing
    """

    assert load_num_ids is None or load_num_ids > 0, "load_num_ids must be a positive integer."

    data_dir = f'/data/batch_{batch_num}/data/'
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

def load_preprocessed_data(load_name:str): # Load pre-processed data from h5s
    """
    Load an xarray Dataset from an HDF5 file.
    """
    ds = xr.open_dataset(load_name, engine="h5netcdf")
    return ds.load()

def save_ds(ds:xr.Dataset, save_name:str, params:dict | None): # Save pre-processed data to h5s

    # Save float64 variables as float32 to save space
    for var in ds.data_vars:
        if ds[var].dtype == np.float64: # Losing some precision here, but saves a lot of space
            ds[var] = ds[var].astype(np.float32)

    # Ensure any open files are closed before saving        
    ds.close()  

    # Compress and save
    encoding = {var: {'compression': 'gzip', 'compression_opts': 4} for var in ds.data_vars}
    ds.to_netcdf(save_name, engine="h5netcdf", encoding=encoding)


    params_out = Path(save_name.split('.')[0] + '_params')
    params_out.with_suffix(".json").write_text(json.dumps(params, indent=2, sort_keys=True))

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

def load_neighbours_hdf5(h5_path:str):
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

def load_psds_hdf5(h5_path:str):
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
    

def save_across_batches_hdf5(h5_path: str, data_name_path:str, batch_num: int, data: np.ndarray, frames: np.ndarray):
    """
    Save data across several batches to an HDF5 file. Architecture is:
        /data_name_path/batch_{batch_num}/data
        /data_name_path/batch_{batch_num}/frames
    """

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    group_path = f"{data_name_path}/batch_{batch_num}/"

    with h5py.File(h5_path, "a") as f:
        g = f.require_group(group_path) # Create group if it doesn't exist

        # Save data, frames
        g.create_dataset('data', data=data, compression="gzip")
        g.create_dataset('frames', data=frames, compression="gzip")

def load_across_batches_hdf5(h5_path: str, data_name_path:str, batch_num: int):
    """
    Load data across several batches from an HDF5 file. Architecture is:
        /data_name_path/batch_{batch_num}/data
        /data_name_path/batch_{batch_num}/frames
    """

    if os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            group_path = f"{data_name_path}/batch_{batch_num}/"
            if group_path in f:
                g = f[group_path]
                data = g['data'][:]
                frames = g['frames'][:]
                return data, frames
            else:
                print(f'Group {group_path} does not exist in {h5_path}.')
                return None, None
    else:
        print(f'File {h5_path} does not exist.')
        return None, None
    
def detections_h5_to_trex_csv(h5_path:str, csv_path:str, start_frame:int = 0, end_frame:int = -1, rescale = True):

    if rescale:
        csv_path = csv_path.split('.')[0] + '_rescaled.csv'
        factor = 1920/7000
    write_header = not os.path.exists(csv_path)

    with h5py.File(h5_path, 'r') as h5file, open(csv_path, 'a') as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow(["x", "y", "frame"])

        end_frame = min(end_frame, len(h5file.keys())) if end_frame > 0 else len(h5file.keys())

        print(h5file.keys())

        for f_idx in tqdm(range(start_frame, end_frame)):
            centroids = h5file[f'f{f_idx}']['centroid']

            for x, y in centroids:
                if rescale:
                    x, y = factor*x, factor*y

                writer.writerow([x, y, f_idx])

def detections_h5_to_xr_dataset(h5_path:str, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1, rescale_factor:float = 1):

    with h5py.File(h5_path, 'r') as f:
        end_frame = min(end_frame, len(f.keys())) if end_frame is not None else len(f.keys())
        datasets = []
        max_detections = 0
        for f_idx in tqdm(range(start_frame, end_frame, subsample)):
            centroids = f[f'f{f_idx}']['centroid']
            heads = f[f'f{f_idx}']['head']
            tails = f[f'f{f_idx}']['tail']
            # conf_head_tail = f[f'f{f_idx}']['conf']

            ds = xr.Dataset(
                {
                    'centroid_x': (['id'], centroids[:, 0]*rescale_factor), # Rescale if necessary to match original image dimensions (e.g. if detections were made on downsampled images)
                    'centroid_y': (['id'], centroids[:, 1]*rescale_factor),
                    'theta': (['id'], np.arctan2(heads[:, 1] - tails[:, 1], heads[:, 0] - tails[:, 0])), # Orientation calculated from head and tail positions

                    # Potentially useful but not currently used variables - can be added back in if needed for tracking or other analyses
                    # 'head_x': (['id'], heads[:, 0]),
                    # 'head_y': (['id'], heads[:, 1]),
                    # 'tail_x': (['id'], tails[:, 0]),
                    # 'tail_y': (['id'], tails[:, 1]),
                    # 'conf_head': (['id'], conf_head_tail[:, 0]),
                    # 'conf_tail': (['id'], conf_head_tail[:, 1]),
                },
                coords={'id': np.arange(len(centroids)), 'frame': f_idx},
            )
            datasets.append(ds)
            if len(centroids) > max_detections:
                max_detections = len(centroids)

    full_ds = xr.concat(datasets, dim='frame', join='outer')
    
    return full_ds

def detections_h5_to_kp_xr(h5_path:str, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1, rescale_factor:float = 1):

    with h5py.File(h5_path, 'r') as f:
        end_frame = min(end_frame, len(f.keys())) if end_frame is not None else len(f.keys())
        datasets = []
        max_detections = 0
        for f_idx in tqdm(range(start_frame, end_frame, subsample)):
            centroids = f[f'f{f_idx}']['centroid']
            heads = f[f'f{f_idx}']['head']
            tails = f[f'f{f_idx}']['tail']
            # conf_head_tail = f[f'f{f_idx}']['conf']

            ds = xr.Dataset(
                {
                    'centroid_x': (['id'], centroids[:, 0]*rescale_factor), # Rescale if necessary to match original image dimensions (e.g. if detections were made on downsampled images)
                    'centroid_y': (['id'], centroids[:, 1]*rescale_factor),
                    'theta': (['id'], np.arctan2(heads[:, 1] - tails[:, 1], heads[:, 0] - tails[:, 0])), # Orientation calculated from head and tail positions

                    # Potentially useful but not currently used variables - can be added back in if needed for tracking or other analyses
                    'head_x': (['id'], heads[:, 0]),
                    'head_y': (['id'], heads[:, 1]),
                    'tail_x': (['id'], tails[:, 0]),
                    'tail_y': (['id'], tails[:, 1]),
                    # 'conf_head': (['id'], conf_head_tail[:, 0]),
                    # 'conf_tail': (['id'], conf_head_tail[:, 1]),
                },
                coords={'id': np.arange(len(centroids)), 'frame': f_idx},
            )
            datasets.append(ds)
            if len(centroids) > max_detections:
                max_detections = len(centroids)

    full_ds = xr.concat(datasets, dim='frame', join='outer')
    
    return full_ds

def cluster_stats_to_h5(aggregated: dict[int, list], output_path: str):
    """
    Save aggregated cluster stats to HDF5.

    Layout per absolute frame group:
        /0/
            ns, areas, medPols, varPols, medDs, varDs, meanThetas, varThetas
                -- 1D float/int arrays, one entry per cluster, no nans

            p_by_layer_data, p_by_layer_indptr    --|
            d_by_layer_data, d_by_layer_indptr      |-- CSR format for ragged
            p_from_edge_data, p_from_edge_indptr    |   layer arrays
            d_from_edge_data, d_from_edge_indptr  --|
                indptr[i]:indptr[i+1] gives the finite values for cluster i
    """
    with h5py.File(output_path, 'w') as f:
        for abs_frame, stats_list in aggregated.items():

            grp = f.create_group(str(abs_frame))

            # ── Flat per-cluster arrays (no nans by construction) ─────────────
            for key, attr in [('ns', 'ns'), ('centroids', 'centroids'), ('areas', 'areas'), ('medPols', 'median_pols'), ('varPols', 'var_pols'), ('medDs', 'median_densities'), ('varDs', 'var_densities'),
                              ('meanThetas', 'mean_thetas'), ('varThetas', 'var_thetas')]:

                grp.create_dataset(key, data=np.concatenate([getattr(s, attr) for s in stats_list]).astype(np.float32), compression='gzip')

            # ── Ragged layer arrays stored as CSR ─────────────────────────────
            def write_csr(name: str, attr: str):
                data_buf, indptr = [], [0]
                for s in stats_list:
                    arr = getattr(s, attr)          # (n_clusters, max_layers), nan-padded
                    for row in arr:
                        finite = row[~np.isnan(row)].astype(np.float32)
                        data_buf.append(finite)
                        indptr.append(indptr[-1] + len(finite))
                grp.create_dataset(f'{name}_data',   data=np.concatenate(data_buf), compression='gzip')
                grp.create_dataset(f'{name}_indptr', data=np.array(indptr, dtype=np.int32), compression='gzip')

            write_csr('p_by_layer',   'p_by_layer')
            write_csr('d_by_layer',   'd_by_layer')
            write_csr('p_from_edge',  'p_from_edge')
            write_csr('d_from_edge',  'd_from_edge')

            grp.attrs['n_clusters'] = sum(len(s.ns) for s in stats_list)

    print(f'Saved to {output_path}')

def load_cluster_stats_h5(input_path: str) -> dict[str, dict]:
    """
    Load HDF5 back into the same dict structure as export_for_plotly,
    but with np.arrays instead of lists (and no Nones — nans where needed).
    """
    out = {}

    with h5py.File(input_path, 'r') as f:
        for group_name in f:
            grp = f[group_name]
            rel_frame = group_name  # keep as string key e.g. 'rel_+0'

            entry = {}

            # Flat arrays
            for key in ['ns', 'centroids', 'areas', 'medPols', 'varPols', 'medDs', 'varDs', 'meanThetas', 'varThetas']:
                entry[key] = grp[key][:]

            # Ragged CSR -> list of 1D arrays (no padding needed, Plotly handles variable length)
            def read_csr(name: str) -> list[np.ndarray]:
                data   = grp[f'{name}_data'][:]
                indptr = grp[f'{name}_indptr'][:]
                # return [data[indptr[i]:indptr[i+1]] for i in range(len(indptr) - 1)]
                return {'data': data, 'indptr': indptr}

            entry['p_by_layer']  = read_csr('p_by_layer')
            entry['d_by_layer']  = read_csr('d_by_layer')
            entry['p_from_edge'] = read_csr('p_from_edge')
            entry['d_from_edge'] = read_csr('d_from_edge')

            out[rel_frame] = entry

    return out
    