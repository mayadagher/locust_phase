'''_____________________________________________________IMPORTS____________________________________________________________'''

from matplotlib.path import Path
from ultralytics import YOLO
import roboflow
import os
import yaml
import supervision as sv
import cv2
from pathlib import Path
from tqdm import tqdm

import torchvision
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed

'''_____________________________________________________FUNCTIONS____________________________________________________________'''

def load_labeled_data(workspace:str, project_name:str, data_version:int, yolo_short_name:str):
    roboflow.login()

    rf = roboflow.Roboflow()
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(data_version).download(yolo_short_name)

    with open(f"{dataset.location}/data.yaml", 'r') as f:
        dataset_yaml = yaml.safe_load(f)
    dataset_yaml["train"] = f"{dataset.location}/train/images"
    dataset_yaml["val"]   = f"{dataset.location}/valid/images"
    dataset_yaml["test"]  = f"{dataset.location}test/images"

    with open(f"{dataset.location}/data.yaml", 'w') as f: # Write into results directory
        yaml.dump(dataset_yaml, f)

    return dataset

def train_model(workspace:str, project_name:str, data_version:int, yolo_short_name:str, yolo_version:str, epochs:int=100, img_sz:int=640):

    # Load labeled data from Roboflow
    dataset = load_labeled_data(workspace, project_name, data_version, yolo_short_name)
    
    # Train new model
    model = YOLO(yolo_version)
    model.train(data=f"{dataset.location}/data.yaml", epochs=epochs, imgsz=img_sz) # Results will be saved in runs/pose/train

def visualize_single_tile_results(weights_path:str, img_idx:int, img_dir:str):
    model = YOLO(weights_path) # Choose best weights from training

    # Load an image from the dataset
    test_image = os.listdir(f"{img_dir}test/images")[img_idx]
    file_name = os.path.join(f"{img_dir}test/images", test_image)
    results = model(file_name)
    print(results[0].keypoints) # Check what keys are available in the results object

    # Show keypoints overlayed on image
    key_points = sv.KeyPoints.from_ultralytics(results[0])

    # Create annotators for vertices and edges
    vertex_annotator = sv.VertexAnnotator(radius=3, color=sv.Color.WHITE)
    edge_annotator = sv.EdgeAnnotator(thickness=2, edges=[(0, 1)])

    annotated_frame = cv2.imread(file_name)
    annotated_frame = edge_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points)
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points)

    annotated_frame = sv.resize_image(
        annotated_frame,
        resolution_wh=(900, 900),
        keep_aspect_ratio=True
    )
    img = sv.cv2_to_pillow(annotated_frame)
    img.save(f'annotated_{test_image}')

def tensor_to_cv2(img: torch.Tensor) -> np.ndarray:
    """
    img: (C, H, W) torch.Tensor
    returns: (H, W, 3) uint8 BGR image
    """
    if img.ndim == 4:
        img = img[0]  # remove batch dim

    # (C, H, W) → (H, W, C)
    img = img.permute(1, 2, 0)

    # If grayscale, expand to 3 channels
    if img.shape[2] == 1:
        img = img.repeat(1, 1, 3)

    img = img.detach().cpu().numpy()

    # Normalize if needed
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # RGB → BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def verify(model:YOLO, tile:torch.Tensor, tile_num:int):

    tile = tile.reshape(1, 3, tile.shape[1], tile.shape[2]) # Add batch dimension
    results = model(tile)
    print(results[0].keypoints) # Check what keys are available in the results object

    # Show keypoints overlayed on image
    key_points = sv.KeyPoints.from_ultralytics(results[0])

    # Create annotators for vertices and edges
    vertex_annotator = sv.VertexAnnotator(radius=3, color=sv.Color.WHITE)
    edge_annotator = sv.EdgeAnnotator(thickness=2, edges=[(0, 1)])

    annotated_frame = tensor_to_cv2(tile)
    annotated_frame = edge_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points)
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points)

    annotated_frame = sv.resize_image(
        annotated_frame,
        resolution_wh=(900, 900),
        keep_aspect_ratio=True
    )
    img = sv.cv2_to_pillow(annotated_frame)
    img.save(f'annotated_{tile_num}.jpg')

def slice_img(model:YOLO, img_path:str, h5_out:h5py.File, frame_num:int, device:str, tile_size:int = 640, img_size = 7000, overlap:float = 0.3, iou:float = 0.5):
    '''
    Efficiently slices frame into overlapping tiles and appends them to an h5 file. Overlap is a fraction of the tile size. Image is assumed to be square.
    '''
    assert tile_size <= img_size, 'Tile size must be less than or equal to image size.'

    img_full = torchvision.io.decode_image(img_path, mode = torchvision.io.ImageReadMode.GRAY).float() / 255.0 # Shape: (1, img_size, img_size)
    img_full = img_full.repeat(3, 1, 1)  # (3, H, W)

    step_size = round(tile_size*(1 - overlap))
    num_tiles = (img_size - tile_size) // step_size


    # Start indices in x and y direction
    starts = np.linspace(0, img_size - tile_size, num_tiles).astype(int)
    ends = starts + tile_size

    # Create tensor of tiles
    img_ten = torch.zeros(len(starts)**2, 3, tile_size, tile_size)

    tile_idx = 0
    for x_idx in range(len(starts)):
        for y_idx in range(len(starts)):
            x_start, x_end = starts[x_idx], ends[x_idx]
            y_start, y_end = starts[y_idx], ends[y_idx]
            tile = img_full[:, x_start:x_end, y_start:y_end]
            img_ten[tile_idx, :, :, :] = tile

            # verify(model, tile, tile_idx)

            tile_idx += 1

    # Batch inference
    results = model.predict(img_ten, save=False, imgsz=tile_size, save_txt=False, verbose=False, iou = iou, device = device)
    # results = model(img_ten)

    # Add results to h5 file
    for ix, x in enumerate(starts):
        for iy, y in enumerate(starts):
            tile_idx = ix*len(starts) + iy

            # h5_out[f'f{frame_num}/tile_{tile_idx}_x{x}_y{y}/boxes/conf'] = results[tile_idx].boxes.conf.cpu() # (N, )
            # h5_out[f'f{frame_num}/tile_{tile_idx}_x{x}_y{y}/boxes/xyxy'] = results[tile_idx].boxes.xyxy.cpu() # (N, 4) = (N, x1, y1, x2, y2)
            h5_out[f'f{frame_num}/tile_{tile_idx}_x{x}_y{y}/conf'] = results[tile_idx].keypoints.conf.cpu() # (N, 2) = (N, head/tail)
            h5_out[f'f{frame_num}/tile_{tile_idx}_x{x}_y{y}/xy'] = results[tile_idx].keypoints.xy.cpu() # (N, 2, 2) = (N, head/tail, x/y)

def slice_folder_to_h5(path_to_model:str, frames_dir: str, h5_in: str, start_idx: int, stop_idx: int, chunk_size: int = 500, tile_size: int = 640, img_size: int = 7000, overlap: float = 0.3, iou: float = 0.5):
    """
    Slice frames [start_idx : stop_idx) into YOLO tiles and store results in a HDF5 file.

    Frames are processed in alphabetical order. If the file already exists, new results are appended (assuming whole frames are completed).
    """
    # Set the device dynamically
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print('Using mps.')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using cuda.')
    else:
        device = torch.device("cpu")
        print('Using cpu.')

    # Load model
    model = YOLO(path_to_model)
    frames_path = Path(frames_dir)

    # Collect and sort frame paths
    frames = sorted(frames_path.glob("*.jpg"))
    stop_idx = len(frames) if stop_idx > len(frames) else stop_idx
    assert 0 <= start_idx < stop_idx <= len(frames), "Invalid start/stop indices"

    frames = frames[start_idx:stop_idx]
    num_frames = len(frames)

    num_chunks = np.ceil(num_frames / chunk_size).astype(int)

    # Iterate over chunks of frames to manage memory usage and allow for progress tracking
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, num_frames)

        print(f"Starting chunk {chunk_idx + 1}/{num_chunks}: frames {start_idx + chunk_start} to {start_idx + chunk_end - 1} → {h5_in}")

        with h5py.File(h5_in, "a") as h5_out:
            for local_idx, frame_path in enumerate(tqdm(frames[chunk_start:chunk_end])):
                global_frame_idx = start_idx + chunk_start + local_idx

                if f"f{global_frame_idx}" in h5_out:
                    print(f"Frame {global_frame_idx} already processed, skipping.")
                    continue

                slice_img(model=model, img_path=str(frame_path), h5_out=h5_out, frame_num=global_frame_idx, tile_size=tile_size, img_size=img_size, overlap=overlap, iou = iou, device = device)

def visualize_frame_results(h5_file:str, img_path:str, frame_num:int):

    '''
    Visualizes the keypoints detected in a single frame by reading from the h5 file and plotting on top of the original image.
    '''
    # Load original image (BGR → RGB, float in [0, 1])
    img_full = cv2.imread(img_path)
    scat = []
    batch = []
    if img_full is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    batch_num = 0
    with h5py.File(h5_file, "r") as h5_in:
        for frame_key in h5_in.keys():
            if frame_key != f"f{frame_num}":
                continue
            for key in h5_in[frame_key].keys():
                row_offset = int(key.split('x')[1].split('_')[0])
                col_offset = int(key.split('y')[1].split('_')[0])

                confs = h5_in[frame_key][f"{key}/conf"][:]
                kps = h5_in[frame_key][f"{key}/xy"][:]

                for kp_idx, _ in enumerate(confs):

                    curr_kps = kps[kp_idx] # (head/tail, x/y)

                    # Transform into global image coordinates
                    curr_kps[:,0] += col_offset # YOLO x is column index
                    curr_kps[:,1] += row_offset # YOLO y is row index
                    curr_kps = curr_kps.astype(int)

                    scat.append(curr_kps)
                    batch.append(batch_num) # Identifies which tile keypoints are from, for colouring
                batch_num += 1
                
    # Colours to make tiles easily distinguishable
    scat = np.array(scat)
    batch = np.array(batch)
    colours = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink']
    
    # print(scat)
    plt.figure(figsize=(20, 20))
    plt.imshow(img_full)
    plt.scatter(scat[:, 0, 0], scat[:, 0, 1], c=[colours[b % len(colours)] for b in batch], s=0.1)
    plt.scatter(scat[:, 1, 0], scat[:, 1, 1], c=[colours[b % len(colours)] for b in batch], s=0.1)
    plt.savefig(f"visualized_frame_{frame_num}.png")

def unprocessed_stats(h5_in:str, subsample: int, n_bins:int, output_dir:str):

    # Assure subsample value is valid (number of frames to use for statistics)
    assert subsample > 0, f"Invalid number of frames for subsample given: {subsample}"
    
    # Define histograms to update over frames
    hist_bb_confs = np.zeros(n_bins)
    hist_head_confs = np.zeros(n_bins)
    hist_tail_confs = np.zeros(n_bins)
    hist_lens = np.zeros(n_bins)

    # Define ranges for histograms
    conf_range = (0, 1)
    len_range = (0, 100)
    
    # Collect all relevant data from all frames for diagnostic statistics
    with h5py.File(h5_in, 'r') as f_in:
        
        # Iterate over frames
        frames = list(f_in.keys())
        frames = [int(f[1:]) for f in frames] # Not assuming frames are contiguous in case there is some error in the inference step
        
        max_frames = min(len(frames), subsample)
        frames = np.random.choice(frames, max_frames, replace = False)

        for f in tqdm(frames):

            # Iterate over tiles
            for key in f_in[f'f{f}'].keys():

                # Update histograms
                hist_bb_confs += np.histogram(f_in[f'f{f}'][f'{key}/boxes/conf'], bins=n_bins, range=conf_range)[0]
                hist_head_confs += np.histogram(f_in[f'f{f}'][f'{key}/conf'][:,0], bins=n_bins, range=conf_range)[0]
                hist_tail_confs += np.histogram(f_in[f'f{f}'][f'{key}/conf'][:,1], bins=n_bins, range=conf_range)[0]
                hist_lens += np.histogram(np.linalg.norm(f_in[f'f{f}'][f'{key}/xy'][:,0,:] - f_in[f'f{f}'][f'{key}/xy'][:,1,:], axis = 1), bins=n_bins, range=len_range)[0]

    # Plot distribution of box confidences to see if it's a good candidate for cleaning, distribution of locust lengths to find appropriate length cutoff(s)
    _, ax  = plt.subplots(3, figsize = (6, 10))

    # Compute centers of hist bins
    conf_centers = np.linspace(*conf_range, n_bins)
    conf_width = np.diff(conf_centers)[0]
    len_centers = np.linspace(*len_range, n_bins)
    len_width = np.diff(len_centers)[0]

    # Plot bars for histograms
    ax[0].bar(conf_centers, hist_bb_confs, conf_width)
    ax[0].set_ylabel('Counts', fontsize = 17)
    ax[0].set_xlabel('Bounding box confidences', fontsize = 17)

    ax[1].bar(conf_centers, hist_head_confs, conf_width, alpha = 0.5, label = 'Head')
    ax[1].bar(conf_centers, hist_tail_confs, conf_width, alpha = 0.5, label = 'Tail')
    ax[1].set_ylabel('Counts', fontsize = 17)
    ax[1].set_xlabel('Keypoint confidences', fontsize = 17)
    ax[1].legend()

    ax[2].bar(len_centers, hist_lens, len_width)
    ax[2].set_ylabel('Counts', fontsize = 17)
    ax[2].set_xlabel('Lengths', fontsize = 17)
    cumul = np.cumsum(hist_lens)/np.sum(hist_lens)
    idx_low = np.where(cumul > 0.1)[0][0]
    idx_high = np.where(cumul > 0.99)[0][0]
    ax[2].axvline(len_centers[idx_low], color = 'k', linestyle = '--')
    ax[2].axvline(len_centers[idx_high], color = 'k', linestyle = '--', label = '10th/99th percentile')
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir + 'plots/kp_preprocess/unprocessed_hists.png')

def remove_duplicates_graph(all_kps:np.ndarray, kp_confs:np.ndarray, kp_radius:float=20, centroid_radius:float=10):

    n = len(all_kps)

    # --- Precompute ---
    heads = all_kps[:, 0, :]
    tails = all_kps[:, 1, :]
    centroids = (heads + tails) / 2
    total_conf = np.mean(kp_confs, axis=1)

    # 1. Build KDTree for keypoints (flattened)
    all_kps = all_kps.reshape(-1, 2) # (n_kps, x/y)
    kp_tree = cKDTree(all_kps)

    pairs = np.array(list(kp_tree.query_pairs(r=kp_radius)))
    if len(pairs) == 0:
        return np.ones(n, dtype=bool)

    # Map kp → detection
    kp_to_det = pairs // 2  # (num_pairs, 2) -> kps alternate head/tail so index 0, 1 correspond to same detection, etc.

    det_i = kp_to_det[:, 0]
    det_j = kp_to_det[:, 1]

    # Remove self-pairs
    valid = det_i != det_j
    det_i = det_i[valid]
    det_j = det_j[valid]

    # 2. Count how many kp matches per detection pair
    edges = np.stack([det_i, det_j], axis=1)

    # Normalize ordering (i < j)
    edges = np.sort(edges, axis=1)

    # Count occurrences
    edge_ids, counts = np.unique(edges, axis=0, return_counts=True)

    # Keep only strong matches (both keypoints matched)
    strong_edges = edge_ids[counts >= 2]

    # 3. Add centroid-based proximity edges
    cent_tree = cKDTree(centroids)
    cent_pairs = np.array(list(cent_tree.query_pairs(r=centroid_radius)))

    if len(cent_pairs) > 0:
        cent_pairs = np.sort(cent_pairs, axis=1)
        strong_edges = np.vstack([strong_edges, cent_pairs])

    # 4. Build graph (sparse adjacency)
    if len(strong_edges) == 0:
        return np.ones(n, dtype=bool)

    row = strong_edges[:, 0]
    col = strong_edges[:, 1]

    adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))

    # Make symmetric
    adj = adj + adj.T

    # 5. Connected components
    n_components, labels = connected_components(adj, directed=False)

    # 6. Keep best detection per component
    keep_mask = np.zeros(n, dtype=bool)

    for comp in range(n_components):
        idx = np.where(labels == comp)[0]

        if len(idx) == 1:
            keep_mask[idx[0]] = True
            continue

        best = idx[np.argmax(total_conf[idx])]
        keep_mask[best] = True

    return keep_mask

def preprocess_frame(frame_num:int, f_in:h5py.File, f_out:h5py.File, tile_offsets:dict):

    kp_confs = []
    all_kps = []
    frame_key = f'f{frame_num}'

    # Skip frames that are already pre-processed
    if frame_key in f_out.keys(): 
        return

    # Collect confidences and keypoints from each tile
    for key in f_in[frame_key].keys():
        kp_confs.append(f_in[frame_key][f"{key}/conf"][:])
        keypoints = f_in[frame_key][f"{key}/xy"][:]
        keypoints[:, :, 0] += tile_offsets[key][0]
        keypoints[:, :, 1] += tile_offsets[key][1]
        all_kps.append(keypoints) # Appending array of shape (num_detections, head/tail, x/y)

    all_kps = np.concatenate(all_kps, axis=0) # (num_detections, head/tail, x/y)
    kp_confs = np.concatenate(kp_confs, axis=0) # (num_detections, head/tail)

    # Filter using locust lengths
    lengths = np.linalg.norm((all_kps[:, 0, :] - all_kps[:, 1, :]), axis = 1) # (N,)
    valid_lengths = lengths > 25 # Average length is 50
    all_kps = all_kps[valid_lengths,:,:]
    kp_confs = kp_confs[valid_lengths,:]

    # Filter using kp and centroid proximities
    keep_mask = remove_duplicates_graph(all_kps, kp_confs)
    all_kps = all_kps[keep_mask]

    # Save data to preprocessed .hdf5 file
    centroids = np.mean(all_kps, axis = 1)
    f_out[f'{frame_key}/head'] = all_kps[:, 0, :] # Save head keypoints
    f_out[f'{frame_key}/tail'] = all_kps[:, 1, :] # Save tail keypoints
    f_out[f'{frame_key}/conf'] = kp_confs # Save confidences of head and tail keypoints
    f_out[f'{frame_key}/centroid'] = centroids # Save centroids computed from keypoints

def preprocess_kps_fast(h5_in:str):
    '''
    Use confidences to fix likely duplicates of the same locust. Additionally, exclude single locusts whose two keypoints are too close together to be biologically plausible. Saves cleaned keypoints in a new h5 file.
    '''

    # Name output file
    try:
        h5_out_path = h5_in.split('un')[0] + h5_in.split('un')[1]
    except:
        raise ValueError("Input h5 file name must contain 'unprocessed'.")
    
    # Open input and output file
    with h5py.File(h5_in, "r") as f_in, h5py.File(h5_out_path, 'a') as f_out:

        # Compute tile offsets once before parallelizing
        tile_offsets = {}
        for key in f_in[list(f_in.keys())[0]].keys():
            y = int(key.split('y')[1].split('_')[0])
            x = int(key.split('x')[1].split('_')[0])
            tile_offsets[key] = (y, x) # YOLO uses y for row, x for column of image

        # Preprocess each frame separately
        for frame_num in tqdm(range(len(f_in.keys()))):
            preprocess_frame(frame_num, f_in, f_out, tile_offsets)
