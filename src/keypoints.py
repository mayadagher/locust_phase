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

def preprocess_kps(h5_in:str, frames_dir:str):
    '''
    Use confidences to fix likely duplicates of the same locust. Additionally, exclude single locusts whose two keypoints are too close together to be biologically plausible. Saves cleaned keypoints in a new h5 file.
    '''

    try:
        h5_out_path = h5_in.split('un')[0] + h5_in.split('un')[1]
    except:
        raise ValueError("Input h5 file name must contain 'unprocessed'.")

    with h5py.File(h5_in, "r") as f_in, h5py.File(h5_out_path, 'a') as f_out:
        for frame_num in tqdm(range(len(f_in.keys()))):
            confs = []
            all = []
            frame_key = f'f{frame_num}'

            if frame_key in f_out.keys(): # Skip frames that are already pre-processed
                continue

            for key in f_in[frame_key].keys():
                confs.append(f_in[frame_key][f"{key}/conf"][:])
                keypoints = f_in[frame_key][f"{key}/xy"][:]
                keypoints[:, :, 0] += int(key.split('y')[1].split('_')[0]) # YOLO y is row index
                keypoints[:, :, 1] += int(key.split('x')[1].split('_')[0]) # YOLO x is column index
                all.append(keypoints) # (num_detections, head/tail, x/y)
            
            all_locusts = np.concatenate(all, axis=0) # (num_detections, head/tail, x/y)
            centroids = np.mean(all_locusts, axis = 1) # (num_detections, x/y)
            confs = np.concatenate(confs, axis=0) # (num_detections, head/tail)

            # Trouble-shooting: visualize confidences
            # fig, ax = plt.subplots(2)
            # ax[0].hist(confs[:,0].flatten(), bins = 50)
            # ax[0].hist(confs[:,1].flatten(), bins = 50)
            # ax[0].legend(['Heads', 'Tails'])
            # ax[0].set_xlabel('Confidence', fontsize = 14)
            # ax[0].set_ylabel('Counts', fontsize = 14)
            # ax[1].hist(np.diff(confs, axis=1).flatten(), bins = 50)
            # ax[1].set_xlabel('Confidence difference (head - tail)', fontsize = 14)
            # ax[1].set_ylabel('Counts', fontsize = 14)
            # plt.tight_layout()
            # plt.savefig('confidence_histograms.png')

            # Use a confidence threshold to exclude low-confidence detections, which are more likely to be false positives and could interfere with the duplicate removal process.
            conf_thresh = 0.01
            valid_confs = (confs[:,0] > conf_thresh) & (confs[:,1] > conf_thresh) # Only keep detections where both head and tail have confidence above threshold
            all_locusts = all_locusts[valid_confs, :, :]
            centroids = centroids[valid_confs, :]
            confs = confs[valid_confs, :]
            
            # Determine which bounding boxes have keypoints that are too close together to be biologically plausible (edge effect within tiles)
            lengths = np.linalg.norm((all_locusts[:, 0, :] - all_locusts[:, 1, :]), axis = 1)
            length_threshold = np.quantile(lengths, [0.05, 0.999]) # Min and max plausible locust length in pixels
            # print(f"Length threshold for frame {frame_num}: {length_threshold}")

            # Trouble-shooting: visualize lengths
            # plt.hist(lengths, bins = 70)
            # plt.xlabel('Locust lengths (px)', fontsize = 14)
            # plt.ylabel('Counts', fontsize = 14)
            # plt.savefig('length_histogram.png')

            # Remove locusts with invalid lengths
            valid_length = (np.linalg.norm((all_locusts[:, 0, :] - all_locusts[:, 1, :]), axis = 1) > length_threshold[0]) & (np.linalg.norm((all_locusts[:, 0, :] - all_locusts[:, 1, :]), axis = 1) < length_threshold[1])
            all_locusts = all_locusts[valid_length, :, :]
            centroids = centroids[valid_length, :]
            confs = confs[valid_length, :]
            all_kps = all_locusts.reshape(-1, 2) # (num_kps, 2), head and tail are alternating rows

            # Create a keypoint to keypoint KDTree
            tree = cKDTree(all_kps)
            pairs = tree.query_pairs(r=20)
            pairs = np.array(list(pairs)) # (num_pairs, 2), gives kp indices of pairs of kp in close proximity. Note that pairs are given in sorted order, so if (a, b) is in pairs, we know a < b.

            # Check if BOTH keypoints are close to two keypoints in another, same bounding box
            exclude = []
            unique_kp_idcs = np.unique(pairs.flatten())
            for kp_idx in unique_kp_idcs:
                kp_detection_idx = kp_idx // 2
                if kp_detection_idx in exclude: # If this keypoint already marked for exclusion as part of a duplicate pair, skip
                    continue

                kp_even = kp_idx % 2 == 0
                kp_counterpart_idx = kp_idx + 1 if kp_even else kp_idx - 1

                if kp_counterpart_idx in unique_kp_idcs: # If both kps in a bb are close to other kps, they may be a duplicate of another bb
                    kp_pairs = np.concatenate([pairs[(pairs[:,0] == kp_idx),1], pairs[(pairs[:,1] == kp_idx),0]]) # Find all pairs that include this keypoint, whether it's the first or second element in the pair
                    kp_counterpart_pairs = np.concatenate([pairs[(pairs[:,0] == kp_counterpart_idx),1], pairs[(pairs[:,1] == kp_counterpart_idx),0]])

                    for pair in kp_pairs:
                        pair_even = pair % 2 == 0
                        pair_counterpart_idx = pair + 1 if pair_even else pair - 1

                        if pair_counterpart_idx in kp_counterpart_pairs: # If the counterpart of the current keypoint is also close to the counterpart of the pair keypoint, we know for sure that these two bbs are duplicates of each other and we can exclude the one with the lower confidence.
                            pair_detection_idx = pair // 2

                            if np.sum(confs[kp_detection_idx]) >= np.sum(confs[pair_detection_idx]): # Keep current keypoint and its counterpart, exclude pair keypoint and its counterpart
                                exclude.append(pair_detection_idx)
                            else: # Keep pair keypoint and its counterpart, exclude current keypoint and its counterpart
                                exclude.append(kp_detection_idx)
                            break

            # print(f'Original number of detections in the frame: {len(lengths)}')
            # print(f'Number of valid-length detections in the frame: {len(all_locusts)}')
            # print(f'Number of keypoints excluded as duplicates: {len(exclude) // 2}')

            # If there are keypoints from different bbs that are VERY close together, check if either of the counterparts have low confidence (if so exclude)
            tight_pairs = tree.query_pairs(r=3) # If keypoints from different detections are very close together, check if the counterpart has low confidence
            tight_pairs = np.array(list(tight_pairs))
            conf_thresh = 0.2
            unique_kp_idcs = np.unique(tight_pairs.flatten())
            for kp_idx in unique_kp_idcs:
                kp_detection_idx = kp_idx // 2
                if kp_detection_idx in exclude: # If this keypoint already marked for exclusion as part of a duplicate pair, skip
                    continue

                kp_even = kp_idx % 2 == 0

                kp_pairs = np.concatenate([tight_pairs[(tight_pairs[:,0] == kp_idx),1], tight_pairs[(tight_pairs[:,1] == kp_idx),0]])
                kp_counterpart_conf = confs[kp_detection_idx, int(not kp_even)] # Get the confidence of the counterpart keypoint
                for pair in kp_pairs:
                    if pair // 2 in exclude: # If the pair keypoint already marked for exclusion as part of a duplicate pair, skip
                        continue
                    pair_even = pair % 2 == 0
                    pair_counterpart_idx = pair + 1 if pair_even else pair - 1

                    pair_counterpart_conf = confs[pair // 2, int(pair_even)]

                    if kp_counterpart_conf < conf_thresh and pair_counterpart_conf > kp_counterpart_conf:
                        exclude.append(kp_detection_idx)
                        break
                    elif pair_counterpart_conf < conf_thresh and kp_counterpart_conf > pair_counterpart_conf:
                        exclude.append(pair // 2)
                        break
            
            # print(f'Number of locusts excluded due to tight proximity + low keypoint confidence: {len(exclude)}')
            # print(f'Number of remaining detections: {len(all_locusts) - len(exclude) // 2}')

            # Remove keypoints that are likely duplicates within and across tiles
            all_locusts = np.delete(all_locusts, exclude, axis=0) # Remove single keypoint duplicates
            centroids = np.delete(centroids, exclude, axis = 0)
            confs = np.delete(confs, exclude, axis=0)

            # Use centroids to determine if two detections are too close to be true (then take the one with higher overall confidence)
            centroid_thresh = 10
            tree = cKDTree(centroids)
            tight_centroids = tree.query_pairs(r=centroid_thresh)
            total_confs = np.mean(confs, axis=1)
            exclude = []
            for (a, b) in tight_centroids:
                if a in exclude or b in exclude:
                    continue
                
                if total_confs[a] > total_confs[b]:
                    exclude.append(b)
                else:
                    exclude.append(a)
            
            # Remove keypoints that are likely duplicates across tiles
            all_locusts = np.delete(all_locusts, exclude, axis=0) # Remove single keypoint duplicates
            centroids = np.delete(centroids, exclude, axis = 0)
            confs = np.delete(confs, exclude, axis=0)

            # print(f'Number of excluded detections by centroid distance: {len(exclude)}')


            # frames_path = Path(frames_dir)
            # frames = sorted(frames_path.glob("*.jpg"))
            # img_full = cv2.imread(frames[frame_num])

            # plt.figure(figsize=(20, 20))
            # plt.imshow(img_full)
            # plt.scatter(all_locusts[:, 0, 0], all_locusts[:, 0, 1], s=0.1)
            # plt.scatter(all_locusts[:, 1, 0], all_locusts[:, 1, 1], s=0.1)
            # for line in range(all_locusts.shape[0]):
            #     plt.plot(all_locusts[line,:,0], all_locusts[line,:,1], linewidth=0.5, color = 'w')
            # plt.savefig(f"visualized_preprocessed_frame_{frame_num}_new.png")

            # break

            f_out[f'{frame_key}/head'] = all_locusts[:, 0, :] # Save head keypoints
            f_out[f'{frame_key}/tail'] = all_locusts[:, 1, :] # Save tail keypoints
            f_out[f'{frame_key}/conf'] = confs # Save confidences of head and tail keypoints
            f_out[f'{frame_key}/centroid'] = centroids # Save centroids computed from keypoints