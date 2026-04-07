import numpy as np
import xarray as xr
from ultralytics import YOLO
import h5py
import torchvision
import torch
from tqdm import tqdm
from pathlib import Path

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

def slice_folder_to_h5(path_to_model:str, frames_dir: str, h5_dir: str, start_idx: int, stop_idx: int, chunk_size: int = 500, tile_size: int = 640, img_size: int = 7000, overlap: float = 0.3, iou: float = 0.5):
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

        print(f"Starting chunk {chunk_idx + 1}/{num_chunks}: frames {start_idx + chunk_start} to {start_idx + chunk_end - 1} → /keypoints/{video_name}_unprocessed_kps.hdf5")

        with h5py.File(h5_dir, "a") as h5_out:
            for local_idx, frame_path in enumerate(tqdm(frames[chunk_start:chunk_end])):
                global_frame_idx = start_idx + chunk_start + local_idx

                if f"f{global_frame_idx}" in h5_out:
                    print(f"Frame {global_frame_idx} already processed, skipping.")
                    continue

                slice_img(model=model, img_path=str(frame_path), h5_out=h5_out, frame_num=global_frame_idx, tile_size=tile_size, img_size=img_size, overlap=overlap, iou = iou, device = device)


if __name__ == '__main__':
    slice_folder_to_h5(path_to_model = path_to_model, frames_dir = frames_dir, h5_dir = h5_dir, start_idx = start_idx, stop_idx = stop_idx, chunk_size = 200, tile_size = 640, img_size = 7000, overlap = 0.3)
