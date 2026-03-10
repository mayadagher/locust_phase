'''_____________________________________________________IMPORTS____________________________________________________________'''

import cv2
import numpy as np
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

from helper_fns import *

'''_____________________________________________________VIDEO ANALYSIS FUNCTIONS____________________________________________________________'''

def load_image_sequence(image_dir:str) -> list:
    image_dir = Path(image_dir)
    frames = sorted(image_dir.glob("*.jpg"))
    if len(frames) == 0:
        raise RuntimeError(f"No JPEGs found in {image_dir}")
    return frames

def compute_global_background(image_dir, max_frames=500):
    """
    Compute a global background image using a temporal median.
    Frames are uniformly subsampled for efficiency.
    """
    frames = sorted(Path(image_dir).glob("*.jpg"))
    if len(frames) == 0:
        raise RuntimeError("No frames found")

    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)

    stack = []
    for i in idx:
        img = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stack.append(img)

    if len(stack) == 0:
        raise RuntimeError("Failed to load background frames")

    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)

def extract_random_tiles(image_dir:str, output_dir:str, n_tiles:int, tile_size:int=320):
    """ Extract random spatial tiles from random images in a directory."""

    frames = load_image_sequence(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read one image to get dimensions
    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]

    if tile_size > w or tile_size > h:
        raise ValueError("Tile size larger than image dimensions.")

    n_frames = len(frames)

    for i in range(n_tiles):
        frame_idx = np.random.randint(0, n_frames)
        img = cv2.imread(str(frames[frame_idx]))

        if img is None:
            continue

        x0 = np.random.randint(0, w - tile_size)
        y0 = np.random.randint(0, h - tile_size)

        tile = img[y0:y0 + tile_size, x0:x0 + tile_size]

        out_path = output_dir / f"tile_{i:06d}_f{frame_idx}_x{x0}_y{y0}.png"
        cv2.imwrite(str(out_path), tile)

        if i % 100 == 0:
            print(f"Extracted {i + 1}/{n_tiles} tiles.")

def compute_single_locust_area(image_dir: str, ds: xr.Dataset, pos_name: str, num_individuals: int, density_radius: float, val_threshold: int, exp_name: str, batch_num: int) -> float:
    plot_first = False
    frames = load_image_sequence(image_dir)

    # Compute local density of each locust given radius in ds
    ds = get_metric_density(ds, pos_name, density_radius)

    # Extract best ids and frames based on density
    densities = ds[f"density_r_{density_radius}"].values
    sorted_idx = np.unravel_index(np.argsort(densities, axis=None), densities.shape)

    best_ids = ds.id.values[sorted_idx[0][:num_individuals]].astype(int)
    best_frames = ds.frame.values[sorted_idx[1][:num_individuals]].astype(int)

    # Extract positions of locusts with lowest densities
    relative_frames = best_frames - int(ds.frame.min())
    best_positions = np.vstack([
        ds[f"x_{pos_name}"].values[best_ids, relative_frames],
        ds[f"y_{pos_name}"].values[best_ids, relative_frames]])
    
    # Compute subsampled background median
    bg = compute_global_background(image_dir, max_frames=100)

    single_locust_areas = []

    for i, f in enumerate(best_frames):
        img = cv2.imread(str(frames[f]))
        if img is None:
            continue

        x, y = best_positions[:, i]
        buffer = 40

        locust_region = img[
            int(y - buffer):int(y + buffer),
            int(x - buffer):int(x + buffer)]
        
        bg_region = bg[
            int(y - buffer):int(y + buffer),
            int(x - buffer):int(x + buffer)]

        if locust_region.size == 0:
            continue

        # Convert to grayscale
        locust_gray = cv2.cvtColor(locust_region, cv2.COLOR_BGR2GRAY)

        if bg_region.shape != locust_gray.shape:
            continue

        # Background subtraction (locusts darker than background)
        fg = bg_region.astype(np.int16) - locust_gray.astype(np.int16)
        fg = np.clip(fg, 0, 255).astype(np.uint8)

        # Binarize background subtracted image
        _, binary = cv2.threshold(fg, val_threshold, 255, cv2.THRESH_BINARY)

        # Get rid of small pixel clusters
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        single_locust_areas.append(np.max(areas) if areas else np.nan) # Take largest cluster as locust area

        # Debugging: save binary image
        if np.max(areas) > 2000 or plot_first:
            _, ax = plt.subplots(1,2)
            ax[0].imshow(locust_region)
            ax[1].imshow(binary, cmap='gray')
            plt.savefig(f'debug_binary_{i}.png')
            print(f'Density: {densities[best_ids[i], relative_frames[i]]}, Area: {np.max(areas)}')
            plot_first = False

    if len(single_locust_areas) == 0:
        raise RuntimeError("Unable to find any locusts.")
    
    # Debugging: see distribution of single locust areas
    plt.figure()
    plt.hist(single_locust_areas, bins=21)
    plt.xlabel('Single locust area (pixels)', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.title(f'Value threshold: {val_threshold}', fontsize = 20)
    plt.savefig(f'./plots/{exp_name}/batch_{batch_num}/single_locust_area/val_thresh_{val_threshold}.png')

    return np.nanmedian(single_locust_areas)

def estimate_locust_number(image_dir: str, per_locust_area: float, val_threshold: int = 60, area_threshold: int = 10, radius_inclusion: float = 3500, start_frame: int = 0, end_frame: int = -1) -> list:

    frames = load_image_sequence(image_dir)
    frames = frames[start_frame:end_frame] if end_frame > 0 else frames[start_frame:]

    # Compute subsampled background median
    bg = compute_global_background(image_dir, max_frames=100)

    num_locusts = []
    mask = None

    # Iterate through all frames
    for frame_idx, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Convert to grayscale and binarize based on value threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Background subtraction (locusts darker than background)
        fg = bg.astype(np.int16) - gray.astype(np.int16)
        fg = np.clip(fg, 0, 255).astype(np.uint8)

        # Binarize background subtracted image
        _, binary = cv2.threshold(fg, val_threshold, 255, cv2.THRESH_BINARY)

        # Get rid of pixel clusters smaller than area threshold
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        binary_cleaned = np.zeros_like(binary)

        for contour in contours:
            if cv2.contourArea(contour) >= area_threshold:
                cv2.drawContours(binary_cleaned, [contour], -1, 255, thickness=cv2.FILLED)

        # Construct arena mask if not already done
        if mask is None:
            h, w = binary_cleaned.shape
            cy, cx = h // 2, w // 2
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_inclusion ** 2

        # Estimate number of locusts by area within masked region divided by per locust area
        locust_area = cv2.countNonZero(binary_cleaned & (mask.astype(np.uint8) * 255))
        num_locusts.append(locust_area / per_locust_area)

        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}, locust estimate: {round(locust_area / per_locust_area)}")

    return num_locusts