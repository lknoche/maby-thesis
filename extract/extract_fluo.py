#!/usr/bin/env python3

from pathlib import Path
from image import DownloadedImageDir
from aliby.tile.tiler import Tiler
import dask.array as da
from skimage.io import imsave
import numpy as np
import torch
from torchvision.transforms.functional import center_crop
import re

# === CONFIG ===
base_dir = Path("/home/lknoche/maby")
image_folder = base_dir / "images_structured" / "Vph1_GFP_005"
h5_file = base_dir / "2179_2024_05_08_MAYBE_training_00" / "Vph1_GFP_005.h5"
output_dir = base_dir / "tiles" / "fluorescence_selected_unetlike_Vph1_GFP_005"
output_dir.mkdir(parents=True, exist_ok=True)

# Actual timepoints you want to extract
desired_tp_numbers = [3, 27, 13, 4, 17]
tile_index = 0
input_channel = 1
crop_size = 96

# === LOAD DATA ===
image = DownloadedImageDir(image_folder)
data = image.get_data_lazy()
tiler = Tiler.from_h5(image, h5_file)

# === Extract all available timepoint numbers from filenames ===
print("🔢 Parsing available timepoints from filenames...")
tiff_files = sorted(image_folder.glob("*_GFP_Z_000.tiff"))
available_tp_numbers = []

pattern = re.compile(r".*_(\d{6})_GFP_Z_000\.tiff")
for f in tiff_files:
    match = pattern.match(f.name)
    if match:
        tp_str = match.group(1)
        tp_num = int(tp_str)
        available_tp_numbers.append(tp_num)

# Ensure unique and sorted
available_tp_numbers = sorted(set(available_tp_numbers))
print(f"🧬 Found {len(available_tp_numbers)} available timepoints")

# Map desired timepoint numbers to indices
tp_indices = []
for tp in desired_tp_numbers:
    try:
        idx = available_tp_numbers.index(tp)
        tp_indices.append(idx)
    except ValueError:
        print(f"⚠️ Timepoint {tp} not found in available files!")

# === Normalize ===
print("🔍 Calculating normalization range...")
input_range = da.percentile(data[:, input_channel].flatten(), [0.1, 99.9]).compute()
print(f"✅ Normalization range: {input_range}")

# === Extract and Save ===
for i, tp_idx in enumerate(tp_indices):
    real_tp = desired_tp_numbers[i]
    tile = tiler.get_tile_data(tile_index, tp_idx, c=input_channel)

    norm_tile = (tile - input_range[0]) / (input_range[1] - input_range[0])
    norm_tile = np.clip(norm_tile, 0, 1)

    tensor_tile = torch.from_numpy(norm_tile)
    cropped = center_crop(tensor_tile, (crop_size, crop_size)).numpy()

    save_path = output_dir / f"Vph1_GFP_005_tp{real_tp:03}.tiff"
    imsave(save_path, (cropped * 255).astype(np.uint8))
    print(f"✅ Saved: {save_path}")

print("🏁 Done.")
