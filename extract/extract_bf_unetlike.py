#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 12:46:56 2025

@author: lisaknoche
"""

#!/usr/bin/env python3

from pathlib import Path
from image import DownloadedImageDir
from aliby.tile.tiler import Tiler
import dask.array as da
from skimage.io import imsave
import numpy as np
import torch
from torchvision.transforms.functional import center_crop

# === CONFIG ===
base_dir = Path("/home/lknoche/maby")
image_folder = base_dir / "images_structured" / "Vph1_GFP_005"
h5_file = base_dir / "2179_2024_05_08_MAYBE_training_00" / "Vph1_GFP_005.h5"
output_dir = base_dir / "tiles" / "brightfield_selected_unetlike_Vph1_GFP_005"
output_dir.mkdir(parents=True, exist_ok=True)

# Timepoints and tile
selected_timepoints = [3, 4, 13, 17, 27]
tile_index = 0
input_channel = 0  # Brightfield channel is usually channel 0
crop_size = 96

# === LOAD DATA ===
image = DownloadedImageDir(image_folder)
data = image.get_data_lazy()
tiler = Tiler.from_h5(image, h5_file)

# Normalization range
print("🔍 Calculating BF normalization range...")
input_range = da.percentile(data[:, input_channel].flatten(), [0.1, 99.9]).compute()
print(f"✅ Brightfield normalization range: {input_range}")

# === TILE EXTRACTION ===
for tp in selected_timepoints:
    tile = tiler.get_tile_data(tile_index, tp, c=input_channel)

    # Normalize to [0, 1]
    norm_tile = (tile - input_range[0]) / (input_range[1] - input_range[0])
    norm_tile = np.clip(norm_tile, 0, 1)

    # Center crop
    tensor_tile = torch.from_numpy(norm_tile)
    cropped = center_crop(tensor_tile, (crop_size, crop_size)).numpy()

    save_path = output_dir / f"Vph1_GFP_005_tp{tp:03}_BF_Z_000.tiff"
    imsave(save_path, (cropped * 255).astype(np.uint8))
    print(f"✅ Saved: {save_path.name}")

print("🏁 Done with brightfield tiling.")
