#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 13:12:56 2025

@author: lisaknoche
"""

#!/usr/bin/env python3

from pathlib import Path
from dataset import LocalPositionDataset, transform
from skimage.io import imsave
import numpy as np

# === CONFIG ===
image_folder = Path("/home/lknoche/maby/images_structured/Vph1_GFP_005")
h5_file = Path("/home/lknoche/maby/2179_2024_05_08_MAYBE_training_00/Vph1_GFP_005.h5")
output_dir = Path("/home/lknoche/maby/tiles/ground_truth_masks_vacuole")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load Dataset (ONLY the ground truth channel) ===
dataset = LocalPositionDataset(
    image_folder=image_folder,
    h5_file=h5_file,
    transform=transform,
    input_channel=0,       # Still needed, even if not saved
    output_channel=1,      # 💡 This is the GT mask
    start_tp=0,
    end_tp=5               # Use whatever range you want
)

# === Save Ground Truth Masks ===
for idx in range(len(dataset)):
    _, y = dataset[idx]
    y_np = (y.numpy() * 255).astype(np.uint8)  # Assumes binary masks
    out_path = output_dir / f"mask_{idx:03d}.tiff"
    imsave(out_path, y_np)
    print(f"✅ Saved: {out_path}")
