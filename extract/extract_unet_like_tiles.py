#!/usr/bin/env python3

from pathlib import Path
from dataset import LocalPositionDataset, transform
import numpy as np
from skimage.io import imsave

# --- Config ---
image_folder = Path("/home/lknoche/maby/images_structured/Htb2_sfGFP_005")
h5_file = Path("/home/lknoche/maby/2179_2024_05_08_MAYBE_training_00/Htb2_sfGFP_005.h5")
output_dir = Path("/home/lknoche/maby/tiles/unet_like_tiles")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load dataset with transform (center crop 96x96 like U-Net) ---
dataset = LocalPositionDataset(
    image_folder=image_folder,
    h5_file=h5_file,
    transform=transform,
    input_channel=0,
    output_channel=1,
    start_tp=0,
    end_tp=5  # First 5 timepoints
)

# --- Save each tile image (input x only) ---
for idx in range(len(dataset)):
    x, _ = dataset[idx]
    x_np = (x.numpy() * 255).astype(np.uint8)  # Convert to 0–255 grayscale
    out_path = output_dir / f"tile_{idx:03d}.tiff"
    imsave(out_path, x_np)
    print(f"✅ Saved: {out_path}")
