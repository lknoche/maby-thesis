#!/usr/bin/env python3

from pathlib import Path
from dataset import LocalPositionDataset, transform
import numpy as np
from skimage.io import imsave

# --- Config ---
image_folder = Path("/home/lknoche/maby/images_structured/Vph1_GFP_005")
h5_file = Path("/home/lknoche/maby/2179_2024_05_08_MAYBE_training_00/Vph1_GFP_005.h5")
output_dir = Path("/home/lknoche/maby/tiles/unet_like_tiles_vacuole")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load dataset (same as test.py) ---
dataset = LocalPositionDataset(
    image_folder=image_folder,
    h5_file=h5_file,
    transform=transform,
    input_channel=0,   # GFP channel
    output_channel=1,  # GT mask
    start_tp=0,
    end_tp=10,         # Only use first 10 timepoints
)

# --- Save tile images (only input image, single channel) ---
for idx in range(len(dataset)):
    x, _ = dataset[idx]   # x shape: [C, H, W]
    img = x[0].numpy()    # Use input channel only
    img = (img * 255).astype(np.uint8)
    out_path = output_dir / f"tile_{idx:03d}.tiff"
    imsave(out_path, img)
    print(f"✅ Saved: {out_path}")
