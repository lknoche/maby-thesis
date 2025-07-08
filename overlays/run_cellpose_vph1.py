#!/usr/bin/env python3

from cellpose import models
from skimage.io import imread, imsave
from pathlib import Path
import numpy as np

# === Config ===
input_folder = Path("/home/lknoche/maby/tiles/fluorescence_selected_unetlike_Vph1_GFP_005")
output_folder = Path("/home/lknoche/maby/figures/cellpose_selected_unetlike")
output_folder.mkdir(parents=True, exist_ok=True)

# Load Cellpose 'cyto' model
model = models.Cellpose(model_type="cyto", gpu=True)

# Get selected files
tiff_files = sorted(input_folder.glob("*.tiff"))
print(f"🔍 Found {len(tiff_files)} tiles.")

# Run Cellpose
for i, tiff_path in enumerate(tiff_files):
    print(f"\n🔍 Processing {tiff_path.name}")
    img = imread(tiff_path)

    # Cellpose expects (H, W), (H, W, 1), or (H, W, 3)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]  # Convert (1, H, W) → (H, W)

    # Ensure type float32 in [0, 1]
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0

    masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0])

    mask_path = output_folder / f"mask_{tiff_path.stem}.tiff"
    imsave(mask_path, masks.astype(np.uint16))
    print(f"✅ Saved to: {mask_path}")

print("🏁 All done.")
