#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 13:07:12 2025

@author: lisaknoche
"""

#!/usr/bin/env python3

from skimage.io import imread, imsave
from skimage.filters import gaussian, threshold_otsu
from pathlib import Path
import numpy as np

# === CONFIG ===
input_folder = Path("/home/lknoche/maby/tiles/fluorescence_selected_unetlike_Vph1_GFP_005")
output_folder = Path("/home/lknoche/maby/figures/otsu_selected_unetlike")
output_folder.mkdir(parents=True, exist_ok=True)

# === MAIN PROCESSING ===
tiff_files = sorted(input_folder.glob("*.tiff"))
print(f"🔍 Found {len(tiff_files)} tiles.")

for i, tiff_path in enumerate(tiff_files):
    print(f"\n🔍 Processing {tiff_path.name}")
    img = imread(tiff_path)

    # Flatten if singleton first axis (e.g., (1, H, W) → (H, W))
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]

    # Convert to float32 in [0, 1] if needed
    if img.dtype != np.float32:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

    # Mask only valid pixels
    valid_mask = img != 0
    if not np.any(valid_mask):
        print("⚠️ Skipping empty image.")
        binary_mask = np.zeros_like(img, dtype=np.uint16)
    else:
        # Normalize valid pixels
        valid_pixels = img[valid_mask]
        min_val, max_val = valid_pixels.min(), valid_pixels.max()
        if max_val == min_val:
            print("⚠️ Uniform image. Using valid mask directly.")
            binary_mask = valid_mask.astype(np.uint16)
        else:
            normalized = np.zeros_like(img)
            normalized[valid_mask] = (valid_pixels - min_val) / (max_val - min_val)
            # Apply Gaussian blur
            blurred = gaussian(normalized, sigma=0.5)
            # Threshold only valid regions
            threshold = threshold_otsu(blurred[valid_mask])
            binary_mask = (blurred > threshold).astype(np.uint16)

    # Save result
    mask_path = output_folder / f"mask_{tiff_path.stem}.tiff"
    imsave(mask_path, binary_mask)
    print(f"✅ Saved to: {mask_path}")

print("🏁 All done.")
