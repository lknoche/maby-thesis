#!/usr/bin/env python3
"""
Create a combined diagnostic grid with brightfield, GFP, U-Net, Cellpose, and Otsu overlays.
Only final overlays are shown in columns 3–5. This version aligns only 3 shared rows.
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np

# --- CONFIG ---
base_dir = Path("/home/lknoche/maby")
tiles_dir = base_dir / "tiles"
figures_dir = base_dir / "figures"
output_path = figures_dir / "combined_overlay_grid_vacuole_allmethods.png"

# Input subfolders
brightfield_dir = tiles_dir / "brightfield_selected_unetlike_Vph1_GFP_005"
fluorescence_dir = tiles_dir / "fluorescence_selected_unetlike_Vph1_GFP_005"

# Pre-combined overlay panels (each row = 5 overlays)
unet_panel_path = figures_dir / "Vph1_GFP_diagnostics_tp30.png"
cellpose_panel_path = figures_dir / "Vph1_GFP_cellpose_BF_overlay_panel.png"
otsu_panel_path = figures_dir / "Vph1_GFP_otsu_BF_overlay_panel.png"

# Aligned row indices
unet_rows_idx = [0, 1, 3]
cellpose_rows_idx = [2, 0, 4]
otsu_rows_idx = [2, 0, 4]
timepoints = [13, 27, 17]
num_samples = len(timepoints)

# --- Load and crop rows from pre-combined panels ---
def crop_rows(image, row_indices, col_index, total_cols):
    h, w = image.shape[:2]
    row_height = h // 5
    col_width = w // total_cols
    return [image[i * row_height:(i + 1) * row_height, col_index * col_width:(col_index + 1) * col_width]
            for i in row_indices]

unet_rows = crop_rows(imread(unet_panel_path), unet_rows_idx, col_index=3, total_cols=4)
cellpose_rows = crop_rows(imread(cellpose_panel_path), cellpose_rows_idx, col_index=1, total_cols=2)
otsu_rows = crop_rows(imread(otsu_panel_path), otsu_rows_idx, col_index=1, total_cols=2)

# --- Plot combined grid ---
fig, axes = plt.subplots(num_samples, 5, figsize=(15, num_samples * 3))

for row, tp in enumerate(timepoints):
    # === 1. Brightfield ===
    bf_path = brightfield_dir / f"Vph1_GFP_005_tp{tp:03}_BF_Z_000.tiff"
    bf_img = imread(bf_path)
    if bf_img.ndim == 3:
        bf_img = bf_img[0]
    axes[row, 0].imshow(bf_img, cmap="gray")
    axes[row, 0].set_title("Brightfield")
    axes[row, 0].axis("off")

    # === 2. Ground Truth (GFP) ===
    gfp_path = fluorescence_dir / f"Vph1_GFP_005_tp{tp:03}.tiff"
    gfp_img = imread(gfp_path)
    if gfp_img.ndim == 3:
        gfp_img = gfp_img[0]
    axes[row, 1].imshow(gfp_img, cmap="gray")
    axes[row, 1].set_title("GFP Ground Truth")
    axes[row, 1].axis("off")

    # === 3. U-Net Overlay (only final overlay) ===
    axes[row, 2].imshow(unet_rows[row])
    axes[row, 2].set_title("U-Net Overlay")
    axes[row, 2].axis("off")

    # === 4. Cellpose Overlay (only mask overlay) ===
    axes[row, 3].imshow(cellpose_rows[row])
    axes[row, 3].set_title("Cellpose Overlay")
    axes[row, 3].axis("off")

    # === 5. Otsu Overlay ===
    axes[row, 4].imshow(otsu_rows[row])
    axes[row, 4].set_title("Otsu Overlay")
    axes[row, 4].axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()
print(f"✅ Saved final overlay grid to: {output_path}")
