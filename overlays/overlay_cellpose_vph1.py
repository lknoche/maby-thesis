#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import label2rgb
import numpy as np

# --- Config ---
brightfield_dir = Path("/home/lknoche/maby/tiles/brightfield_selected_unetlike_Vph1_GFP_005")
mask_dir = Path("/home/lknoche/maby/figures/cellpose_selected_unetlike")
figures_dir = Path("/home/lknoche/maby/figures")
figures_dir.mkdir(exist_ok=True)

timepoints = [3,27, 13, 4, 17]

# --- Plot grid ---
fig, axes = plt.subplots(len(timepoints), 2, figsize=(6, 3 * len(timepoints)))

for row, tp in enumerate(timepoints):
    tile_path = brightfield_dir / f"Vph1_GFP_005_tp{tp:03}_BF_Z_000.tiff"
    mask_path = mask_dir / f"mask_Vph1_GFP_005_tp{tp:03}.tiff"

    bf_img = imread(tile_path)
    mask = imread(mask_path)

    if bf_img.ndim == 3:
        bf_img = bf_img[0]

    if bf_img.max() > 1:
        bf_img = bf_img.astype(np.float32) / 255.0

    overlay = label2rgb(mask, image=bf_img, alpha=0.5, bg_label=0)

    axes[row, 0].imshow(bf_img, cmap="gray")
    axes[row, 0].set_title(f"Brightfield")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(overlay)
    axes[row, 1].set_title("Cellpose Mask Overlay")
    axes[row, 1].axis("off")

plt.tight_layout()
save_path = figures_dir / "Vph1_GFP_cellpose_BF_overlay_panel.png"
plt.savefig(save_path, dpi=300)
plt.close()
print(f"✅ Saved overlay panel: {save_path}")
print(f"🧪 Used timepoints: {timepoints}")
