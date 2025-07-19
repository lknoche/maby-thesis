#!/usr/bin/env python3

from pathlib import Path
from skimage.io import imread, imsave

# --- CONFIG ---
input_dir = Path("/home/lknoche/maby/fluorescence_full_images/Vph1_GFP_005")
output_dir = Path("/home/lknoche/maby/tiles/fluorescence_selected_tp_p005")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Parameters ---
timepoints = [3, 4, 13, 17, 27]
position = 5
z_slice = "00"

print(f"🔍 Extracting {len(timepoints)} images for p{position:03d}, z{z_slice}...")

for tp in timepoints:
    filename = f"Vph1_GFP_005_t{tp:03d}_p{position:03d}_z{z_slice}.tiff"
    input_path = input_dir / filename

    if input_path.exists():
        img = imread(input_path)
        out_path = output_dir / filename
        imsave(out_path, img)
        print(f"✅ Saved: {out_path.name}")
    else:
        print(f"⚠️  Missing file: {input_path.name}")

print("\n🏁 Done.")
