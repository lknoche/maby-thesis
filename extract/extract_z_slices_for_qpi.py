#!/usr/bin/env python3

from pathlib import Path
from skimage.io import imread, imsave

# --- Config ---
input_folder = Path("/home/lknoche/maby/images_structured/Htb2_sfGFP_005")
output_folder = Path("/home/lknoche/maby/qpi_data/Htb2_sfGFP_005")
output_folder.mkdir(parents=True, exist_ok=True)

# --- Step 1: Find all Brightfield Z-slices ---
z_slices = sorted(input_folder.glob("*Brightfield_*.tiff"))
z_stack = 5
timepoints = 5
required = z_stack * timepoints

if len(z_slices) < required:
    raise ValueError(f"Only found {len(z_slices)} Z-slices, need at least {required}.")

# --- Step 2: Copy 25 Z-slices and rename as Z000.tif ... Z024.tif ---
for i in range(required):
    src = z_slices[i]
    dst = output_folder / f"Z{i:03d}.tif"
    img = imread(src)
    imsave(dst, img)
    print(f"✅ Saved slice: {dst.name} (from {src.name})")

print(f"\n🎉 Done! Extracted {required} Z-slices for 5 timepoints. You can now run main.py.")

