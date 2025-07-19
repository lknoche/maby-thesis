#!/usr/bin/env python3

from pathlib import Path
from skimage.io import imsave
from dataset import LocalPositionDataset, transform

# === CONFIG ===
base_dir = Path("/home/lknoche/maby")
data_dir = base_dir / "images_structured"
metadata_dir = base_dir / "2179_2024_05_08_MAYBE_training_00"
out_dir = base_dir / "tiles/gt_fluorescence_tiles"
out_dir.mkdir(parents=True, exist_ok=True)

# === GFP TYPE to export ===
gfp_type = "Vph1_GFP"        # z.B. auch: "Htb2_sfGFP"
position = "005"
image_folder = data_dir / f"{gfp_type}_{position}"
h5_file = metadata_dir / f"{gfp_type}_{position}.h5"

# === Lade Dataset ===
dataset = LocalPositionDataset(
    image_folder=image_folder,
    h5_file=h5_file,
    transform=transform,
)

print(f"📦 Loaded {len(dataset)} tiles from {gfp_type}_{position}")

# === Exportiere y[1] als TIFF ===
for i in range(len(dataset)):
    _, y = dataset[i]
    fluorescence_img = y[1].numpy()  # float32, values [0, 1]
    out_path = out_dir / f"flu_tile_{i:03d}.tiff"
    imsave(out_path, (fluorescence_img * 255).astype("uint8"))

print(f"✅ Saved {len(dataset)} fluorescence images to {out_dir}")
