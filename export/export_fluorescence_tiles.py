#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from skimage.io import imsave
import h5py
from tqdm import tqdm

# --- Config ---
h5_file = Path("/home/lknoche/maby/2179_2024_05_08_MAYBE_training_00/Vph1_GFP_005.h5")
output_dir = Path("/home/lknoche/maby/fluorescence_full_images/Vph1_GFP_005")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load HDF5 and export 2D images ---
with h5py.File(h5_file, "r") as f:
    try:
        data = f["extraction"]["GFP_Z_bgsub"]["max"]["imBackground"]["values"]
        print(f"✅ Loaded data with shape: {data.shape}")
    except KeyError:
        raise RuntimeError("❌ Could not find 2D fluorescence images in HDF5.")

    if data.ndim != 4:
        raise RuntimeError(f"❌ Expected 4D data (tp, trap, H, W), but got shape {data.shape}")

    n_tp, n_traps, height, width = data.shape
    print(f"✅ Exporting {n_tp} timepoints × {n_traps} traps")

    for tp in tqdm(range(n_tp), desc="Exporting timepoints"):
        for trap in range(n_traps):
            img = data[tp, trap]
            if img.ndim != 2:
                continue  # skip corrupted or empty
            fname = output_dir / f"tp{tp:04d}_trap{trap:03d}.tiff"
            imsave(fname, img.astype(np.float32), check_contrast=False)
