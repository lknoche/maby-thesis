import os
import subprocess
from pathlib import Path
from omero.gateway import BlitzGateway
from PIL import Image
import numpy as np
import logging

# --- Config ---
HOST = 'staffa.bio.ed.ac.uk'
USERNAME = 'upload'
PASSWORD = 'gothamc1ty'
PORT = 4064
DATASET_ID = 2179
GFP_TYPE = "Hog1"  
GFP_PREFIX = GFP_TYPE  
TIFF_DIR = Path("images_structured")
TIFF_DIR.mkdir(exist_ok=True)
LOG_PATH = Path("check_timepoints.log")

# --- Logging Setup ---
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s"
)

def tiffs_already_downloaded(gfp_prefix):
    structured_dir = TIFF_DIR / f"{gfp_prefix}_001"
    if not structured_dir.exists():
        return False
    return any(structured_dir.glob("*.tiff"))

def download_tiffs():
    print("📡 Connecting to OMERO...")
    conn = BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT)
    conn.connect()
    dataset = conn.getObject("Dataset", DATASET_ID)
    if dataset is None:
        raise ValueError(f"No dataset with ID {DATASET_ID} found.")

    print(f"⬇️ Downloading TIFFs with original names for {GFP_PREFIX}...")
    for image in dataset.listChildren():
        name = image.getName()
        if not name.startswith(GFP_PREFIX):
            continue

        img_dir = TIFF_DIR / name
        img_dir.mkdir(exist_ok=True)

        pixels = image.getPrimaryPixels()
        size_z = image.getSizeZ()
        size_t = image.getSizeT()
        size_c = image.getSizeC()
        logging.info(f"{name}: Z={size_z}, T={size_t}, C={size_c}")

        for t in range(size_t):
            for c in range(size_c):
                for z in range(size_z):
                    plane = pixels.getPlane(z, c, t)
                    plane_np = np.asarray(plane)
                    if c == 0:
                        c_name = "Brightfield"
                    elif c == 1:
                        c_name = "GFP_Z"
                    else:
                        c_name = f"C{c}"
                    filename = f"{name}_{t:06d}_{c_name}_{z:03d}.tiff"
                    outpath = img_dir / filename
                    if not outpath.exists():
                        Image.fromarray(plane_np).save(outpath)
                        print(f"✅ Saved {outpath}")

    conn.close()

def analyze_timepoints():
    print("🔍 Analyzing completeness of timepoints...")
    expected_channels = ["Brightfield", "GFP_Z"]
    expected_z = 5  # Adjust if needed
    for subfolder in TIFF_DIR.glob(f"{GFP_TYPE}_GFP_*"):
        counts = {}
        for file in subfolder.glob("*.tiff"):
            parts = file.name.split("_")
            try:
                t_idx = int(parts[3])
            except Exception:
                continue
            counts.setdefault(t_idx, []).append(file.name)
        for t, files in sorted(counts.items()):
            if len(files) < len(expected_channels) * expected_z:
                logging.warning(f"{subfolder.name}: Timepoint {t} is incomplete ({len(files)} files)")
            else:
                logging.info(f"{subfolder.name}: Timepoint {t} is complete ({len(files)} files)")

def run_training():
    print("🚀 Starting training with train.py...")
    cmd = [
        "python", "train.py",
        "-b", str(Path.cwd()),
        "-g", GFP_TYPE
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0

def cleanup_tiffs():
    print("🧹 Cleaning up TIFFs (excluding position 005)...")
    for subfolder in TIFF_DIR.glob(f"{GFP_TYPE}_*"):
        pos_id = subfolder.name.split("_")[-1]
        if pos_id != "005":
            print(f"🗑️ Deleting: {subfolder}")
            for tif in subfolder.glob("*.tiff"):
                tif.unlink()

def main():
    if not tiffs_already_downloaded(GFP_PREFIX):
        download_tiffs()
    else:
        print("✅ TIFFs already present. Skipping download.")

    analyze_timepoints()

    success = run_training()

    if success:
        # cleanup_tiffs()  # optional cleanup
        print("✅ Training completed successfully.")
    else:
        print("❌ Training failed. TIFFs kept for debugging.")

if __name__ == "__main__":
    main()
