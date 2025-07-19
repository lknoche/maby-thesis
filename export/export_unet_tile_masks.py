#!/usr/bin/env python3

from pathlib import Path
from model import UNet
from dataset import LocalPositionDataset, transform
import torch
import numpy as np
from skimage.filters import threshold_otsu
from skimage.io import imsave
from tqdm import tqdm

# === CONFIG ===
base_dir = Path("/home/lknoche/maby")
output_dir = base_dir / "figures/unet_tile_masks"
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = base_dir / "output/2025-05-26_train_Htb2_sfGFP_resume/model_39.pt"
image_folder = base_dir / "images_structured/Htb2_sfGFP_005"
h5_file = base_dir / "2179_2024_05_08_MAYBE_training_00/Htb2_sfGFP_005.h5"

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Model ===
model = UNet(depth=3, in_channels=5, out_channels=5, final_activation=torch.nn.Sigmoid()).to(device)
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state["model"])
model.eval()

# === Load Dataset ===
dataset = LocalPositionDataset(
    image_folder=image_folder,
    h5_file=h5_file,
    transform=transform,
)

print(f"📦 Exporting U-Net predictions as masks for {len(dataset)} tiles")

# === Predict + Threshold ===
for idx in tqdm(range(len(dataset))):
    x, _ = dataset[idx]
    x_tensor = x.unsqueeze(0).float().to(device)

    with torch.no_grad():
        y_hat = model(x_tensor).cpu().squeeze(0)[1]  # channel 1 = nucleus

    otsu_val = threshold_otsu(y_hat.numpy())
    mask = (y_hat > otsu_val).numpy().astype(np.uint8) * 255

    out_path = output_dir / f"tile_{idx:03d}_unet.tiff"
    imsave(out_path, mask)
    # print(f"✅ Saved: {out_path}")

print("🎉 All predicted U-Net tile masks exported.")

