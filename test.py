#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MABY testing script — ALL or partial timepoints, fixed @0.1 or Otsu.
"""

import argparse
import logging
from pathlib import Path
import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from skimage import filters
from dataset import LocalPositionDataset, transform
from model import UNet

evaluation_map = {
    "Hog1_GFP": "fixed",
    "Vph1_GFP": "fixed",
    "Htb2_sfGFP": "fixed",
    "YST_665": "otsu",
}

def get_end_tp(h5_path):
    with h5py.File(h5_path, "r") as f:
        all_tps = np.unique(f["cell_info/timepoint"][:])
        return len(all_tps)

def find_latest_experiment(output_dir, gfp_type):
    candidates = sorted([d for d in output_dir.glob(f"*{gfp_type}*") if d.is_dir()], reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No output folder found for GFP type: {gfp_type}")
    return candidates[0]

def identify_nucleus(nmi: np.ndarray):
    if np.max(nmi) - np.min(nmi) < 1e-4:
        return np.zeros(nmi.shape, dtype=bool)
    norm = (nmi - np.min(nmi)) / (np.max(nmi) - np.min(nmi) + 1e-8)
    blurred = filters.gaussian(norm, sigma=1.0)
    otsu_val = filters.threshold_otsu(blurred)
    return blurred > otsu_val

def fluo_to_otsu_segmentation(pred_batch):
    pred_batch = pred_batch[:, 1]
    seg = torch.zeros_like(pred_batch)
    for i in range(pred_batch.shape[0]):
        pred_np = pred_batch[i].cpu().numpy()
        seg_np = identify_nucleus(pred_np)
        seg[i] = torch.from_numpy(seg_np.astype(np.float32))
    return seg

def fluo_to_fixed_segmentation(x, threshold):
    return (x[:, 1] > threshold).float()

def f1_score(y_true, y_pred):
    y_true = y_true.flatten(1)
    y_pred = y_pred.flatten(1)
    tp = (y_true * y_pred).sum(dim=1)
    fp = ((1 - y_true) * y_pred).sum(dim=1)
    fn = (y_true * (1 - y_pred)).sum(dim=1)
    return 2 * tp / (2 * tp + fp + fn + 1e-8)

def iou_score(y_true, y_pred):
    y_true = y_true.flatten(1)
    y_pred = y_pred.flatten(1)
    intersection = (y_true * y_pred).sum(dim=1)
    union = ((y_true + y_pred) > 0).float().sum(dim=1)
    return intersection / (union + 1e-8)

def evaluate(gfp_type, model_checkpoint="model_19.pt", start_tp=0, end_tp=None):
    base_dir = Path.cwd()
    data_dir = base_dir / "images_structured"
    metadata_dir = base_dir / "2179_2024_05_08_MAYBE_training_00"
    output_dir = base_dir / "output"
    experiment_dir = find_latest_experiment(output_dir, gfp_type)
    print(f"🧪 Using model from: {experiment_dir}")
    model_path = experiment_dir / model_checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model_checkpoint_data = torch.load(model_path, map_location=torch.device("cpu"))

    h5_path = metadata_dir / f"{gfp_type}_005.h5"
    if end_tp is None:
        end_tp = 100
    print(f"📈 Evaluating TPs {start_tp} to {end_tp}")

    test_dataset = LocalPositionDataset(
        image_folder=data_dir / f"{gfp_type}_005",
        h5_file=h5_path,
        transform=transform,
        start_tp=start_tp,
        end_tp=end_tp,
        fixed_output_range=True,
    )

    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(depth=3, in_channels=5, out_channels=5, final_activation=None)
    model.load_state_dict(model_checkpoint_data["model"])
    model.to(device)
    model.eval()

    criterion = nn.CosineSimilarity(dim=1)
    similarities, f1s, ious = [], [], []

    method = evaluation_map.get(gfp_type, "fixed")

    with torch.inference_mode():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x = x.float().to(device)
            y = y.float().to(device)
            y_hat_raw = model(x)
            y_hat = (y_hat_raw - y_hat_raw.min()) / (y_hat_raw.max() - y_hat_raw.min() + 1e-8)

            similarities.extend(criterion(y.flatten(1), y_hat.flatten(1)).cpu().tolist())

            if method == "otsu":
                seg_y_hat = fluo_to_otsu_segmentation(y_hat)
                seg_y = fluo_to_otsu_segmentation(y)
                f1s.extend(f1_score(seg_y, seg_y_hat).tolist())
                ious.extend(iou_score(seg_y, seg_y_hat).tolist())
            else:
                threshold = 0.1
                seg_y_hat = fluo_to_fixed_segmentation(y_hat, threshold)
                seg_y = fluo_to_fixed_segmentation(y, threshold)
                f1s.extend(f1_score(seg_y, seg_y_hat).tolist())

    logging.info(f"Mean similarity: {np.mean(similarities):.4f}")
    if method == "otsu":
        logging.info(f"Mean Otsu+Gaussian F1 score: {np.mean(f1s):.4f}")
        logging.info(f"Mean Otsu+Gaussian IoU score: {np.mean(ious):.4f}")
    else:
        logging.info(f"Mean F1@0.1 score: {np.mean(f1s):.4f}")

    tag = f"{start_tp}_{end_tp}" if (start_tp > 0 or end_tp is not None) else "all"
    out_file = experiment_dir / (
        f"test_results_otsu_{tag}.csv" if method == "otsu" else f"test_results_fixed_0.1_{tag}.csv"
    )
    with open(out_file, "w") as f:
        if method == "otsu":
            f.write("Similarity,F1_Otsu,IoU_Otsu\n")
            for s, f1v, iou in zip(similarities, f1s, ious):
                f.write(f"{s:.4f},{f1v:.4f},{iou:.4f}\n")
        else:
            f.write("Similarity,F1@0.1\n")
            for s, f1v in zip(similarities, f1s):
                f.write(f"{s:.4f},{f1v:.4f}\n")
    logging.info(f"✅ Results saved to {out_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gfp_type", required=True, help="e.g. Vph1_GFP or Htb2_sfGFP")
    parser.add_argument("-m", "--model_checkpoint", default="model_19.pt", help="Model file")
    parser.add_argument("--start_tp", type=int, default=0, help="Start timepoint")
    parser.add_argument("--end_tp", type=int, default=None, help="End timepoint (exclusive)")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
    args = parse_args()
    evaluate(
        gfp_type=args.gfp_type,
        model_checkpoint=args.model_checkpoint,
        start_tp=args.start_tp,
        end_tp=args.end_tp
    )
# %%

