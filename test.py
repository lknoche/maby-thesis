#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MABY testing script — simplified.
Just pass the GFP type (e.g. Htb2_sfGFP), and it auto-selects the latest run.
"""

import argparse
from dataset import LocalPositionDataset, transform
from model import UNet
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm
from datetime import datetime

def fluo_to_segmentation(fluo, threshold=0.5):
    fluo = fluo.flatten(1)
    return (fluo > threshold).float()

def f1_score(y_true, y_pred):
    tp = (y_true * y_pred).sum(dim=1)
    fp = ((1 - y_true) * y_pred).sum(dim=1)
    fn = (y_true * (1 - y_pred)).sum(dim=1)
    return 2 * tp / (2 * tp + fp + fn)

def find_latest_experiment(output_dir, gfp_type):
    candidates = sorted(
        [d for d in output_dir.glob(f"*{gfp_type}") if d.is_dir()],
        reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No output folder found for GFP type: {gfp_type}")
    return candidates[0]

def evaluate(
    gfp_type,
    test_position=5,
    start_tp=0,
    end_tp=100,
    test_batch_size=16,
    unet_depth=3,
    in_channels=5,
    out_channels=5,
    model_checkpoint="model_19.pt",
    output_file_name="test_results.csv",
):
    base_dir = Path.cwd()
    data_dir = base_dir / "images_structured"
    metadata_dir = base_dir / "2179_2024_05_08_MAYBE_training_00"
    output_dir = base_dir / "output"

    experiment_dir = find_latest_experiment(output_dir, gfp_type)
    print(f"🧪 Using model from: {experiment_dir}")
    model_path = experiment_dir / model_checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model_checkpoint_data = torch.load(model_path)

    test_dataset = LocalPositionDataset(
        image_folder=data_dir / f"{gfp_type}_{test_position:03d}",
        h5_file=metadata_dir / f"{gfp_type}_{test_position:03d}.h5",
        transform=transform,
        start_tp=start_tp,
        end_tp=end_tp,
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(depth=unet_depth, in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(model_checkpoint_data["model"])
    model.to(device)
    model.eval()

    criterion = nn.CosineSimilarity(dim=1)

    similarities = []
    segmentation_f1 = {t: [] for t in [0.1 * i for i in range(1, 10)]}
    with torch.inference_mode():
        for x, y in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            x = x.float().to(device)
            y = y.float().to(device)
            y = y / y.flatten(1).max(dim=1).values[:, None, None, None]
            y_hat = model(x)
            similarities.extend(criterion(y.flatten(1), y_hat.flatten(1)).cpu().tolist())

            for threshold in segmentation_f1:
                seg_y = fluo_to_segmentation(y, threshold)
                seg_y_hat = fluo_to_segmentation(y_hat, threshold)
                segmentation_f1[threshold].extend(f1_score(seg_y, seg_y_hat).tolist())

    logging.info(f"Mean similarity: {sum(similarities) / len(similarities):.4f}")
    for threshold, f1 in segmentation_f1.items():
        logging.info(f"Mean F1 score at threshold {threshold:.1f}: {sum(f1) / len(f1):.4f}")

    with open(experiment_dir / output_file_name, "w") as fd:
        fd.write("Similarity," + ",".join([f"F1@{t:.1f}" for t in segmentation_f1]) + "\n")
        for s, f1s in zip(similarities, zip(*segmentation_f1.values())):
            fd.write(f"{s}," + ",".join(map(str, f1s)) + "\n")
    logging.info("Results saved.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gfp_type", type=str, required=True, help="e.g., Vph1 or Htb2_sfGFP")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
    args = parse_args()
    evaluate(args.gfp_type)
