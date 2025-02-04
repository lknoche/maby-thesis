"""
The MABY testing script.
- [x] Load a model from a checkpoint.
- [x] Load a test dataset.
- [x] Run the model on the test dataset and compute Correlation Coefficient on the fly
- [x] Compute segmentation metrics on the fly
"""

import argparse
from dataset import PositionDataset, transform
from dlmbl_unet import UNet
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm


def fluo_to_segmentation(fluo, threshold=0.5):
    """
    Convert a fluorescence image to a segmentation using simple thresholding.
    """
    fluo = fluo.flatten(1)
    return (fluo > threshold).float()


def f1_score(y_true, y_pred):
    """
    Compute the F1 score between two batches of binary segmentations.
    Does not handle the case where both y_true and y_pred are all zeros.
    Does not reduce the batch dimension.
    """
    tp = (y_true * y_pred).sum(dim=1)
    fp = ((1 - y_true) * y_pred).sum(dim=1)
    fn = (y_true * (1 - y_pred)).sum(dim=1)
    return 2 * tp / (2 * tp + fp + fn)


def evaluate(
    gfp_type,
    train_date,
    test_position=5,
    start_tp=0,
    end_tp=100,
    test_batch_size=16,
    unet_depth=3,
    in_channels=5,
    out_channels=5,
    model_checkpoint="best_model.pth",
    output_file_name="test_results.csv",
):
    base_dir = Path("/nrs/funke/adjavond/maby/data")
    data_dir = base_dir / "MAYBE_training_00"
    metadata_dir = base_dir / "2179_2024_05_08_MAYBE_training_00"
    output_dir = Path("/nrs/funke/adjavond/maby/output")
    experiment_dir = output_dir / f"{train_date}_train_{gfp_type}"

    assert experiment_dir.exists(), f"{experiment_dir} does not exist."
    model_checkpoint = torch.load(experiment_dir / model_checkpoint)

    logging.info(f"Loaded checkpoint for epoch {model_checkpoint['epoch']}")

    test_dataset = PositionDataset(
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
        num_workers=32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(
        depth=unet_depth,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    model.load_state_dict(model_checkpoint["model"])
    model = model.to(device)
    model.eval()

    # Define the correlation coefficient
    criterion = nn.CosineSimilarity(dim=1)

    similarities = []
    segmentation_f1 = {
        0.1: [],
        0.2: [],
        0.3: [],
        0.4: [],
        0.5: [],
        0.6: [],
        0.7: [],
        0.8: [],
        0.9: [],
    }
    with torch.inference_mode():
        for x, y in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            x = x.float().to(device)
            y = y.float().to(device)
            y = y / y.flatten(1).max(dim=1).values[:, None, None, None]
            y_hat = model(x)
            similarities.extend(
                criterion(y.flatten(1), y_hat.flatten(1))
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )

            for threshold, f1 in segmentation_f1.items():
                seg_y = fluo_to_segmentation(y, threshold=threshold)
                seg_y_hat = fluo_to_segmentation(y_hat, threshold=threshold)
                f1.extend(f1_score(seg_y, seg_y_hat).tolist())

    logging.info(f"Mean similarity: {sum(similarities) / len(similarities)}")
    for threshold, f1 in segmentation_f1.items():
        logging.info(f"Mean F1 score at threshold {threshold}: {sum(f1) / len(f1)}")
    # Store the correlations
    with open(experiment_dir / output_file_name, "w") as fd:
        # Write the similarities F1 scores for each threshold
        f1_header = ",".join([f"F1@{t}" for t in segmentation_f1.keys()])
        fd.write(f"Similarity,{f1_header}\n")
        for s, f1s in zip(similarities, zip(*segmentation_f1.values())):
            f1s = ",".join([str(f) for f in f1s])
            fd.write(f"{s},{f1s}\n")
    logging.info("Results saved.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gfp_type", type=str, required=True)
    parser.add_argument("-t", "--train_date", type=str, required=True)
    parser.add_argument("-p", "--test_position", type=int, default=5)
    parser.add_argument("-e", "--end_tp", type=int, default=100)
    parser.add_argument("-b", "--test_batch_size", type=int, default=16)
    parser.add_argument("-d", "--unet_depth", type=int, default=3)
    parser.add_argument("-i", "--in_channels", type=int, default=5)
    parser.add_argument("-o", "--out_channels", type=int, default=5)
    parser.add_argument("-m", "--model_checkpoint", type=str, default="best_model.pth")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
    )
    args = parse_args()
    evaluate(
        args.gfp_type,
        args.train_date,
        args.test_position,
        args.end_tp,
        args.test_batch_size,
        args.unet_depth,
        args.in_channels,
        args.out_channels,
        args.model_checkpoint,
    )
