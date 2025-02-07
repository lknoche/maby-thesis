"""
The MABY training script.
"""

import argparse
import datetime
from dataset import PositionDataset, augment, transform
from model import UNet
import logging
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import wandb


def main(
    gfp_type: str,
    train_positions: list[int] = [1, 2, 3],
    val_position: int = 4,
    test_position: int = 5,
    unet_depth: int = 3,
    in_channels: int = 5,
    out_channels: int = 5,
    lr: float = 1e-3,
    batch_size: int = 16,
    val_batch_size: int = 16,
    num_workers: int = 32,
    num_epochs: int = 40,
):
    # Training dataset
    base_dir = Path("/nrs/funke/adjavond/maby/data")
    data_dir = base_dir / "MAYBE_training_00"
    metadata_dir = base_dir / "2179_2024_05_08_MAYBE_training_00"
    output_dir = Path("/nrs/funke/adjavond/maby/output")

    today = datetime.date.today().strftime("%Y-%m-%d")
    experiment_name = f"{today}_train_{gfp_type}"
    this_experiment_dir = output_dir / experiment_name

    this_experiment_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Loading the training dataset")
    train_dataset = torch.utils.data.ConcatDataset(
        [
            PositionDataset(
                image_folder=data_dir / f"{gfp_type}_{position:03d}",
                h5_file=metadata_dir / f"{gfp_type}_{position:03d}.h5",
                transform=augment,
                end_tp=100,
            )
            for position in train_positions
        ]
    )

    logging.info("Loading the validation dataset")
    # Validation dataset
    val_dataset = PositionDataset(
        image_folder=data_dir / f"{gfp_type}_{val_position:03d}",
        h5_file=metadata_dir / f"{gfp_type}_{val_position:03d}.h5",
        transform=transform,
        end_tp=100,
    )

    logging.info("Setting up the model")
    # Setup the model
    model = UNet(
        depth=unet_depth,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Setup wandb
    logging.info("Setting up wandb")
    run = wandb.init(
        project="maybe",
        name=experiment_name,
        notes=f"Training the {gfp_type} model",
        config={
            "unet_depth": unet_depth,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "lr": lr,
            "batch_size": batch_size,
            "val_batch_size": val_batch_size,
            "num_workers": num_workers,
            "num_epochs": num_epochs,
            "train_positions": train_positions,
            "val_position": val_position,
            "test_position": test_position,
        },
    )
    to_log = {}

    logging.info("Starting training")
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []
        model.train()
        # Training
        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}"):
            x, y = x.float().to(device), y.float().to(device)
            # Divide by the max value per item in batch
            y = y / y.flatten(1).max(dim=1).values[:, None, None, None]

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        to_log["train_loss"] = sum(epoch_losses) / len(epoch_losses)
        # Validation
        validation_loss = 0
        with torch.inference_mode():
            for x, y in tqdm(val_dataloader, total=len(val_dataloader)):
                model.eval()
                x, y = x.float().to(device), y.float().to(device)
                y = y / y.flatten(1).max(dim=1).values[:, None, None, None]
                y_hat = model(x)
                validation_loss += criterion(y_hat, y).item()
        to_log["val_loss"] = validation_loss / len(val_dataloader)
        logging.info(
            f"Epoch {epoch}: Train loss: {to_log['train_loss']}, Val loss: {to_log['val_loss']}"
        )
        to_log["images"] = wandb.Image(x[0][:, None], caption="Input")
        to_log["ground_truth"] = wandb.Image(y[0][:, None], caption="Ground truth")
        to_log["prediction"] = wandb.Image(y_hat[0][:, None], caption="Prediction")
        run.log(to_log, step=epoch)
        # Save the model, optimizer, epoch, and losses
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "losses": epoch_losses,
            },
            this_experiment_dir / f"model_{epoch}.pt",
        )

    run.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gfp_type", type=str, required=True)
    parser.add_argument("-e", "--num_epochs", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
    )
    args = parse_args()
    main(args.gfp_type, num_epochs=args.num_epochs)
