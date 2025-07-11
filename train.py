import argparse
import datetime
import logging
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from dataset import LocalPositionDataset, augment, transform
from model import UNet
import matplotlib.pyplot as plt


def main(base_dir: str, gfp_type: str, num_epochs: int = 30):
    logging.basicConfig(level=logging.INFO)
    base_dir = Path(base_dir)
    data_dir = base_dir / "images_structured"
    metadata_dir = base_dir / "2179_2024_05_08_MAYBE_training_00"
    output_dir = base_dir / "output"
    today = datetime.date.today().strftime("%Y-%m-%d")
    experiment_name = f"{today}_train_{gfp_type}"
    this_experiment_dir = output_dir / experiment_name
    this_experiment_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Loading the training dataset")
    train_positions = [1, 2, 3, 4]
    train_dataset = ConcatDataset([
        LocalPositionDataset(
            image_folder=data_dir / f"{gfp_type}_{pos:03d}",
            h5_file=metadata_dir / f"{gfp_type}_{pos:03d}.h5",
            transform=augment,
            end_tp=100,
        ) for pos in train_positions
    ])

    logging.info("Loading the validation dataset")
    val_dataset = LocalPositionDataset(
        image_folder=data_dir / f"{gfp_type}_005",
        h5_file=metadata_dir / f"{gfp_type}_005.h5",
        transform=transform,
        end_tp=100,
    )

    logging.info("Setting up the model")
    model = UNet(depth=3, in_channels=5, out_channels=5, final_activation=None)  # no activation
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    logging.info("Setting up wandb")
    run = wandb.init(
        project="maybe",
        name=experiment_name,
        notes="Full dataset training with input-only normalization and histogram logging",
        config={
            "epochs": num_epochs,
            "gfp": gfp_type,
            "train_positions": train_positions,
            "val_position": 5,
        },
    )

    logging.info("Starting full training")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.float().to(device), y.float().to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        logging.info(f"Epoch {epoch}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # 🔍 Debug: Log value ranges
        logging.info(f"[DEBUG] x min/max: {x.min().item():.4f} / {x.max().item():.4f}")
        logging.info(f"[DEBUG] y min/max: {y.min().item():.4f} / {y.max().item():.4f}")
        logging.info(f"[DEBUG] y_hat min/max: {y_hat.min().item():.4f} / {y_hat.max().item():.4f}")

        if epoch == 0:
            hist_path = this_experiment_dir / "yhat_hist_epoch0.png"
            plt.hist(y_hat.detach().cpu().numpy().flatten(), bins=100)
            plt.title("y_hat distribution at epoch 0")
            plt.savefig(hist_path)
            logging.info(f"✅ Saved histogram to {hist_path}")
            plt.close()

        run.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train/input": wandb.Image(x[0][0].detach().cpu().numpy(), caption="Input ch 0"),
            "train/target": wandb.Image(y[0][0].detach().cpu().numpy(), caption="Target ch 0"),
            "train/prediction": wandb.Image(y_hat[0][0].detach().cpu().numpy(), caption="Prediction ch 0"),
        }, step=epoch)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, this_experiment_dir / f"model_{epoch}.pt")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_dir", required=True)
    parser.add_argument("-g", "--gfp_type", required=True)
    parser.add_argument("-e", "--num_epochs", type=int, default=20)
    parsed = parser.parse_args()
    main(parsed.base_dir, parsed.gfp_type, parsed.num_epochs)
