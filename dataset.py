import torch
from torch.utils.data import Dataset
from image import DownloadedImageDir
from aliby.tile.tiler import Tiler
import dask.array as da
import logging
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F

def augment(x, threshold=0.5):
    x = torch.from_numpy(x)
    if torch.rand(1) > threshold:
        x = F.hflip(x)
    if torch.rand(1) > threshold:
        x = F.vflip(x)
    if torch.rand(1) > threshold:
        x = F.rotate(x, 90)
    return transforms.RandomCrop((96, 96))(x)

def transform(x):
    x = torch.from_numpy(x)
    return F.center_crop(x, (96, 96))

class PositionDataset(Dataset):
    def __init__(
        self,
        image,
        h5_file: str,
        transform=None,
        percentile=[0.1, 99.9],
        input_channel=0,
        output_channel=1,
        start_tp=0,
        end_tp=None,
    ):
        self.data = image.get_data_lazy()
        self.tiler = Tiler.from_h5(image, h5_file)
        self.transform = transform
        self.start_tp = start_tp
        self.end_tp = min(end_tp or self.tiler.no_processed, self.data.shape[0])
        self.items = [
            (tile_idx, timepoint)
            for tile_idx in range(self.tiler.no_tiles)
            for timepoint in range(self.start_tp, self.end_tp)
        ]
        self.input_channel = input_channel
        self.output_channel = output_channel

        # Only normalize input
        logging.warning("Calculating input normalization range")
        self.input_range = da.percentile(
            self.data[:, self.input_channel].flatten(), percentile
        ).compute()

    def __len__(self):
        return self.tiler.no_tiles * (self.end_tp - self.start_tp)

    def __getitem__(self, idx):
        tile_idx, timepoint = self.items[idx]
        x = self.tiler.get_tile_data(tile_idx, timepoint, c=self.input_channel)
        y = self.tiler.get_tile_data(tile_idx, timepoint, c=self.output_channel)

        # Normalize x only
        x = (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        x = da.clip(x, 0, 1)

        # Don't normalize y
        y = da.clip(y, 0, None)

        if self.transform:
            stacked = np.stack([x, y], axis=0)
            stacked = self.transform(stacked)
            x, y = stacked[0], stacked[1]

        return x, y

class LocalPositionDataset(PositionDataset):
    def __init__(
        self,
        image_folder: str,
        h5_file: str,
        transform=None,
        percentile=[0.1, 99.9],
        input_channel=0,
        output_channel=1,
        start_tp=0,
        end_tp=None,
    ):
        image = DownloadedImageDir(image_folder)
        super().__init__(
            image,
            h5_file,
            transform,
            percentile,
            input_channel,
            output_channel,
            start_tp,
            end_tp,
        )
