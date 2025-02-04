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
    # These parts are shared between the two channels
    x = torch.from_numpy(x)
    # Random flip
    if torch.rand(1) > threshold:
        x = F.hflip(x)
    if torch.rand(1) > threshold:
        x = F.vflip(x)
    # Random rotate
    if torch.rand(1) > threshold:
        x = F.rotate(x, 90)

    # # # These parts are not shared between the two channels
    # x, y = x[0], x[1]
    # #  Random noise on fluorescence channel
    # if torch.rand(1) > threshold:
    #     x = x + torch.randn_like(x) * 0.05
    # x = torch.stack([x, y], axis=0)
    # # Return random crop
    return transforms.RandomCrop((96, 96))(x)


def transform(x):
    x = torch.from_numpy(x)
    return F.center_crop(x, (96, 96))


class PositionDataset(Dataset):
    """
    Dataset for a single position.
    """

    def __init__(
        self,
        image_folder: str,
        h5_file: str,
        transform=None,
        percentile=[0.1, 99.9],
        per_tp_normalization=False,
        input_channel=0,
        output_channel=1,
        start_tp=0,
        end_tp=None,
        m_std=3.0,
    ):
        image = DownloadedImageDir(image_folder)
        self.data = image.data
        self.tiler = Tiler.from_h5(image, h5_file)
        self.transform = transform
        # Figure out the metadata
        self.start_tp = start_tp
        self.end_tp = end_tp or self.tiler.no_processed
        self.items = [
            (tile_idx, timepoint)
            for tile_idx in range(self.tiler.no_tiles)
            for timepoint in range(self.start_tp, self.end_tp)
        ]
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.per_tp_normalization = per_tp_normalization
        if self.per_tp_normalization:
            self.input_range = []
            self.output_range = []
            self._init_normalization_per_tp(percentile)
        else:
            self._init_normalization(percentile)

    def _init_normalization_per_tp(self, percentile=[0.1, 99.9]):
        """
        Set up the range of the data for normalization.
        """
        logging.warning("Calculating normalization range, this may take a while.")
        for timepoint in tqdm(range(self.start_tp, self.end_tp)):
            input_range = da.percentile(
                self.data[timepoint, self.input_channel].flatten(), percentile
            ).compute()
            output_range = da.percentile(
                self.data[timepoint, self.output_channel].flatten(), percentile
            ).compute()
            self.input_range.append(input_range)
            self.output_range.append(output_range)
        self.input_range = np.array(self.input_range)
        self.output_range = np.array(self.output_range)

    def _init_normalization(self, percentile=[0.1, 99.9]):
        """
        Set up the range of the data for normalization.
        """
        logging.warning("Calculating normalization range, this may take a while.")
        self.input_range = da.percentile(
            self.data[:, self.input_channel].flatten(), percentile
        ).compute()
        self.output_range = da.percentile(
            self.data[:, self.output_channel].flatten(), percentile
        ).compute()

    def __len__(self):
        return self.tiler.no_tiles * (self.end_tp - self.start_tp)

    def __getitem__(self, idx):
        tile_idx, timepoint = self.items[idx]
        x = self.tiler.get_tile_data(tile_idx, timepoint, c=self.input_channel)
        y = self.tiler.get_tile_data(tile_idx, timepoint, c=self.output_channel)
        # Percentile normalization
        if self.per_tp_normalization:
            x = (x - self.input_range[timepoint, 0]) / (
                self.input_range[timepoint, 1] - self.input_range[timepoint, 0]
            )
            y = (y - self.output_range[timepoint, 0]) / (
                self.output_range[timepoint, 1] - self.output_range[timepoint, 0]
            )
        else:
            x = (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
            y = (y - self.output_range[0]) / (
                self.output_range[1] - self.output_range[0]
            )
        # Clip to [0, 1]
        x = da.clip(x, 0, 1)
        y = da.clip(y, 0, 1)
        if self.transform:
            stacked = np.stack([x, y], axis=0)
            stacked = self.transform(stacked)
            x, y = stacked[0], stacked[1]
        return x, y
