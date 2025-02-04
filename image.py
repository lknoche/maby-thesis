from aliby.io.image import BaseLocalImage
from pathlib import Path
import re
import dask  # for delayed
import dask.array as da
from skimage.io import imread


class DownloadedExperiment:
    """
    Groups all of the positions for a single experiment.
    Each subdirectory corresponds to one position.
    """

    def __init__(self, image_folder: str):
        self.image_folder = Path(image_folder)
        self.positions = self._get_positions()

    def _get_positions(self):
        """
        Get the positions in the image folder
        """
        return sorted([x.name for x in self.image_folder.iterdir() if x.is_dir()])


class DownloadedImageDir(BaseLocalImage):
    # TODO match to BaseLocalImage
    """
    Images are stored in a directory as tiff files by default.
    """

    def __init__(self, image_folder: str, suffix: str = "tif"):
        self.image_folder = Path(image_folder)
        # Assuming the experiment name is the parent folder
        self.experiment_name = self.image_folder.parent.name
        # Assuming the position name is the folder name
        self._name = self.image_folder.name
        self.suffix = suffix
        self.meta, self.filenames = self._read_metadata()

    def _read_metadata(self):
        """
        Get all metadata for a position by reading its files.
        """
        files = sorted(
            [x.stem for x in self.image_folder.iterdir() if x.suffix == ".tif"]
        )
        file_metadata = [self._read_filename(file) for file in files]
        # Get all channels
        channels = set([x["channel"] for x in file_metadata])
        channels = sorted(channels)
        # Group metadata by channel
        metadata = {
            channel: [x for x in file_metadata if x["channel"] == channel]
            for channel in channels
        }
        # Get the maximum timepoint for each channel
        # NOTE +1 because 0-based indexing
        max_timepoint = {
            channel: max([x["timepoint"] for x in metadata[channel]]) + 1
            for channel in channels
        }
        # Make sure that all channels have the same number of timepoints
        if len(set(max_timepoint.values())) > 1:
            raise ValueError("Channels have different number of timepoints")
        # Get the maximum z slice
        # NOTE +1 because 0-based indexing
        max_z = {
            channel: max([x["z"] for x in metadata[channel]]) + 1
            for channel in channels
        }
        # Make sure that all timepoints have the same number of z slices
        # NOTE +1 because 0-based indexing
        for channel in channels:
            for timepoint in range(max_timepoint[channel]):
                max_z_timepoint = (
                    max(
                        [
                            x["z"]
                            for x in metadata[channel]
                            if x["timepoint"] == timepoint
                        ]
                    )
                    + 1
                )
                if max_z_timepoint != max_z[channel]:
                    raise ValueError(
                        f"Timepoint {timepoint} has different number of z slices"
                    )
        # Make sure that both channels have the same number of z slices
        if len(set(max_z.values())) > 1:
            raise ValueError("Channels have different number of z slices")
        # Keep only the maximum timepoint and z slice
        max_z = max(max_z.values())
        max_timepoint = max(max_timepoint.values())
        # Finally, assuming all images have the same shape, get the shape of the first image
        img = imread(self.image_folder / (files[0] + f".{self.suffix}"))
        size_y, size_x = img.shape
        dtype = img.dtype

        # Create the metadata for the experiment
        experiment_metadata = {
            "channels": channels,
            "size_t": max_timepoint,
            "size_c": len(channels),
            "size_z": max_z,
            "size_y": size_y,
            "size_x": size_x,
            "dtype": dtype,
        }
        # Create the filenames list in the correct order (TZCYX)
        filenames = []
        for timepoint in range(max_timepoint):
            for channel in channels:
                for z in range(max_z):
                    filenames.append(
                        f"{self.experiment_name}_{timepoint:06d}_{channel}_{z:03d}"
                    )
        return experiment_metadata, filenames

    def _read_filename(self, filename: str) -> dict:
        """
        Filenames are descriptive of the timepoint and channels.
        For example, in MAYBE_training_00/Hog1_GFP_001/MAYBE_training_00_000000_GFP_Z_000.tif
        We can see the order:
            Experiment name: MAYBE_training_00
            Position name: Hog1_GFP_001
            Filename is made up of:
                Experiment name: MAYBE_training_00
                Timepoint: 000000
                Channel: GFP_Z
                Z slice: 000

        Parameters
        ----------
        filename : str
            Filename of the image without the suffix
        """
        # Remove the experiment name
        filename = filename.replace(self.experiment_name + "_", "")
        # Use a single regex to capture timepoint, channel, and z slice
        match = re.search(
            r"(?P<timepoint>\d{6})_(?P<channel>[A-Za-z_]+)_(?P<z>\d{3})$", filename
        )
        if match:
            result = match.groupdict()
            result["timepoint"] = int(result["timepoint"])
            result["z"] = int(result["z"])
            return result
        else:
            raise ValueError("Filename does not match the expected pattern")

    @property
    def name(self):
        return self._name

    @property
    def dimorder(self):
        return "TCZYX"

    def get_data_lazy(self):
        """
        Lazy loading of the data using dask to create 5d array in TZCYX order.
        """
        lazy_arrays = [
            da.from_delayed(
                dask.delayed(imread)(
                    self.image_folder / (filename + f".{self.suffix}")
                ),
                shape=(self.metadata["size_y"], self.metadata["size_x"]),
                dtype=self.metadata["dtype"],
            )
            for filename in self.filenames
        ]
        # Stack the lazy arrays
        lazy_data = da.stack(lazy_arrays)
        # Reshape to TZCYX
        lazy_data = lazy_data.reshape(
            (
                self.metadata["size_t"],
                self.metadata["size_c"],
                self.metadata["size_z"],
                self.metadata["size_y"],
                self.metadata["size_x"],
            )
        )
        return lazy_data
