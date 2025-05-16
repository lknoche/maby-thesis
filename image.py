from pathlib import Path
import re
import dask
import dask.array as da
from skimage.io import imread
import collections


class DownloadedExperiment:
    def __init__(self, image_folder: str):
        self.image_folder = Path(image_folder)
        self.positions = self._get_positions()

    def _get_positions(self):
        return sorted([x.name for x in self.image_folder.iterdir() if x.is_dir()])


class DownloadedImageDir:
    def __init__(self, image_folder: str, suffix: str = "tiff"):
        self.image_folder = Path(image_folder)
        self.experiment_name = self.image_folder.name
        self._name = self.image_folder.name if self.image_folder.is_dir() else self.image_folder.parent.name
        self.suffix = suffix
        self.metadata, self.filenames = self._read_metadata()

    def _read_filename(self, filename: str) -> dict:
        filename = filename.replace(self.experiment_name + "_", "")
        match = re.search(r"(?P<timepoint>\d{6})_(?P<channel>[A-Za-z_]+)_(?P<z>\d{3})$", filename)
        if match:
            result = match.groupdict()
            result["timepoint"] = int(result["timepoint"])
            result["z"] = int(result["z"])
            return result
        else:
            raise ValueError("Filename does not match the expected pattern")

    def _read_metadata(self):
        files = sorted([x.stem for x in self.image_folder.iterdir() if x.suffix == ".tiff"])
        file_metadata = [self._read_filename(file) for file in files]

        grouped = collections.defaultdict(lambda: collections.defaultdict(set))  # {channel: {tp: set(z)}}
        for meta in file_metadata:
            grouped[meta["channel"]][meta["timepoint"]].add(meta["z"])

        expected_z = {
            channel: max(len(zs) for zs in tps.values())
            for channel, tps in grouped.items()
        }

        valid_timepoints = set.intersection(*[
            {tp for tp, zs in tps.items() if len(zs) == expected_z[channel]}
            for channel, tps in grouped.items()
        ])

        if not valid_timepoints:
            raise ValueError("No valid timepoints with full z-stacks across all channels.")

        channels = sorted(grouped.keys())
        max_z = max(expected_z.values())
        max_timepoint = max(valid_timepoints)

        first_file = next((f for f in self.image_folder.iterdir() if f.suffix == ".tiff"), None)
        img = imread(first_file)
        size_y, size_x = img.shape
        dtype = img.dtype

        meta = {
            "channels": channels,
            "size_t": max_timepoint + 1,
            "size_c": len(channels),
            "size_z": max_z,
            "size_y": size_y,
            "size_x": size_x,
            "dtype": dtype,
        }

        filenames = [
            f"{self.experiment_name}_{t:06d}_{c}_{z:03d}"
            for t in sorted(valid_timepoints)
            for c in channels
            for z in range(expected_z[c])
        ]

        missing_tps = sorted(set(range(max_timepoint + 1)) - valid_timepoints)
        if missing_tps:
            print(f"\u26a0\ufe0f Skipped {len(missing_tps)} incomplete timepoints: {missing_tps[:5]}{'...' if len(missing_tps) > 5 else ''}")

        return meta, filenames

    @property
    def name(self):
        return self._name

    @property
    def dimorder(self):
        return "TCZYX"

    def get_data_lazy(self):
        lazy_arrays = []
        tcz_index = []

        channels = self.metadata["channels"]
        z_slices = self.metadata["size_z"]

        for filename in self.filenames:
            path = self.image_folder / f"{filename}.{self.suffix}"
            match = re.search(r"_(\d{6})_([A-Za-z_]+)_(\d{3})$", filename)
            if not match:
                continue
            t, c, z = match.groups()
            t, z = int(t), int(z)
            if c not in channels:
                continue
            lazy_arrays.append(
                da.from_delayed(
                    dask.delayed(imread)(path),
                    shape=(self.metadata["size_y"], self.metadata["size_x"]),
                    dtype=self.metadata["dtype"],
                )
            )
            tcz_index.append((t, c, z))

        tcz_sorted = sorted(zip(tcz_index, lazy_arrays), key=lambda x: (x[0][0], channels.index(x[0][1]), x[0][2]))
        tcz_index_sorted, lazy_arrays = zip(*tcz_sorted)

        timepoints = sorted(set(t for t, _, _ in tcz_index_sorted))
        T, C, Z = len(timepoints), len(channels), z_slices

        expected_size = T * C * Z
        if len(lazy_arrays) != expected_size:
            raise ValueError(f"Mismatch: expected {expected_size} images, got {len(lazy_arrays)}. Cannot reshape.")

        lazy_data = da.stack(lazy_arrays)
        lazy_data = lazy_data.reshape((T, C, Z, self.metadata["size_y"], self.metadata["size_x"]))
        return lazy_data

    @property
    def data(self):
        return self.get_data_lazy()
