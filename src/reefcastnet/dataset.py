from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr


def years_mask(times: pd.DatetimeIndex, y0: int, y1: int) -> np.ndarray:
    years = times.year.to_numpy()
    return (years >= y0) & (years <= y1)


class SSTWeeklyForecastDataset(Dataset):
    """Weekly SSTA forecasting dataset.

    Returns tensors for ReefCastNet training.
    Dynamic tensors use [T, C, H, W]; static and masks use [C, H, W].
    """
    def __init__(
        self,
        zarr_path: str | Path,
        split: str,
        cfg: Dict[str, Any],
        patch_size: Optional[tuple[int, int]] = None,
        random_crop: bool = True,
    ) -> None:
        self.zarr_path = Path(zarr_path)
        self.split = split
        self.cfg = cfg
        self.input_len = int(cfg["forecast"]["input_len"])
        self.output_len = int(cfg["forecast"]["output_len"])
        self.random_crop = random_crop and split == "train"
        ps = patch_size if patch_size is not None else cfg["data"].get("patch_size")
        self.patch_size = None if ps is None else (int(ps[0]), int(ps[1]))

        self.ds = xr.open_zarr(self.zarr_path)
        self.times = pd.DatetimeIndex(self.ds.time.values)
        y0, y1 = cfg["data"][f"{split}_years"]
        mask = years_mask(self.times, int(y0), int(y1))
        all_starts = np.arange(0, len(self.times) - self.input_len - self.output_len + 1)
        target_end = all_starts + self.input_len + self.output_len - 1
        self.indices = all_starts[mask[target_end]]
        if len(self.indices) == 0:
            raise ValueError(f"No samples for split {split}. Check years and data length.")

        self.static_names = cfg["model"].get("static_channels", ["mmm", "reef_mask", "reef_buffer", "lat", "lon"])
        static_map = {
            "mmm": self.ds["mmm"].values.astype(np.float32),
            "reef_mask": self.ds["reef_mask"].values.astype(np.float32),
            "reef_buffer": self.ds["reef_buffer"].values.astype(np.float32),
            "lat": self.ds["lat_grid"].values.astype(np.float32),
            "lon": self.ds["lon_grid"].values.astype(np.float32),
            "ocean_mask": self.ds["ocean_mask"].values.astype(np.float32),
        }
        self.static_full = np.stack([static_map[name] for name in self.static_names], axis=0).astype(np.float32)
        self.ocean_full = static_map["ocean_mask"][None]
        self.reef_full = static_map["reef_mask"][None]
        self.buffer_full = static_map["reef_buffer"][None]
        self.mmm_full = static_map["mmm"][None]
        self.H, self.W = self.ocean_full.shape[-2:]

    def __len__(self) -> int:
        return int(len(self.indices))

    def _crop_slices(self) -> tuple[slice, slice]:
        if self.patch_size is None:
            return slice(None), slice(None)
        ph, pw = self.patch_size
        ph = min(ph, self.H)
        pw = min(pw, self.W)
        if self.random_crop:
            top = np.random.randint(0, self.H - ph + 1) if self.H > ph else 0
            left = np.random.randint(0, self.W - pw + 1) if self.W > pw else 0
        else:
            top = max((self.H - ph) // 2, 0)
            left = max((self.W - pw) // 2, 0)
        return slice(top, top + ph), slice(left, left + pw)

    @staticmethod
    def _time_features(weeks: np.ndarray) -> np.ndarray:
        theta = 2 * np.pi * weeks.astype(np.float32) / 52.0
        return np.stack([np.sin(theta), np.cos(theta)], axis=-1).astype(np.float32)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        start = int(self.indices[item])
        end = start + self.input_len + self.output_len
        ys, xs = self._crop_slices()
        ssta = self.ds["ssta"].isel(time=slice(start, end), lat=ys, lon=xs).values.astype(np.float32)
        sst = self.ds["sst"].isel(time=slice(start, end), lat=ys, lon=xs).values.astype(np.float32)
        clim = self.ds["climatology"].isel(time=slice(start, end), lat=ys, lon=xs).values.astype(np.float32)
        weeks = self.ds["week_of_year"].isel(time=slice(start, end)).values.astype(np.int16)

        x_ssta = ssta[: self.input_len, None]
        y_ssta = ssta[self.input_len :, None]
        past_sst = sst[: self.input_len, None]
        future_sst = sst[self.input_len :, None]
        future_clim = clim[self.input_len :, None]

        static = self.static_full[:, ys, xs]
        ocean = self.ocean_full[:, ys, xs]
        reef = self.reef_full[:, ys, xs]
        buffer = self.buffer_full[:, ys, xs]
        mmm = self.mmm_full[:, ys, xs]

        time_feats = self._time_features(weeks[: self.input_len])
        future_time_feats = self._time_features(weeks[self.input_len :])
        return {
            "x_ssta": torch.from_numpy(x_ssta),
            "y_ssta": torch.from_numpy(y_ssta),
            "past_sst": torch.from_numpy(past_sst),
            "future_sst": torch.from_numpy(future_sst),
            "future_clim": torch.from_numpy(future_clim),
            "static": torch.from_numpy(static),
            "ocean_mask": torch.from_numpy(ocean),
            "reef_mask": torch.from_numpy(reef),
            "reef_buffer": torch.from_numpy(buffer),
            "mmm": torch.from_numpy(mmm),
            "time_feats": torch.from_numpy(time_feats),
            "future_time_feats": torch.from_numpy(future_time_feats),
            "start_index": torch.tensor(start, dtype=torch.long),
        }
