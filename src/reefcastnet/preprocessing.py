from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation


def _find_name(names: Iterable[str], candidates: Sequence[str]) -> str:
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"Could not find any of {candidates} in {list(names)}")


def standardize_da(ds: xr.Dataset, var: str) -> xr.DataArray:
    if var not in ds:
        if len(ds.data_vars) == 1:
            var = list(ds.data_vars)[0]
        else:
            raise KeyError(f"Variable {var!r} not in dataset. Found {list(ds.data_vars)}")
    da = ds[var]
    lat_name = _find_name(da.dims, ["lat", "latitude", "y"])
    lon_name = _find_name(da.dims, ["lon", "longitude", "x"])
    time_name = _find_name(da.dims, ["time"])
    ren = {}
    if lat_name != "lat":
        ren[lat_name] = "lat"
    if lon_name != "lon":
        ren[lon_name] = "lon"
    if time_name != "time":
        ren[time_name] = "time"
    da = da.rename(ren).transpose("time", "lat", "lon")
    return da


def open_year_files(paths: Sequence[str | Path], var: str) -> xr.DataArray:
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("No NetCDF files found.")
    arrays = []
    for p in paths:
        ds = xr.open_dataset(p, decode_times=True)
        arrays.append(standardize_da(ds, var))
    da = xr.concat(arrays, dim="time").sortby("time")
    _, unique_idx = np.unique(pd.DatetimeIndex(da.time.values), return_index=True)
    da = da.isel(time=np.sort(unique_idx))
    sample = float(da.isel(time=slice(0, min(10, da.sizes["time"]))).mean(skipna=True).values)
    if sample > 100:
        da = da - 273.15
    return da.astype("float32")


def make_reef_mask_from_boxes(lat: np.ndarray, lon: np.ndarray, reef_boxes: Sequence[Sequence[Any]]) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for box in reef_boxes:
        if len(box) != 5:
            raise ValueError(f"reef_boxes entry must be [name, lat_min, lat_max, lon_min, lon_max], got {box}")
        _, la0, la1, lo0, lo1 = box
        mask |= (lat2d >= float(la0)) & (lat2d <= float(la1)) & (lon2d >= float(lo0)) & (lon2d <= float(lo1))
    return mask


def weekly_climatology(
    sst_weekly: np.ndarray,
    times: pd.DatetimeIndex,
    train_years: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SSTA, time-matched climatology, week climatology, and MMM.

    Week 53 is merged into week 52 to prevent sparse bins.
    """
    years = times.year.to_numpy()
    weeks = times.isocalendar().week.to_numpy().astype(np.int16)
    weeks = np.minimum(weeks, 52)
    train_mask = (years >= train_years[0]) & (years <= train_years[1])
    if train_mask.sum() < 20:
        raise ValueError("Too few training weeks for climatology.")
    H, W = sst_weekly.shape[1], sst_weekly.shape[2]
    clim_by_week = np.full((53, H, W), np.nan, dtype=np.float32)
    for w in range(1, 53):
        idx = train_mask & (weeks == w)
        if idx.any():
            clim_by_week[w] = np.nanmean(sst_weekly[idx], axis=0)
    train_mean = np.nanmean(sst_weekly[train_mask], axis=0).astype(np.float32)
    for w in range(1, 53):
        clim_by_week[w] = np.where(np.isfinite(clim_by_week[w]), clim_by_week[w], train_mean)
    mmm = np.nanmax(clim_by_week[1:53], axis=0).astype(np.float32)
    clim_for_time = clim_by_week[weeks]
    ssta = (sst_weekly - clim_for_time).astype(np.float32)
    return ssta, clim_for_time.astype(np.float32), clim_by_week.astype(np.float32), mmm


def write_zarr_dataset(ds: xr.Dataset, out_path: Path, chunks: dict | None = None) -> None:
    """Write zarr with dask chunks when dask is installed; otherwise write directly."""
    if chunks is not None:
        try:
            import dask  # noqa: F401
            ds = ds.chunk(chunks)
        except Exception:
            pass
    ds.to_zarr(out_path, mode="w")


def build_weekly_dataset(cfg: Dict[str, Any], files: Sequence[str | Path], out_path: str | Path) -> Path:
    data_cfg = cfg["data"]
    da = open_year_files(files, cfg["erddap"]["sst_var"])
    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.sortby("lat")
    weekly = da.resample(time=data_cfg.get("weekly_freq", "7D")).mean(skipna=True).astype("float32")
    times = pd.DatetimeIndex(weekly.time.values)
    lat = weekly.lat.values.astype(np.float32)
    lon = weekly.lon.values.astype(np.float32)
    sst = weekly.values.astype(np.float32)

    finite_ratio = np.isfinite(sst).mean(axis=0)
    ocean_mask = finite_ratio > 0.25

    train_years = tuple(map(int, data_cfg["train_years"]))
    ssta, clim_for_time, clim_by_week, mmm = weekly_climatology(sst, times, train_years)

    if data_cfg.get("fill_nan", "zero_anomaly") == "zero_anomaly":
        ssta = np.where(np.isfinite(ssta), ssta, 0.0).astype(np.float32)
        sst = np.where(np.isfinite(sst), sst, clim_for_time).astype(np.float32)
    else:
        ssta = np.where(np.isfinite(ssta), ssta, 0.0).astype(np.float32)
        sst = np.where(np.isfinite(sst), sst, clim_for_time).astype(np.float32)

    reef_mask = make_reef_mask_from_boxes(lat, lon, data_cfg.get("reef_boxes", []))
    reef_mask &= ocean_mask
    reef_buffer_cells = int(data_cfg.get("reef_buffer_cells", 4))
    reef_buffer = binary_dilation(reef_mask, iterations=reef_buffer_cells) & ocean_mask & (~reef_mask)

    lat_norm = ((lat - lat.min()) / max(lat.max() - lat.min(), 1e-6) * 2 - 1).astype(np.float32)
    lon_norm = ((lon - lon.min()) / max(lon.max() - lon.min(), 1e-6) * 2 - 1).astype(np.float32)
    lat_grid, lon_grid = np.meshgrid(lat_norm, lon_norm, indexing="ij")
    weeks = np.minimum(times.isocalendar().week.to_numpy().astype(np.int16), 52)

    ds = xr.Dataset(
        data_vars=dict(
            sst=(("time", "lat", "lon"), sst),
            ssta=(("time", "lat", "lon"), ssta),
            climatology=(("time", "lat", "lon"), clim_for_time.astype(np.float32)),
            clim_by_week=(("week", "lat", "lon"), clim_by_week.astype(np.float32)),
            mmm=(("lat", "lon"), mmm.astype(np.float32)),
            ocean_mask=(("lat", "lon"), ocean_mask.astype(np.float32)),
            reef_mask=(("lat", "lon"), reef_mask.astype(np.float32)),
            reef_buffer=(("lat", "lon"), reef_buffer.astype(np.float32)),
            lat_grid=(("lat", "lon"), lat_grid.astype(np.float32)),
            lon_grid=(("lat", "lon"), lon_grid.astype(np.float32)),
            week_of_year=(("time",), weeks),
        ),
        coords=dict(time=times, lat=lat, lon=lon, week=np.arange(53, dtype=np.int16)),
        attrs=dict(description="Weekly NOAA CRW CoralTemp SST dataset for ReefCastNet"),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    write_zarr_dataset(ds, out_path, {"time": 64, "lat": min(128, len(lat)), "lon": min(128, len(lon))})

    meta = {
        "num_weeks": int(len(times)),
        "lat_size": int(len(lat)),
        "lon_size": int(len(lon)),
        "train_years": list(train_years),
        "reef_cells": int(reef_mask.sum()),
        "reef_buffer_cells": int(reef_buffer.sum()),
        "ocean_cells": int(ocean_mask.sum()),
    }
    with open(out_path.parent / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_path


def make_toy_zarr(out_path: str | Path, weeks: int = 160, h: int = 32, w: int = 48, seed: int = 42) -> Path:
    """Create a synthetic SST-like dataset for smoke tests without downloading NOAA data."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2018-01-01", periods=weeks, freq="7D")
    lat = np.linspace(0, 25, h, dtype=np.float32)
    lon = np.linspace(100, 125, w, dtype=np.float32)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    phase = np.arange(weeks)[:, None, None] / 52.0 * 2 * np.pi
    base = 27.0 + 2.2 * np.sin(phase) + 0.03 * (lat2d[None] - 12.5)
    mode = 0.6 * np.sin(phase * 0.5 + lon2d[None] / 8.0) + 0.4 * np.cos(lat2d[None] / 5.0)
    noise = rng.normal(0, 0.12, size=(weeks, h, w)).astype(np.float32)
    sst = (base + mode + noise).astype(np.float32)
    sst[100:118, 10:20, 20:32] += 1.8

    reef_boxes = [["toy_reef", 7.0, 14.0, 110.0, 117.0]]
    ssta, clim_for_time, clim_by_week, mmm = weekly_climatology(sst, times, (2018, 2019))
    reef_mask = make_reef_mask_from_boxes(lat, lon, reef_boxes)
    ocean_mask = np.ones((h, w), dtype=bool)
    reef_buffer = binary_dilation(reef_mask, iterations=3) & (~reef_mask)
    lat_norm = ((lat - lat.min()) / (lat.max() - lat.min()) * 2 - 1).astype(np.float32)
    lon_norm = ((lon - lon.min()) / (lon.max() - lon.min()) * 2 - 1).astype(np.float32)
    lat_grid, lon_grid = np.meshgrid(lat_norm, lon_norm, indexing="ij")
    weeks_iso = np.minimum(times.isocalendar().week.to_numpy().astype(np.int16), 52)
    ds = xr.Dataset(
        data_vars=dict(
            sst=(("time", "lat", "lon"), sst),
            ssta=(("time", "lat", "lon"), ssta.astype(np.float32)),
            climatology=(("time", "lat", "lon"), clim_for_time.astype(np.float32)),
            clim_by_week=(("week", "lat", "lon"), clim_by_week.astype(np.float32)),
            mmm=(("lat", "lon"), mmm.astype(np.float32)),
            ocean_mask=(("lat", "lon"), ocean_mask.astype(np.float32)),
            reef_mask=(("lat", "lon"), reef_mask.astype(np.float32)),
            reef_buffer=(("lat", "lon"), reef_buffer.astype(np.float32)),
            lat_grid=(("lat", "lon"), lat_grid.astype(np.float32)),
            lon_grid=(("lat", "lon"), lon_grid.astype(np.float32)),
            week_of_year=(("time",), weeks_iso),
        ),
        coords=dict(time=times, lat=lat, lon=lon, week=np.arange(53, dtype=np.int16)),
    )
    out_path = Path(out_path)
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_zarr_dataset(ds, out_path, {"time": 64, "lat": h, "lon": w})
    return out_path
