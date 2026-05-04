#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import json
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import njit
from numcodecs import Blosc


DEFAULT_INPUT = "/data1/user/lz/osita_data/scs_5s25n/ostia_scs_daily.zarr"
DEFAULT_OUTPUT_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis"

_WORKER_DS: xr.Dataset | None = None
_WORKER_OCEAN_MASK: np.ndarray | None = None
_WORKER_BASELINE_INDICES: np.ndarray | None = None
_WORKER_WINDOW_INDICES_LOCAL: list[np.ndarray] | None = None
_WORKER_DOY_IDX: np.ndarray | None = None
_WORKER_YEAR_IDX: np.ndarray | None = None
_WORKER_BREAK_BEFORE: np.ndarray | None = None
_WORKER_N_YEARS: int | None = None
_WORKER_MIN_DURATION: int | None = None
_WORKER_PERCENTILE: float | None = None


def _log(message: str) -> None:
    print(message, flush=True)


def _maybe_remove(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _compressor() -> Blosc:
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def _write_zarr(ds: xr.Dataset, path: Path, chunks: dict[str, int], overwrite: bool) -> None:
    _maybe_remove(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.chunk({k: v for k, v in chunks.items() if k in ds.dims})
    encoding: dict[str, dict[str, Any]] = {}
    compressor = _compressor()
    for name, da in ds.data_vars.items():
        var_chunks = tuple(chunks[d] for d in da.dims if d in chunks)
        enc: dict[str, Any] = {"compressor": compressor}
        if var_chunks:
            enc["chunks"] = var_chunks
        encoding[name] = enc
    _log(f"[write] {path}")
    with ProgressBar():
        ds.to_zarr(path, mode="w", consolidated=True, encoding=encoding)


def _doy366(times: pd.DatetimeIndex) -> np.ndarray:
    # 使用闰年作为日序参考，保证 3 月至 12 月在闰年和非闰年中的日序一致。
    # 2 月 29 日保留为第 60 天，避免阈值计算时季节窗口错位。
    return np.array([pd.Timestamp(f"2000-{ts:%m-%d}").dayofyear for ts in times], dtype=np.int16)


def _window_indices_for_doy(base_doy: np.ndarray, window_days: int) -> list[np.ndarray]:
    if window_days < 1 or window_days % 2 != 1:
        raise ValueError("--window-days must be a positive odd integer.")
    half = window_days // 2
    out = []
    for doy in range(1, 367):
        window = ((np.arange(doy - half, doy + half + 1) - 1) % 366) + 1
        out.append(np.flatnonzero(np.isin(base_doy, window)))
    return out


def _compute_climatology_threshold_block(
    baseline_sst: np.ndarray,
    window_indices: list[np.ndarray],
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    doy_len = 366
    _, height, width = baseline_sst.shape
    climatology = np.full((doy_len, height, width), np.nan, dtype=np.float32)
    threshold = np.full((doy_len, height, width), np.nan, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, idx in enumerate(window_indices):
            vals = baseline_sst[idx]
            climatology[i] = np.nanmean(vals, axis=0).astype(np.float32)
            threshold[i] = np.nanpercentile(vals, percentile, axis=0).astype(np.float32)
    return climatology, threshold


@njit
def _annual_mhw_metrics_numba(
    sst: np.ndarray,
    climatology: np.ndarray,
    threshold: np.ndarray,
    doy_idx: np.ndarray,
    year_idx: np.ndarray,
    break_before: np.ndarray,
    ocean: np.ndarray,
    n_years: int,
    min_duration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_len, n_pix = sst.shape
    frequency = np.empty((n_years, n_pix), dtype=np.float32)
    total_days = np.empty((n_years, n_pix), dtype=np.float32)
    mean_duration = np.empty((n_years, n_pix), dtype=np.float32)
    max_duration = np.empty((n_years, n_pix), dtype=np.float32)
    max_intensity = np.empty((n_years, n_pix), dtype=np.float32)
    cumulative_intensity = np.empty((n_years, n_pix), dtype=np.float32)
    valid_days = np.zeros((n_years, n_pix), dtype=np.float32)

    for y in range(n_years):
        for n in range(n_pix):
            frequency[y, n] = np.nan
            total_days[y, n] = np.nan
            mean_duration[y, n] = np.nan
            max_duration[y, n] = np.nan
            max_intensity[y, n] = np.nan
            cumulative_intensity[y, n] = np.nan

    for n in range(n_pix):
        if not ocean[n]:
            continue

        for y in range(n_years):
            frequency[y, n] = 0.0
            total_days[y, n] = 0.0
            mean_duration[y, n] = np.nan
            max_duration[y, n] = 0.0
            max_intensity[y, n] = np.nan
            cumulative_intensity[y, n] = 0.0

        run_len = 0
        run_year = -1
        run_cum = 0.0
        run_max = -1.0e20

        for t in range(t_len):
            y = year_idx[t]

            if run_len > 0 and (break_before[t] or y != run_year):
                if run_len >= min_duration:
                    frequency[run_year, n] += 1.0
                    total_days[run_year, n] += float(run_len)
                    cumulative_intensity[run_year, n] += run_cum
                    if float(run_len) > max_duration[run_year, n]:
                        max_duration[run_year, n] = float(run_len)
                    if np.isnan(max_intensity[run_year, n]) or run_max > max_intensity[run_year, n]:
                        max_intensity[run_year, n] = run_max
                run_len = 0
                run_year = -1
                run_cum = 0.0
                run_max = -1.0e20

            v = sst[t, n]
            d = doy_idx[t]
            clim = climatology[d, n]
            p90 = threshold[d, n]
            is_valid = not np.isnan(v) and not np.isnan(clim) and not np.isnan(p90)

            if is_valid:
                valid_days[y, n] += 1.0

            if is_valid and v > p90:
                intensity = v - clim
                if run_len == 0:
                    run_year = y
                    run_cum = 0.0
                    run_max = -1.0e20
                run_len += 1
                run_cum += intensity
                if intensity > run_max:
                    run_max = intensity
            else:
                if run_len >= min_duration:
                    frequency[run_year, n] += 1.0
                    total_days[run_year, n] += float(run_len)
                    cumulative_intensity[run_year, n] += run_cum
                    if float(run_len) > max_duration[run_year, n]:
                        max_duration[run_year, n] = float(run_len)
                    if np.isnan(max_intensity[run_year, n]) or run_max > max_intensity[run_year, n]:
                        max_intensity[run_year, n] = run_max
                run_len = 0
                run_year = -1
                run_cum = 0.0
                run_max = -1.0e20

        if run_len >= min_duration:
            frequency[run_year, n] += 1.0
            total_days[run_year, n] += float(run_len)
            cumulative_intensity[run_year, n] += run_cum
            if float(run_len) > max_duration[run_year, n]:
                max_duration[run_year, n] = float(run_len)
            if np.isnan(max_intensity[run_year, n]) or run_max > max_intensity[run_year, n]:
                max_intensity[run_year, n] = run_max

        for y in range(n_years):
            if frequency[y, n] > 0.0:
                mean_duration[y, n] = total_days[y, n] / frequency[y, n]

    return frequency, total_days, mean_duration, max_duration, max_intensity, cumulative_intensity, valid_days


def _warm_numba_metrics(min_duration: int) -> None:
    sst = np.full((2, 1), np.nan, dtype=np.float32)
    climatology = np.full((366, 1), np.nan, dtype=np.float32)
    threshold = np.full((366, 1), np.nan, dtype=np.float32)
    doy_idx = np.array([0, 1], dtype=np.int16)
    year_idx = np.array([0, 0], dtype=np.int16)
    break_before = np.array([False, False], dtype=np.bool_)
    ocean = np.array([True], dtype=np.bool_)
    _annual_mhw_metrics_numba(
        sst,
        climatology,
        threshold,
        doy_idx,
        year_idx,
        break_before,
        ocean,
        1,
        int(min_duration),
    )


def _compute_lat_block_arrays(
    sst_block: np.ndarray,
    ocean_block: np.ndarray,
    baseline_indices: np.ndarray,
    window_indices_local: list[np.ndarray],
    percentile: float,
    doy_idx: np.ndarray,
    year_idx: np.ndarray,
    break_before: np.ndarray,
    n_years: int,
    min_duration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    doy_len = 366
    lon_len = sst_block.shape[2]
    baseline_block = sst_block[baseline_indices]
    climatology_block, threshold_block = _compute_climatology_threshold_block(
        baseline_block,
        window_indices_local,
        percentile,
    )
    climatology_block = np.where(ocean_block[None, :, :], climatology_block, np.nan).astype(np.float32)
    threshold_block = np.where(ocean_block[None, :, :], threshold_block, np.nan).astype(np.float32)

    flat_sst = sst_block.reshape((sst_block.shape[0], -1))
    flat_clim = climatology_block.reshape((doy_len, -1))
    flat_threshold = threshold_block.reshape((doy_len, -1))
    flat_ocean = ocean_block.reshape(-1)
    daily_valid_ocean_counts = np.zeros(sst_block.shape[0], dtype=np.int64)
    if flat_ocean.any():
        daily_valid_ocean_counts += np.isfinite(flat_sst[:, flat_ocean]).sum(axis=1).astype(np.int64)

    metrics = _annual_mhw_metrics_numba(
        flat_sst,
        flat_clim,
        flat_threshold,
        doy_idx,
        year_idx,
        break_before,
        flat_ocean,
        n_years,
        int(min_duration),
    )
    (
        frequency,
        total_days,
        mean_duration,
        max_duration,
        max_intensity,
        cumulative_intensity,
        valid_days,
    ) = [arr.reshape((n_years, sst_block.shape[1], lon_len)) for arr in metrics]

    return (
        climatology_block,
        threshold_block,
        frequency,
        total_days,
        mean_duration,
        max_duration,
        max_intensity,
        cumulative_intensity,
        valid_days,
        daily_valid_ocean_counts,
    )


def _init_lat_block_worker(
    input_zarr: str,
    baseline_indices: np.ndarray,
    window_indices_local: list[np.ndarray],
    doy_idx: np.ndarray,
    year_idx: np.ndarray,
    break_before: np.ndarray,
    n_years: int,
    min_duration: int,
    percentile: float,
) -> None:
    global _WORKER_DS
    global _WORKER_OCEAN_MASK
    global _WORKER_BASELINE_INDICES
    global _WORKER_WINDOW_INDICES_LOCAL
    global _WORKER_DOY_IDX
    global _WORKER_YEAR_IDX
    global _WORKER_BREAK_BEFORE
    global _WORKER_N_YEARS
    global _WORKER_MIN_DURATION
    global _WORKER_PERCENTILE

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    _WORKER_DS = xr.open_zarr(input_zarr, consolidated=True)
    _WORKER_OCEAN_MASK = _WORKER_DS["ocean_mask"].values.astype(bool)
    _WORKER_BASELINE_INDICES = baseline_indices
    _WORKER_WINDOW_INDICES_LOCAL = window_indices_local
    _WORKER_DOY_IDX = doy_idx
    _WORKER_YEAR_IDX = year_idx
    _WORKER_BREAK_BEFORE = break_before
    _WORKER_N_YEARS = int(n_years)
    _WORKER_MIN_DURATION = int(min_duration)
    _WORKER_PERCENTILE = float(percentile)


def _compute_lat_block_worker(
    lat_range: tuple[int, int],
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        _WORKER_DS is None
        or _WORKER_OCEAN_MASK is None
        or _WORKER_BASELINE_INDICES is None
        or _WORKER_WINDOW_INDICES_LOCAL is None
        or _WORKER_DOY_IDX is None
        or _WORKER_YEAR_IDX is None
        or _WORKER_BREAK_BEFORE is None
        or _WORKER_N_YEARS is None
        or _WORKER_MIN_DURATION is None
        or _WORKER_PERCENTILE is None
    ):
        raise RuntimeError("Latitude-block worker was not initialized.")

    lat0, lat1 = lat_range
    sst_block = _WORKER_DS["sst"].isel(lat=slice(lat0, lat1)).values.astype(np.float32)
    ocean_block = _WORKER_OCEAN_MASK[lat0:lat1]
    block_outputs = _compute_lat_block_arrays(
        sst_block,
        ocean_block,
        _WORKER_BASELINE_INDICES,
        _WORKER_WINDOW_INDICES_LOCAL,
        _WORKER_PERCENTILE,
        _WORKER_DOY_IDX,
        _WORKER_YEAR_IDX,
        _WORKER_BREAK_BEFORE,
        _WORKER_N_YEARS,
        _WORKER_MIN_DURATION,
    )
    return (lat0, lat1, *block_outputs)


def _lat_ranges(lat_len: int, lat_block_size: int) -> list[tuple[int, int]]:
    if lat_block_size < 1:
        raise ValueError("--lat-block-size must be positive.")
    return [(lat0, min(lat0 + lat_block_size, lat_len)) for lat0 in range(0, lat_len, lat_block_size)]


def _resolve_worker_count(requested_workers: int, n_blocks: int) -> int:
    if requested_workers < 0:
        raise ValueError("--workers must be non-negative.")
    if requested_workers == 0:
        requested_workers = os.cpu_count() or 1
    return max(1, min(int(requested_workers), int(n_blocks)))


def _expected_day_counts(years: np.ndarray) -> np.ndarray:
    return np.array([366 if pd.Timestamp(int(y), 12, 31).dayofyear == 366 else 365 for y in years], dtype=np.float32)


def _weighted_annual_means(metrics: xr.Dataset) -> pd.DataFrame:
    weights = np.cos(np.deg2rad(metrics["lat"])).astype("float32")
    weights = weights.broadcast_like(metrics["ocean_mask"]).where(metrics["ocean_mask"] == 1, 0.0)
    rows = {"year": metrics["year"].values.astype(int)}
    for name in [
        "mhw_frequency",
        "mhw_total_days",
        "mhw_mean_duration",
        "mhw_max_duration",
        "mhw_max_intensity",
        "mhw_cumulative_intensity",
        "valid_days",
    ]:
        rows[name] = metrics[name].weighted(weights).mean(("lat", "lon"), skipna=True).values.astype(float)
    return pd.DataFrame(rows)


def build_products(args: argparse.Namespace) -> None:
    input_zarr = Path(args.input_zarr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(input_zarr, consolidated=True)
    if "sst" not in ds:
        raise KeyError("Input daily Zarr must contain sst.")
    if "ocean_mask" not in ds:
        raise KeyError("Input daily Zarr must contain ocean_mask.")

    times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    lat = ds["lat"].values.astype(np.float32)
    lon = ds["lon"].values.astype(np.float32)
    years = np.arange(times.year.min(), times.year.max() + 1, dtype=np.int16)
    year_lookup = {int(y): i for i, y in enumerate(years)}
    year_idx = np.array([year_lookup[int(y)] for y in times.year], dtype=np.int16)
    doy_idx = (_doy366(times) - 1).astype(np.int16)

    day_deltas = np.diff(times.values).astype("timedelta64[D]").astype(np.int64)
    break_before = np.zeros(len(times), dtype=np.bool_)
    break_before[1:] = day_deltas != 1

    expected = pd.date_range(times.min(), times.max(), freq="D")
    missing_calendar_days = expected.difference(times)
    missing_month_counts = (
        missing_calendar_days.to_period("M").value_counts().sort_index().astype(int).to_dict()
        if len(missing_calendar_days)
        else {}
    )

    baseline_mask = (times >= pd.Timestamp(args.baseline_start)) & (times <= pd.Timestamp(args.baseline_end))
    if baseline_mask.sum() < 365 * 10:
        raise ValueError("Too few baseline daily samples for climatology.")
    baseline_indices = np.flatnonzero(baseline_mask)
    base_doy = doy_idx[baseline_indices] + 1
    window_indices_local = _window_indices_for_doy(base_doy, args.window_days)

    lat_len = len(lat)
    lon_len = len(lon)
    n_years = len(years)
    doy_len = 366

    climatology_all = np.full((doy_len, lat_len, lon_len), np.nan, dtype=np.float32)
    threshold_all = np.full((doy_len, lat_len, lon_len), np.nan, dtype=np.float32)
    frequency_all = np.full((n_years, lat_len, lon_len), np.nan, dtype=np.float32)
    total_days_all = np.full_like(frequency_all, np.nan)
    mean_duration_all = np.full_like(frequency_all, np.nan)
    max_duration_all = np.full_like(frequency_all, np.nan)
    max_intensity_all = np.full_like(frequency_all, np.nan)
    cumulative_intensity_all = np.full_like(frequency_all, np.nan)
    valid_days_all = np.full_like(frequency_all, np.nan)
    ocean_mask = ds["ocean_mask"].values.astype(bool)
    daily_valid_ocean_counts = np.zeros(len(times), dtype=np.int64)

    lat_block_size = int(args.lat_block_size)
    blocks = _lat_ranges(lat_len, lat_block_size)
    workers = _resolve_worker_count(int(args.workers), len(blocks))
    _log(
        f"[compute] lat_blocks={len(blocks)}, lat_block_size={lat_block_size}, "
        f"workers={workers}, percentile={args.percentile}"
    )
    _warm_numba_metrics(int(args.min_duration))

    def store_block(
        lat0: int,
        lat1: int,
        climatology_block: np.ndarray,
        threshold_block: np.ndarray,
        frequency: np.ndarray,
        total_days: np.ndarray,
        mean_duration: np.ndarray,
        max_duration: np.ndarray,
        max_intensity: np.ndarray,
        cumulative_intensity: np.ndarray,
        valid_days: np.ndarray,
        daily_valid_block: np.ndarray,
    ) -> None:
        climatology_all[:, lat0:lat1, :] = climatology_block
        threshold_all[:, lat0:lat1, :] = threshold_block
        frequency_all[:, lat0:lat1, :] = frequency
        total_days_all[:, lat0:lat1, :] = total_days
        mean_duration_all[:, lat0:lat1, :] = mean_duration
        max_duration_all[:, lat0:lat1, :] = max_duration
        max_intensity_all[:, lat0:lat1, :] = max_intensity
        cumulative_intensity_all[:, lat0:lat1, :] = cumulative_intensity
        valid_days_all[:, lat0:lat1, :] = valid_days
        daily_valid_ocean_counts[:] += daily_valid_block

    if workers == 1:
        for block_no, (lat0, lat1) in enumerate(blocks, start=1):
            _log(f"[block {block_no}/{len(blocks)}] lat {lat0}:{lat1} / {lat_len}")
            sst_block = ds["sst"].isel(lat=slice(lat0, lat1)).values.astype(np.float32)
            ocean_block = ocean_mask[lat0:lat1]
            block_outputs = _compute_lat_block_arrays(
                sst_block,
                ocean_block,
                baseline_indices,
                window_indices_local,
                args.percentile,
                doy_idx,
                year_idx,
                break_before,
                n_years,
                int(args.min_duration),
            )
            store_block(lat0, lat1, *block_outputs)
    else:
        futures = {}
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_lat_block_worker,
            initargs=(
                str(input_zarr),
                baseline_indices,
                window_indices_local,
                doy_idx,
                year_idx,
                break_before,
                n_years,
                int(args.min_duration),
                float(args.percentile),
            ),
        )
        try:
            for block in blocks:
                futures[pool.submit(_compute_lat_block_worker, block)] = block
            for done_no, future in enumerate(as_completed(futures), start=1):
                (
                    lat0,
                    lat1,
                    climatology_block,
                    threshold_block,
                    frequency,
                    total_days,
                    mean_duration,
                    max_duration,
                    max_intensity,
                    cumulative_intensity,
                    valid_days,
                    daily_valid_block,
                ) = future.result()
                store_block(
                    lat0,
                    lat1,
                    climatology_block,
                    threshold_block,
                    frequency,
                    total_days,
                    mean_duration,
                    max_duration,
                    max_intensity,
                    cumulative_intensity,
                    valid_days,
                    daily_valid_block,
                )
                _log(f"[done {done_no}/{len(blocks)}] lat {lat0}:{lat1} / {lat_len}")
        except BaseException:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    doy = np.arange(1, 367, dtype=np.int16)
    clim_ds = xr.Dataset(
        {
            "sst_climatology": (("doy", "lat", "lon"), climatology_all),
            "sst_p90_threshold": (("doy", "lat", "lon"), threshold_all),
            "ocean_mask": (("lat", "lon"), ocean_mask.astype("uint8")),
        },
        coords={"doy": doy, "lat": lat, "lon": lon},
        attrs={
            "title": "South China Sea daily SST climatology and marine heatwave threshold",
            "source_zarr": str(input_zarr),
            "baseline_period": f"{args.baseline_start} to {args.baseline_end}",
            "percentile": float(args.percentile),
            "window_days": int(args.window_days),
            "calendar": "366-day month-day climatology; Feb 29 retained",
            "method": "Centered day-of-year window over baseline years, skipping NaNs.",
        },
    )
    _write_zarr(
        clim_ds,
        output_dir / "daily_climatology_threshold.zarr",
        {"doy": 366, "lat": args.output_lat_chunk, "lon": args.output_lon_chunk},
        args.overwrite,
    )

    expected_days = _expected_day_counts(years)
    metrics_ds = xr.Dataset(
        {
            "mhw_frequency": (("year", "lat", "lon"), frequency_all),
            "mhw_total_days": (("year", "lat", "lon"), total_days_all),
            "mhw_mean_duration": (("year", "lat", "lon"), mean_duration_all),
            "mhw_max_duration": (("year", "lat", "lon"), max_duration_all),
            "mhw_max_intensity": (("year", "lat", "lon"), max_intensity_all),
            "mhw_cumulative_intensity": (("year", "lat", "lon"), cumulative_intensity_all),
            "valid_days": (("year", "lat", "lon"), valid_days_all),
            "expected_days": (("year",), expected_days.astype(np.float32)),
            "ocean_mask": (("lat", "lon"), ocean_mask.astype("uint8")),
        },
        coords={"year": years.astype(np.int16), "lat": lat, "lon": lon},
        attrs={
            "title": "South China Sea annual marine heatwave metrics",
            "source_zarr": str(input_zarr),
            "definition": (
                "MHW event = SST above daily 90th percentile threshold for at least "
                f"{args.min_duration} consecutive valid days. Missing days and calendar gaps break events."
            ),
            "intensity_reference": "SST minus daily climatological mean.",
            "gap_joining": "No short-gap joining is applied in this first-pass implementation.",
        },
    )
    metrics_zarr = output_dir / "mhw_annual_metrics.zarr"
    _write_zarr(
        metrics_ds,
        metrics_zarr,
        {"year": 1, "lat": args.output_lat_chunk, "lon": args.output_lon_chunk},
        args.overwrite,
    )

    annual_csv = output_dir / "mhw_annual_area_mean.csv"
    _maybe_remove(annual_csv, args.overwrite)
    annual_df = _weighted_annual_means(metrics_ds)
    annual_df.to_csv(annual_csv, index=False)
    _log(f"[write] {annual_csv}")

    zero_valid_dates = times[daily_valid_ocean_counts == 0]
    summary = {
        "input_zarr": str(input_zarr),
        "output_dir": str(output_dir),
        "outputs": {
            "daily_climatology_threshold_zarr": str(output_dir / "daily_climatology_threshold.zarr"),
            "mhw_annual_metrics_zarr": str(metrics_zarr),
            "mhw_annual_area_mean_csv": str(annual_csv),
            "summary_json": str(output_dir / "daily_mhw_summary.json"),
        },
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "time_len": int(len(times)),
        "expected_calendar_day_count": int(len(expected)),
        "missing_calendar_day_count": int(len(missing_calendar_days)),
        "missing_calendar_month_counts": {str(k): int(v) for k, v in missing_month_counts.items()},
        "zero_valid_ocean_day_count": int(len(zero_valid_dates)),
        "zero_valid_ocean_month_counts": {
            str(k): int(v)
            for k, v in zero_valid_dates.to_period("M").value_counts().sort_index().astype(int).to_dict().items()
        },
        "baseline_start": args.baseline_start,
        "baseline_end": args.baseline_end,
        "percentile": float(args.percentile),
        "window_days": int(args.window_days),
        "min_duration": int(args.min_duration),
        "ocean_grid_cell_count": int(ocean_mask.sum()),
        "lat_block_size": int(args.lat_block_size),
        "lat_block_count": int(len(blocks)),
        "workers_requested": int(args.workers),
        "workers_used": int(workers),
        "method_notes": [
            "Missing source dates and all-NaN dates are excluded from valid-day counts.",
            "Missing days and calendar gaps break heatwave events.",
            "The three known problematic local months are therefore not treated as no-event months.",
        ],
    }
    summary_path = output_dir / "daily_mhw_summary.json"
    _maybe_remove(summary_path, args.overwrite)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[write] {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily marine heatwave climatology and annual metrics.")
    parser.add_argument("--input-zarr", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-start", default="1991-01-01")
    parser.add_argument("--baseline-end", default="2020-12-31")
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--window-days", type=int, default=11)
    parser.add_argument("--min-duration", type=int, default=5)
    parser.add_argument("--lat-block-size", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel latitude-block worker processes. Use 0 for os.cpu_count().",
    )
    parser.add_argument("--output-lat-chunk", type=int, default=100)
    parser.add_argument("--output-lon-chunk", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_products(args)


if __name__ == "__main__":
    main()
