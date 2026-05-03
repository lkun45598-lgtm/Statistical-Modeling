#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from numcodecs import Blosc

try:
    from scipy.stats import linregress
except Exception:  # pragma: no cover - scipy is available on the target machine.
    linregress = None


DEFAULT_INPUT = "/data1/user/lz/osita_data/scs_5s25n/ostia_scs_monthly.zarr"
DEFAULT_OUTPUT_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis"


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


def _encoding(ds: xr.Dataset, chunks: dict[str, int]) -> dict[str, dict[str, Any]]:
    compressor = _compressor()
    encoding: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        var_chunks = tuple(chunks[d] for d in da.dims if d in chunks)
        enc: dict[str, Any] = {"compressor": compressor}
        if var_chunks:
            enc["chunks"] = var_chunks
        encoding[name] = enc
    return encoding


def _write_zarr(ds: xr.Dataset, path: Path, chunks: dict[str, int], overwrite: bool) -> None:
    _maybe_remove(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.chunk({k: v for k, v in chunks.items() if k in ds.dims})
    print(f"[write] {path}")
    with ProgressBar():
        ds.to_zarr(path, mode="w", consolidated=True, encoding=_encoding(ds, chunks))


def _years_since_start(time_values: np.ndarray) -> np.ndarray:
    times = np.asarray(time_values).astype("datetime64[ns]")
    return ((times - times[0]) / np.timedelta64(1, "D")).astype("float64") / 365.2425


def _linear_trend_1d(values: np.ndarray, years: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(values) & np.isfinite(years)
    x = years[mask]
    y = values[mask]
    if len(y) < 3:
        return {"n": int(len(y)), "slope_c_per_year": None}

    if linregress is not None:
        fit = linregress(x, y)
        slope = float(fit.slope)
        result = {
            "n": int(len(y)),
            "slope_c_per_year": slope,
            "slope_c_per_decade": slope * 10.0,
            "linear_change_c": slope * float(x[-1] - x[0]),
            "intercept_at_start_c": float(fit.intercept),
            "r_value": float(fit.rvalue),
            "p_value": float(fit.pvalue),
            "stderr": float(fit.stderr),
            "mean_c": float(np.nanmean(y)),
            "first_c": float(y[0]),
            "last_c": float(y[-1]),
        }
        return result

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_value = float(np.sqrt(max(0.0, 1.0 - ss_res / ss_tot))) if ss_tot else 0.0
    return {
        "n": int(len(y)),
        "slope_c_per_year": float(slope),
        "slope_c_per_decade": float(slope * 10.0),
        "linear_change_c": float(slope * (x[-1] - x[0])),
        "intercept_at_start_c": float(intercept),
        "r_value": r_value,
        "p_value": None,
        "stderr": None,
        "mean_c": float(np.nanmean(y)),
        "first_c": float(y[0]),
        "last_c": float(y[-1]),
    }


def _linear_trend_map(da: xr.DataArray, years: xr.DataArray, min_obs: int) -> xr.DataArray:
    valid = xr.where(da.notnull(), 1.0, 0.0)
    n = valid.sum("time")
    n_safe = n.where(n > 0)
    t_mean = (years * valid).sum("time") / n_safe
    y_mean = da.fillna(0.0).sum("time") / n_safe
    dt = years - t_mean
    dy = da - y_mean
    denominator = ((dt**2) * valid).sum("time")
    denominator_safe = denominator.where(denominator > 0)
    slope = (dt * dy).where(da.notnull()).sum("time") / denominator_safe
    return slope.where(n >= min_obs)


def _area_weights(ds: xr.Dataset) -> xr.DataArray:
    if "ocean_mask" not in ds:
        raise KeyError("Input monthly Zarr must contain ocean_mask.")
    lat_weights = np.cos(np.deg2rad(ds["lat"])).astype("float32")
    weights = lat_weights.broadcast_like(ds["ocean_mask"]).where(ds["ocean_mask"] == 1, 0.0)
    weights.name = "area_weight"
    return weights


def build_products(args: argparse.Namespace) -> None:
    input_zarr = Path(args.input_zarr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(input_zarr, consolidated=True)
    if "sst" not in ds:
        raise KeyError("Input monthly Zarr must contain sst.")

    sst = ds["sst"].chunk({"time": args.time_chunk, "lat": args.lat_chunk, "lon": args.lon_chunk})
    ocean = ds["ocean_mask"]
    sst_ocean = sst.where(ocean == 1)

    baseline = sst_ocean.sel(time=slice(args.baseline_start, args.baseline_end))
    if int(baseline.sizes.get("time", 0)) == 0:
        raise ValueError("Baseline period produced no monthly samples.")

    climatology = baseline.groupby("time.month").mean("time", skipna=True).astype("float32")
    ssta = (sst_ocean.groupby("time.month") - climatology).astype("float32")

    monthly_products = xr.Dataset(
        {
            "ssta": ssta,
            "sst_climatology": climatology,
            "ocean_mask": ocean.astype("uint8"),
        },
        attrs={
            "title": "South China Sea monthly SST anomaly products",
            "source_zarr": str(input_zarr),
            "source_region": str(ds.attrs.get("region", "")),
            "baseline_period": f"{args.baseline_start} to {args.baseline_end}",
            "area_weighting": "cos(latitude) weights with ocean_mask == 1",
        },
    )
    if "land_mask" in ds:
        monthly_products["land_mask"] = ds["land_mask"].astype("uint8")

    _write_zarr(
        monthly_products,
        output_dir / "monthly_ssta.zarr",
        {"time": args.time_chunk, "month": 12, "lat": args.lat_chunk, "lon": args.lon_chunk},
        args.overwrite,
    )

    weights = _area_weights(ds)
    sst_mean = sst_ocean.weighted(weights).mean(("lat", "lon"), skipna=True)
    ssta_mean = ssta.weighted(weights).mean(("lat", "lon"), skipna=True)
    ocean_fraction = (ocean == 1).mean()

    print("[compute] area mean SST/SSTA")
    with ProgressBar():
        sst_mean_c, ssta_mean_c, ocean_fraction_c = dask.compute(sst_mean, ssta_mean, ocean_fraction)

    time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    area_mean = pd.DataFrame(
        {
            "time": time_index.strftime("%Y-%m-%d"),
            "year": time_index.year,
            "month": time_index.month,
            "sst_area_mean_c": sst_mean_c.values.astype("float64"),
            "ssta_area_mean_c": ssta_mean_c.values.astype("float64"),
        }
    )
    area_mean_csv = output_dir / "scs_monthly_area_mean_sst_ssta.csv"
    _maybe_remove(area_mean_csv, args.overwrite)
    area_mean.to_csv(area_mean_csv, index=False)
    print(f"[write] {area_mean_csv}")

    missing_area_mean = area_mean.loc[
        ~np.isfinite(area_mean["sst_area_mean_c"]) | ~np.isfinite(area_mean["ssta_area_mean_c"]),
        ["time", "year", "month"],
    ].to_dict(orient="records")

    years = _years_since_start(ds["time"].values)
    trend_summary = {
        "sst_area_mean": _linear_trend_1d(area_mean["sst_area_mean_c"].to_numpy(), years),
        "ssta_area_mean": _linear_trend_1d(area_mean["ssta_area_mean_c"].to_numpy(), years),
    }

    years_da = xr.DataArray(years, dims="time", coords={"time": ds["time"]}, name="years_since_start")
    ssta_slope = _linear_trend_map(ssta, years_da, args.min_obs).astype("float32")
    sst_mean_map = sst_ocean.mean("time", skipna=True).astype("float32")
    sst_std_map = sst_ocean.std("time", skipna=True).astype("float32")
    trend_maps = xr.Dataset(
        {
            "ssta_slope_c_per_year": ssta_slope,
            "ssta_slope_c_per_decade": (ssta_slope * 10.0).astype("float32"),
            "sst_mean_c": sst_mean_map,
            "sst_std_c": sst_std_map,
            "ocean_mask": ocean.astype("uint8"),
        },
        attrs={
            "title": "South China Sea monthly SST anomaly trend maps",
            "source_zarr": str(input_zarr),
            "baseline_period": f"{args.baseline_start} to {args.baseline_end}",
            "trend_method": "ordinary least-squares slope over monthly SSTA",
            "trend_unit": "degree_Celsius per year",
        },
    )
    _write_zarr(
        trend_maps,
        output_dir / "monthly_sst_trend.zarr",
        {"lat": args.lat_chunk, "lon": args.lon_chunk},
        args.overwrite,
    )

    metadata = {
        "input_zarr": str(input_zarr),
        "output_dir": str(output_dir),
        "outputs": {
            "monthly_ssta_zarr": str(output_dir / "monthly_ssta.zarr"),
            "monthly_trend_zarr": str(output_dir / "monthly_sst_trend.zarr"),
            "area_mean_csv": str(area_mean_csv),
            "summary_json": str(output_dir / "monthly_sst_summary.json"),
        },
        "time_start": str(ds["time"].values[0]),
        "time_end": str(ds["time"].values[-1]),
        "time_len": int(ds.sizes["time"]),
        "lat_size": int(ds.sizes["lat"]),
        "lon_size": int(ds.sizes["lon"]),
        "baseline_start": args.baseline_start,
        "baseline_end": args.baseline_end,
        "ocean_grid_fraction": float(ocean_fraction_c.values),
        "missing_area_mean_month_count": len(missing_area_mean),
        "missing_area_mean_months": missing_area_mean,
        "trend_summary": trend_summary,
    }
    summary_path = output_dir / "monthly_sst_summary.json"
    _maybe_remove(summary_path, args.overwrite)
    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build monthly SST anomaly and trend products for the South China Sea.")
    parser.add_argument("--input-zarr", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-start", default="1991-01-01")
    parser.add_argument("--baseline-end", default="2020-12-31")
    parser.add_argument("--time-chunk", type=int, default=12)
    parser.add_argument("--lat-chunk", type=int, default=250)
    parser.add_argument("--lon-chunk", type=int, default=250)
    parser.add_argument("--min-obs", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_products(args)


if __name__ == "__main__":
    main()
