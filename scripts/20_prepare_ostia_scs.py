#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc


DEFAULT_INPUT = (
    "/data/sst_data/sst_missing_value_imputation/"
    "copernicus_data/copernicus_sst_monthly_1991_2021.nc"
)
DEFAULT_OUTPUT_DIR = "/data1/user/lz/osita_data"


def _maybe_remove(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _resolve_input_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        preferred = path / "copernicus_sst_monthly_1991_2021.nc"
        if preferred.exists():
            return preferred
        sibling = path.parent / "copernicus_data" / "copernicus_sst_monthly_1991_2021.nc"
        if sibling.exists():
            return sibling
        nc_files = sorted(path.glob("*.nc"))
        if len(nc_files) == 1:
            return nc_files[0]
        if nc_files:
            raise FileNotFoundError(
                f"Directory {path} contains multiple .nc files; please pass the target file explicitly."
            )
    raise FileNotFoundError(f"Could not resolve OSTIA input path: {path}")


def _coord_name(ds: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.coords:
            return name
    raise KeyError(f"Could not find any coordinate in {names}")


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
    with ProgressBar():
        ds.to_zarr(path, mode="w", consolidated=True, encoding=_encoding(ds, chunks))


def _sample_stats(ds: xr.Dataset) -> dict[str, Any]:
    samples = []
    time_len = int(ds.sizes["time"])
    for idx in [0, time_len // 2, time_len - 1]:
        arr = ds["sst"].isel(time=idx).values
        finite = arr[np.isfinite(arr)]
        if finite.size:
            sst_min = float(finite.min())
            sst_mean = float(finite.mean())
            sst_max = float(finite.max())
        else:
            sst_min = None
            sst_mean = None
            sst_max = None
        samples.append(
            {
                "index": int(idx),
                "time": str(ds["time"].values[idx]),
                "sst_c_min": sst_min,
                "sst_c_mean": sst_mean,
                "sst_c_max": sst_max,
                "nan_rate": float(np.isnan(arr).mean()),
            }
        )
    return {
        "time_start": str(ds["time"].values[0]),
        "time_end": str(ds["time"].values[-1]),
        "time_len": time_len,
        "lat_min": float(ds["lat"].values.min()),
        "lat_max": float(ds["lat"].values.max()),
        "lon_min": float(ds["lon"].values.min()),
        "lon_max": float(ds["lon"].values.max()),
        "lat_size": int(ds.sizes["lat"]),
        "lon_size": int(ds.sizes["lon"]),
        "sample_stats": samples,
    }


def prepare(args: argparse.Namespace) -> None:
    input_path = _resolve_input_path(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = xr.open_dataset(input_path, chunks={"time": args.source_time_chunk})
    lat_name = _coord_name(src, ("latitude", "lat"))
    lon_name = _coord_name(src, ("longitude", "lon"))
    if "analysed_sst" not in src:
        raise KeyError("Input dataset must contain analysed_sst")

    lat_vals = src[lat_name].values
    lat_slice = slice(args.lat_min, args.lat_max) if lat_vals[0] < lat_vals[-1] else slice(args.lat_max, args.lat_min)
    scs = src.sel({lat_name: lat_slice, lon_name: slice(args.lon_min, args.lon_max)})
    scs = scs.rename({lat_name: "lat", lon_name: "lon"})

    sst_c = (scs["analysed_sst"] - 273.15).astype("float32")
    sst_c.name = "sst"
    sst_c.attrs.update(
        {
            "long_name": "sea surface foundation temperature",
            "units": "degree_Celsius",
            "source_variable": "analysed_sst",
            "conversion": "analysed_sst_kelvin - 273.15",
        }
    )

    if "mask" in scs:
        source_mask = scs["mask"].isel(time=0).astype("uint8").rename("source_mask")
        ocean_mask = ((source_mask & np.uint8(1)) > 0).astype("uint8").rename("ocean_mask")
        land_mask = ((source_mask & np.uint8(2)) > 0).astype("uint8").rename("land_mask")
    else:
        source_mask = xr.full_like(sst_c.isel(time=0), 0, dtype="uint8").rename("source_mask")
        ocean_mask = xr.where(np.isfinite(sst_c.isel(time=0)), 1, 0).astype("uint8").rename("ocean_mask")
        land_mask = (1 - ocean_mask).astype("uint8").rename("land_mask")

    daily = xr.Dataset(
        {
            "sst": sst_c,
            "source_mask": source_mask,
            "ocean_mask": ocean_mask,
            "land_mask": land_mask,
        },
        coords={"time": scs["time"], "lat": scs["lat"], "lon": scs["lon"]},
        attrs={
            "title": "OSTIA South China Sea daily SST crop",
            "source_file": str(input_path),
            "source_title": str(src.attrs.get("title", "")),
            "source_institution": str(src.attrs.get("institution", "")),
            "source_references": str(src.attrs.get("references", "")),
            "region": f"{args.lat_min}-{args.lat_max}N, {args.lon_min}-{args.lon_max}E",
            "notes": "SST converted from Kelvin to degree Celsius. Static masks are taken from the first time step.",
        },
    )

    daily_chunks = {"time": args.time_chunk, "lat": args.lat_chunk, "lon": args.lon_chunk}
    daily_path = output_dir / "ostia_scs_daily.zarr"
    print(f"[write] {daily_path}")
    _write_zarr(daily, daily_path, daily_chunks, args.overwrite)

    monthly = daily[["sst"]].resample(time="MS").mean(skipna=True).astype("float32")
    monthly["source_mask"] = daily["source_mask"]
    monthly["ocean_mask"] = daily["ocean_mask"]
    monthly["land_mask"] = daily["land_mask"]
    monthly.attrs.update(daily.attrs)
    monthly.attrs["title"] = "OSTIA South China Sea monthly mean SST crop"
    monthly_path = output_dir / "ostia_scs_monthly.zarr"
    print(f"[write] {monthly_path}")
    _write_zarr(monthly, monthly_path, {"time": 12, "lat": args.lat_chunk, "lon": args.lon_chunk}, args.overwrite)

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "daily_zarr": str(daily_path),
        "monthly_zarr": str(monthly_path),
        "region": {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        },
        "daily": _sample_stats(daily),
        "monthly_time_len": int(monthly.sizes["time"]),
        "processing": {
            "workers": args.workers,
            "source_time_chunk": args.source_time_chunk,
            "output_chunks": daily_chunks,
            "compressor": "zstd clevel=3 bitshuffle",
        },
    }
    metadata_path = output_dir / "metadata.json"
    _maybe_remove(metadata_path, args.overwrite)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] metadata written to {metadata_path}")
    src.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OSTIA South China Sea cropped SST data.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lat-min", type=float, default=0.0)
    parser.add_argument("--lat-max", type=float, default=25.0)
    parser.add_argument("--lon-min", type=float, default=100.0)
    parser.add_argument("--lon-max", type=float, default=125.0)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--source-time-chunk", type=int, default=32)
    parser.add_argument("--time-chunk", type=int, default=32)
    parser.add_argument("--lat-chunk", type=int, default=250)
    parser.add_argument("--lon-chunk", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import dask

    dask.config.set(scheduler="threads", num_workers=args.workers)
    prepare(args)


if __name__ == "__main__":
    main()
