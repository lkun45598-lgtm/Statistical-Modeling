from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_OUTPUT_DIR = Path("/data1/user/lz/osita_data/external_drivers")
DEFAULT_START = "1991-01-01"
DEFAULT_END = "2021-12-01"

DEFAULT_REGION = {
    "lat_min": 0.0,
    "lat_max": 25.0,
    "lon_min": 100.0,
    "lon_max": 125.0,
}

TEXT_SOURCES = {
    "nino34_cpc": {
        "url": "https://psl.noaa.gov/data/correlation/nina34.anom.data",
        "description": "NOAA/CPC Niño 3.4 monthly SST anomaly index",
    },
    "pdo": {
        "url": "https://psl.noaa.gov/pdo/data/pdo.timeseries.sstens.data",
        "description": "NOAA/PSL PDO monthly index",
    },
    "dmi": {
        "url": "https://psl.noaa.gov/data/timeseries/month/data/dmi.had.long.data",
        "description": "NOAA/PSL DMI (HadISST) monthly index",
    },
    "soi": {
        "url": "https://psl.noaa.gov/data/timeseries/month/data/soi.long.data",
        "description": "PSL monthly SOI index",
    },
}

NETCDF_SOURCES = {
    "uwnd_10m": {
        "url": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/surface_gauss/uwnd.10m.mon.mean.nc",
        "var_name": "uwnd",
        "description": "NCEP/NCAR Reanalysis 1 monthly mean 10m zonal wind",
    },
    "vwnd_10m": {
        "url": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/surface_gauss/vwnd.10m.mon.mean.nc",
        "var_name": "vwnd",
        "description": "NCEP/NCAR Reanalysis 1 monthly mean 10m meridional wind",
    },
}


@dataclass
class DownloadedFile:
    key: str
    source_url: str
    local_path: Path
    size_bytes: int


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Codex downloader)",
            "Accept": "*/*",
        },
    )


def download_file(url: str, dest: Path, force: bool = False, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(_request(url), timeout=120) as response, tmp.open("wb") as f:
                shutil.copyfileobj(response, f, length=1024 * 1024)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            last_err = err
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                continue
    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_err


def _parse_psl_monthly_series(path: Path) -> pd.DataFrame:
    rows: list[tuple[pd.Timestamp, float]] = []
    year_pattern = re.compile(r"^\s*(\d{4})\s+")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("https://"):
                continue
            match = year_pattern.match(line)
            if not match:
                continue
            tokens = line.replace(",", " ").split()
            if len(tokens) < 13:
                continue
            try:
                year = int(tokens[0])
                values = [float(tok) for tok in tokens[1:13]]
            except ValueError:
                continue
            for month, value in enumerate(values, start=1):
                if value <= -99:
                    value = np.nan
                rows.append((pd.Timestamp(year=year, month=month, day=1), float(value)))
    if not rows:
        raise ValueError(f"No monthly values parsed from {path}")
    df = pd.DataFrame(rows, columns=["time", "value"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return df


def _merge_monthly_series(series_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for key, df in series_map.items():
        renamed = df.rename(columns={"value": key})
        merged = renamed if merged is None else merged.merge(renamed, on="time", how="outer")
    if merged is None:
        raise ValueError("No series to merge")
    merged = merged.sort_values("time").reset_index(drop=True)
    return merged


def _select_time_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    mask = (out["time"] >= pd.Timestamp(start)) & (out["time"] <= pd.Timestamp(end))
    return out.loc[mask].reset_index(drop=True)


def _derive_oni_like_from_nino34(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "nino34_cpc" in out.columns:
        out["nino34_cpc_3m_centered"] = (
            out.set_index("time")["nino34_cpc"].rolling(window=3, center=True, min_periods=3).mean().reset_index(drop=True)
        )
    return out


def _normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords:
        return ds
    lon = ds["lon"]
    if float(lon.min()) < 0.0:
        ds = ds.assign_coords(lon=(((lon + 360.0) % 360.0).astype(lon.dtype)))
    return ds.sortby("lon")


def _crop_wind_dataset(
    ds: xr.Dataset,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start: str,
    end: str,
) -> xr.Dataset:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError(f"Expected lat/lon coordinates, found {list(ds.coords)}")

    if float(ds.lat[0]) > float(ds.lat[-1]):
        ds = ds.sortby("lat")
    ds = _normalize_lon(ds)

    ds = ds.sel(time=slice(start, end))
    ds = ds.sel(lat=slice(lat_min, lat_max))
    ds = ds.sel(lon=slice(lon_min, lon_max))
    return ds


def _build_wind_product(
    uwnd_path: Path,
    vwnd_path: Path,
    output_dir: Path,
    region: dict[str, float],
    start: str,
    end: str,
    force: bool = False,
) -> dict:
    wind_zarr = output_dir / "ncep_wind_scs_monthly.zarr"
    wind_csv = output_dir / "ncep_wind_scs_region_mean.csv"
    wind_meta = output_dir / "ncep_wind_metadata.json"

    if wind_zarr.exists() and wind_csv.exists() and wind_meta.exists() and not force:
        return {
            "wind_zarr": str(wind_zarr),
            "wind_csv": str(wind_csv),
            "wind_meta": str(wind_meta),
        }

    ds_u = xr.open_dataset(uwnd_path).load()
    ds_v = xr.open_dataset(vwnd_path).load()

    if "uwnd" not in ds_u:
        raise KeyError(f"Could not find uwnd variable in {uwnd_path}")
    if "vwnd" not in ds_v:
        raise KeyError(f"Could not find vwnd variable in {vwnd_path}")

    ds = xr.Dataset(
        data_vars={
            "u10": ds_u["uwnd"],
            "v10": ds_v["vwnd"],
        }
    )
    ds = _crop_wind_dataset(
        ds,
        region["lat_min"],
        region["lat_max"],
        region["lon_min"],
        region["lon_max"],
        start=start,
        end=end,
    )
    ds["wind_speed"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2).astype("float32")

    ds = ds.astype("float32")
    ds["u10"].attrs.update({"units": "m s-1", "long_name": "10 m zonal wind"})
    ds["v10"].attrs.update({"units": "m s-1", "long_name": "10 m meridional wind"})
    ds["wind_speed"].attrs.update({"units": "m s-1", "long_name": "10 m wind speed"})
    ds.attrs.update(
        {
            "description": "NCEP/NCAR Reanalysis 1 monthly mean wind crop for South China Sea",
            "region": json.dumps(region, ensure_ascii=False),
        }
    )

    if wind_zarr.exists():
        shutil.rmtree(wind_zarr)
    ds.chunk({"time": 32, "lat": min(32, ds.sizes["lat"]), "lon": min(32, ds.sizes["lon"])}).to_zarr(wind_zarr, mode="w")

    region_mean = ds[["u10", "v10", "wind_speed"]].mean(dim=("lat", "lon"), skipna=True).to_dataframe().reset_index()
    region_mean.to_csv(wind_csv, index=False)

    meta = {
        "source_files": {
            "uwnd_10m": str(uwnd_path),
            "vwnd_10m": str(vwnd_path),
        },
        "region": region,
        "time_start": str(ds.time.values[0]),
        "time_end": str(ds.time.values[-1]),
        "time_len": int(ds.sizes["time"]),
        "lat_min": float(ds.lat.min()),
        "lat_max": float(ds.lat.max()),
        "lon_min": float(ds.lon.min()),
        "lon_max": float(ds.lon.max()),
        "lat_size": int(ds.sizes["lat"]),
        "lon_size": int(ds.sizes["lon"]),
        "variables": list(ds.data_vars),
    }
    wind_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "wind_zarr": str(wind_zarr),
        "wind_csv": str(wind_csv),
        "wind_meta": str(wind_meta),
    }


def build_output(
    output_dir: Path,
    region: dict[str, float],
    start: str,
    end: str,
    force: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[DownloadedFile] = []
    parsed_series: dict[str, pd.DataFrame] = {}

    for key, meta in TEXT_SOURCES.items():
        raw_path = raw_dir / Path(meta["url"]).name
        download_file(meta["url"], raw_path, force=force)
        downloaded.append(DownloadedFile(key=key, source_url=meta["url"], local_path=raw_path, size_bytes=raw_path.stat().st_size))
        parsed_series[key] = _parse_psl_monthly_series(raw_path)

    combined = _merge_monthly_series(parsed_series)
    combined = _derive_oni_like_from_nino34(combined)
    full_csv = output_dir / "climate_indices_monthly_full.csv"
    combined.to_csv(full_csv, index=False)

    aligned = _select_time_window(combined, start=start, end=end)
    aligned_csv = output_dir / "climate_indices_monthly_1991_2021.csv"
    aligned.to_csv(aligned_csv, index=False)

    for key, meta in NETCDF_SOURCES.items():
        raw_path = raw_dir / Path(meta["url"]).name
        download_file(meta["url"], raw_path, force=force)
        downloaded.append(DownloadedFile(key=key, source_url=meta["url"], local_path=raw_path, size_bytes=raw_path.stat().st_size))

    wind_product = _build_wind_product(
        raw_dir / Path(NETCDF_SOURCES["uwnd_10m"]["url"]).name,
        raw_dir / Path(NETCDF_SOURCES["vwnd_10m"]["url"]).name,
        output_dir,
        region,
        start,
        end,
        force=force,
    )

    meta = {
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "region": region,
        "time_window": {"start": start, "end": end},
        "text_sources": {key: {"url": meta["url"], "description": meta["description"]} for key, meta in TEXT_SOURCES.items()},
        "netcdf_sources": {key: {"url": meta["url"], "description": meta["description"]} for key, meta in NETCDF_SOURCES.items()},
        "downloads": [
            {
                "key": item.key,
                "source_url": item.source_url,
                "local_path": str(item.local_path),
                "size_bytes": item.size_bytes,
            }
            for item in downloaded
        ],
        "outputs": {
            "climate_indices_full_csv": str(full_csv),
            "climate_indices_aligned_csv": str(aligned_csv),
            **wind_product,
        },
        "notes": [
            "nino34_cpc_3m_centered is a convenience derived series computed from downloaded Niño 3.4 data.",
            "wind fields are cropped to the South China Sea region and saved as monthly means.",
        ],
    }
    (output_dir / "download_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# External Drivers",
                "",
                "This directory contains downloaded climate indices and NCEP/NCAR monthly wind fields for South China Sea analysis.",
                "",
                "Main aligned table: `climate_indices_monthly_1991_2021.csv`",
                "Wind crop: `ncep_wind_scs_monthly.zarr`",
                "",
                "Sources:",
                "- NOAA PSL Niño 3.4 monthly SST anomaly index",
                "- NOAA PSL PDO monthly index",
                "- NOAA PSL DMI monthly index",
                "- PSL monthly SOI index",
                "- NCEP/NCAR Reanalysis 1 monthly 10 m winds",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return meta


def _print_summary(meta: dict) -> None:
    outputs = meta["outputs"]
    print("[ok] external drivers downloaded and processed")
    print(f"[ok] aligned climate indices: {outputs['climate_indices_aligned_csv']}")
    print(f"[ok] full climate indices: {outputs['climate_indices_full_csv']}")
    print(f"[ok] wind crop zarr: {outputs['wind_zarr']}")
    print(f"[ok] wind region mean csv: {outputs['wind_csv']}")
    print(f"[ok] metadata: {Path(outputs['wind_meta']).parent / 'download_metadata.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download external climate drivers for South China Sea analysis.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for downloaded data.")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date for aligned outputs (YYYY-MM-DD).")
    parser.add_argument("--end", default=DEFAULT_END, help="End date for aligned outputs (YYYY-MM-DD).")
    parser.add_argument("--lat-min", type=float, default=DEFAULT_REGION["lat_min"], help="South China Sea crop minimum latitude.")
    parser.add_argument("--lat-max", type=float, default=DEFAULT_REGION["lat_max"], help="South China Sea crop maximum latitude.")
    parser.add_argument("--lon-min", type=float, default=DEFAULT_REGION["lon_min"], help="South China Sea crop minimum longitude.")
    parser.add_argument("--lon-max", type=float, default=DEFAULT_REGION["lon_max"], help="South China Sea crop maximum longitude.")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing files.")
    args = parser.parse_args()

    region = {
        "lat_min": float(args.lat_min),
        "lat_max": float(args.lat_max),
        "lon_min": float(args.lon_min),
        "lon_max": float(args.lon_max),
    }
    meta = build_output(args.output_dir, region=region, start=args.start, end=args.end, force=args.force)
    _print_summary(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
