from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
from tqdm import tqdm


def build_griddap_nc_url(
    base_url: str,
    dataset: str,
    variable: str,
    start_time: str,
    end_time: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> str:
    """Build an ERDDAP griddap NetCDF URL for NOAA CRW products.

    Coordinate order for CRW griddap products is normally time, latitude, longitude.
    """
    query = (
        f"{variable}"
        f"[({start_time}):1:({end_time})]"
        f"[({lat_min}):1:({lat_max})]"
        f"[({lon_min}):1:({lon_max})]"
    )
    return f"{base_url.rstrip('/')}/{dataset}.nc?" + quote(query, safe="[]():,=./-TZ_")


def stream_download(url: str, out_path: str | Path, chunk_size: int = 2**20, timeout: int = 120) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    tmp_path.replace(out_path)


def year_start_end(year: int, suffix: str = "T12:00:00Z") -> tuple[str, str]:
    return f"{year}-01-01{suffix}", f"{year}-12-31{suffix}"


def download_years(
    *,
    base_url: str,
    dataset: str,
    variable: str,
    years: Iterable[int],
    out_dir: str | Path,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    time_suffix: str = "T12:00:00Z",
    force: bool = False,
) -> list[Path]:
    out_paths: list[Path] = []
    out_dir = Path(out_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        out_path = out_dir / f"{dataset}_{variable}_{year}.nc"
        out_paths.append(out_path)
        if out_path.exists() and not force:
            print(f"[skip] {out_path}")
            continue
        start, end = year_start_end(int(year), time_suffix)
        url = build_griddap_nc_url(base_url, dataset, variable, start, end, lat_min, lat_max, lon_min, lon_max)
        print(f"[download] {url}")
        stream_download(url, out_path)
    return out_paths
