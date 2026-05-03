#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path

from reefcastnet.config import load_config, project_paths
from reefcastnet.erddap import download_years


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/south_china_sea.yaml")
    ap.add_argument("--years", nargs="*", type=int, default=None, help="Optional list/range: 2020 2021 or 2010 2025")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = project_paths(cfg)
    if args.years is None:
        y0, y1 = cfg["data"]["years"]
        years = range(int(y0), int(y1) + 1)
    elif len(args.years) == 2 and args.years[1] - args.years[0] > 1:
        years = range(args.years[0], args.years[1] + 1)
    else:
        years = args.years

    r = cfg["region"]
    e = cfg["erddap"]
    download_years(
        base_url=e["base_url"],
        dataset=e["sst_dataset"],
        variable=e["sst_var"],
        years=years,
        out_dir=paths["raw"],
        lat_min=r["lat_min"],
        lat_max=r["lat_max"],
        lon_min=r["lon_min"],
        lon_max=r["lon_max"],
        time_suffix=e.get("time_suffix", "T12:00:00Z"),
        force=args.force,
    )


if __name__ == "__main__":
    main()
