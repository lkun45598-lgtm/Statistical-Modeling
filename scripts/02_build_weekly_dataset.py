#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path

from reefcastnet.config import load_config, project_paths
from reefcastnet.preprocessing import build_weekly_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/south_china_sea.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = project_paths(cfg)
    dataset = cfg["erddap"]["sst_dataset"]
    raw_dir = paths["raw"] / dataset
    files = sorted(raw_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found under {raw_dir}. Run scripts/01_download_noaa_crw.py first.")
    out = paths["processed"] / f"{cfg['data']['dataset_name']}.zarr"
    build_weekly_dataset(cfg, files, out)
    print(f"[ok] processed dataset written to {out}")


if __name__ == "__main__":
    main()
