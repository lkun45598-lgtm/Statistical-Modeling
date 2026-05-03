#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from reefcastnet.config import load_config, project_paths
from reefcastnet.preprocessing import make_toy_zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy.yaml")
    ap.add_argument("--weeks", type=int, default=160)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=48)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = project_paths(cfg)
    out = paths["processed"] / f"{cfg['data']['dataset_name']}.zarr"
    make_toy_zarr(out, weeks=args.weeks, h=args.height, w=args.width)
    print(f"[ok] toy dataset written to {out}")


if __name__ == "__main__":
    main()
