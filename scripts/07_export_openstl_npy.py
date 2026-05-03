#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from reefcastnet.config import load_config, project_paths
from reefcastnet.dataset import SSTWeeklyForecastDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/south_china_sea.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = project_paths(cfg)
    zarr_path = Path(args.zarr) if args.zarr else paths["processed"] / f"{cfg['data']['dataset_name']}.zarr"
    out_dir = Path(args.out_dir) if args.out_dir else paths["processed"] / "openstl_npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SSTWeeklyForecastDataset(zarr_path, args.split, cfg, random_crop=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    xs, ys = [], []
    for i, batch in enumerate(tqdm(loader, desc=f"export-{args.split}")):
        if i >= args.max_samples:
            break
        # OpenSTL convention is often [N,T,C,H,W].
        xs.append(batch["x_ssta"].numpy()[0])
        ys.append(batch["y_ssta"].numpy()[0])
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    np.save(out_dir / f"{args.split}_x.npy", x)
    np.save(out_dir / f"{args.split}_y.npy", y)
    print(f"[ok] {out_dir / f'{args.split}_x.npy'} {x.shape}")
    print(f"[ok] {out_dir / f'{args.split}_y.npy'} {y.shape}")


if __name__ == "__main__":
    main()
