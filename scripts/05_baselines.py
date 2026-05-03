#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from reefcastnet.config import load_config, project_paths, save_json
from reefcastnet.dataset import SSTWeeklyForecastDataset
from reefcastnet.metrics import sst_metrics, thermal_metrics, merge_metric_dicts


def baseline_pred(batch, mode: str):
    y = batch["y_ssta"]
    if mode == "persistence":
        last = batch["x_ssta"][:, -1:]
        return last.expand(-1, y.shape[1], -1, -1, -1).clone()
    if mode == "climatology":
        # Since target is SSTA, climatology baseline predicts zero anomaly.
        return torch.zeros_like(y)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/south_china_sea.yaml")
    ap.add_argument("--baseline", choices=["persistence", "climatology"], default="persistence")
    ap.add_argument("--split", default="test")
    ap.add_argument("--zarr", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = project_paths(cfg)
    zarr_path = Path(args.zarr) if args.zarr else paths["processed"] / f"{cfg['data']['dataset_name']}.zarr"
    ds = SSTWeeklyForecastDataset(zarr_path, args.split, cfg, random_crop=False)
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"].get("num_workers", 0)))

    metrics = []
    for batch in tqdm(loader, desc=f"{args.baseline}-{args.split}"):
        pred = baseline_pred(batch, args.baseline)
        metric = {}
        metric.update(sst_metrics(pred, batch["y_ssta"], batch["ocean_mask"], cfg["forecast"]["lead_weeks"]))
        metric.update(thermal_metrics(
            pred, batch["y_ssta"], batch["future_clim"], batch["past_sst"], batch["mmm"],
            batch["ocean_mask"], cfg["forecast"]["lead_weeks"], cfg["forecast"]["dhw_window_weeks"], cfg["forecast"]["bleaching_threshold_c"],
        ))
        metrics.append(metric)
    summary = merge_metric_dicts(metrics)
    out_dir = paths["outputs"] / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_dir / f"{args.baseline}_{args.split}.json")
    print(summary)


if __name__ == "__main__":
    main()
