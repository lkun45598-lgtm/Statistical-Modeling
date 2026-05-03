#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from reefcastnet.config import load_config, project_paths, save_json
from reefcastnet.dataset import SSTWeeklyForecastDataset
from reefcastnet.metrics import sst_metrics, thermal_metrics, merge_metric_dicts
from reefcastnet.models import build_model
from reefcastnet.train_utils import load_checkpoint, to_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/south_china_sea.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--save-npz", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = project_paths(cfg)
    zarr_path = Path(args.zarr) if args.zarr else paths["processed"] / f"{cfg['data']['dataset_name']}.zarr"
    out_dir = Path(args.checkpoint).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SSTWeeklyForecastDataset(zarr_path, args.split, cfg, random_crop=False)
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"].get("num_workers", 0)))

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    metrics = []
    pred_examples = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"eval-{args.split}")):
            batch = to_device(batch, device)
            out = model(batch["x_ssta"], batch["static"], batch["time_feats"], batch["future_time_feats"])
            metric = {}
            metric.update(sst_metrics(out["pred_ssta"], batch["y_ssta"], batch["ocean_mask"], cfg["forecast"]["lead_weeks"]))
            metric.update(thermal_metrics(
                out["pred_ssta"], batch["y_ssta"], batch["future_clim"], batch["past_sst"], batch["mmm"],
                batch["ocean_mask"], cfg["forecast"]["lead_weeks"], cfg["forecast"]["dhw_window_weeks"], cfg["forecast"]["bleaching_threshold_c"],
            ))
            metrics.append(metric)
            if args.save_npz and len(pred_examples) < 4:
                pred_examples.append({
                    "pred_ssta": out["pred_ssta"].cpu().numpy(),
                    "true_ssta": batch["y_ssta"].cpu().numpy(),
                    "future_sst": batch["future_sst"].cpu().numpy(),
                    "future_clim": batch["future_clim"].cpu().numpy(),
                    "ocean_mask": batch["ocean_mask"].cpu().numpy(),
                    "reef_mask": batch["reef_mask"].cpu().numpy(),
                })
    summary = merge_metric_dicts(metrics)
    save_json(summary, out_dir / f"eval_{args.split}.json")
    print(summary)
    if args.save_npz and pred_examples:
        # Save the first batch examples compactly.
        ex = pred_examples[0]
        np.savez_compressed(out_dir / f"examples_{args.split}.npz", **ex)
        print(f"[ok] examples saved to {out_dir / f'examples_{args.split}.npz'}")


if __name__ == "__main__":
    main()
