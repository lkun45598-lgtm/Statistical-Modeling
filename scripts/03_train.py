#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from reefcastnet.config import load_config, set_seed, project_paths, is_main_process, save_json
from reefcastnet.dataset import SSTWeeklyForecastDataset
from reefcastnet.losses import ReefCastLoss
from reefcastnet.metrics import sst_metrics, thermal_metrics, merge_metric_dicts
from reefcastnet.models import build_model
from reefcastnet.train_utils import (
    init_distributed, cleanup_distributed, wrap_model, make_loader,
    to_device, AverageMeter, reduce_dict, save_checkpoint,
)


def run_epoch(model, loader, criterion, optimizer, scaler, device, cfg, train: bool):
    model.train(train)
    meter = AverageMeter()
    metric_list = []
    pbar = tqdm(loader, disable=not is_main_process(), desc="train" if train else "val")
    for batch in pbar:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(train):
            use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(
                    batch["x_ssta"],
                    batch["static"],
                    batch["time_feats"],
                    batch["future_time_feats"],
                )
                loss, log = criterion(outputs, batch)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    if cfg["train"].get("grad_clip_norm"):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg["train"].get("grad_clip_norm"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
                    optimizer.step()
        meter.update(log, n=batch["x_ssta"].shape[0])
        if not train:
            with torch.no_grad():
                metric = {}
                metric.update(sst_metrics(outputs["pred_ssta"], batch["y_ssta"], batch["ocean_mask"], cfg["forecast"]["lead_weeks"]))
                metric.update(thermal_metrics(
                    outputs["pred_ssta"], batch["y_ssta"], batch["future_clim"], batch["past_sst"], batch["mmm"],
                    batch["ocean_mask"], cfg["forecast"]["lead_weeks"], cfg["forecast"]["dhw_window_weeks"], cfg["forecast"]["bleaching_threshold_c"],
                ))
                metric_list.append(metric)
        if is_main_process():
            pbar.set_postfix({k: f"{v:.4f}" for k, v in list(meter.avg().items())[:3]})
    out = meter.avg()
    out = reduce_dict(out)
    if metric_list:
        out.update(merge_metric_dicts(metric_list))
    return out


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/south_china_sea.yaml")
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))
    paths = project_paths(cfg)
    zarr_path = Path(args.zarr) if args.zarr else paths["processed"] / f"{cfg['data']['dataset_name']}.zarr"
    run_name = args.run_name or cfg["model"].get("name", "reefcastnet_simvp")
    out_dir = paths["outputs"] / run_name

    use_dist, rank, local_rank, world_size, device = init_distributed(bool(cfg["runtime"].get("distributed", True)))
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(cfg, out_dir / "config_resolved.json")

    train_ds = SSTWeeklyForecastDataset(zarr_path, "train", cfg, random_crop=True)
    val_ds = SSTWeeklyForecastDataset(zarr_path, "val", cfg, random_crop=False)
    train_loader = make_loader(train_ds, cfg, "train", use_dist)
    val_loader = make_loader(val_ds, cfg, "val", use_dist)

    model = build_model(cfg)
    model = wrap_model(model, use_dist, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and bool(cfg["train"].get("amp", True))) else None
    criterion = ReefCastLoss(cfg)

    best = float("inf")
    bad = 0
    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_log = run_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg, train=True)
        val_log = run_epoch(model, val_loader, criterion, optimizer, scaler, device, cfg, train=False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_log.items()}, **{f"val_{k}": v for k, v in val_log.items()}}
        if is_main_process():
            append_csv(out_dir / "history.csv", row)
            current = val_log.get("loss_total", val_log.get("loss_ssta", 999.0))
            print(f"[epoch {epoch}] val={current:.6f}")
            if current < best:
                best = current
                bad = 0
                save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, val_log, cfg)
            else:
                bad += 1
            if epoch % int(cfg["train"].get("save_every", 5)) == 0:
                save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, val_log, cfg)
            if bad >= int(cfg["train"].get("early_stop_patience", 12)):
                print("[early stop]")
                break
    cleanup_distributed()


if __name__ == "__main__":
    main()
