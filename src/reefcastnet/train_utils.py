from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .config import get_rank_world


def init_distributed(enabled: bool = True) -> tuple[bool, int, int, int, torch.device]:
    rank, local_rank, world_size = get_rank_world()
    use_dist = enabled and world_size > 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if use_dist and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return use_dist, rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: torch.nn.Module, use_dist: bool, device: torch.device) -> torch.nn.Module:
    model = model.to(device)
    if use_dist:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)
    return model


def make_loader(dataset, cfg: Dict[str, Any], split: str, use_dist: bool) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=(split == "train")) if use_dist else None
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def reduce_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return d
    keys = sorted(d.keys())
    vals = torch.tensor([d[k] for k in keys], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= dist.get_world_size()
    return {k: float(v.item()) for k, v in zip(keys, vals)}


class AverageMeter:
    def __init__(self):
        self.sum: Dict[str, float] = {}
        self.count = 0

    def update(self, d: Dict[str, float], n: int = 1) -> None:
        for k, v in d.items():
            self.sum[k] = self.sum.get(k, 0.0) + float(v) * n
        self.count += n

    def avg(self) -> Dict[str, float]:
        return {k: v / max(self.count, 1) for k, v in self.sum.items()}


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer, epoch: int, metrics: Dict[str, float], cfg: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "model": state_dict,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "cfg": cfg,
    }, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location="cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt
