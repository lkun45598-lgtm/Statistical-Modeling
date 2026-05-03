from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def project_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    raw = Path(cfg["data"]["raw_dir"]).expanduser()
    processed = Path(cfg["data"]["processed_dir"]).expanduser()
    out = Path(cfg["project"].get("output_dir", "outputs")).expanduser()
    return {"raw": raw, "processed": processed, "outputs": out}


def get_rank_world() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
