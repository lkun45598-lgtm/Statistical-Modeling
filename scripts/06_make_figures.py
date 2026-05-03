#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="examples_test.npz from evaluation")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lead-index", type=int, default=-1)
    args = ap.parse_args()
    path = Path(args.npz)
    out_dir = Path(args.out_dir) if args.out_dir else path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(path)
    pred_sst = data["pred_ssta"] + data["future_clim"]
    true_sst = data["future_sst"]
    li = args.lead_index
    # first sample, chosen lead
    pred = pred_sst[0, li, 0]
    true = true_sst[0, li, 0]
    err = pred - true
    for arr, name, title in [(true, "truth", "True SST"), (pred, "prediction", "Predicted SST"), (err, "error", "Prediction Error")]:
        plt.figure(figsize=(6, 4))
        plt.imshow(arr, origin="lower")
        plt.colorbar(label="deg C")
        plt.title(title)
        plt.tight_layout()
        out = out_dir / f"{name}_lead{li}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[ok] {out}")


if __name__ == "__main__":
    main()
