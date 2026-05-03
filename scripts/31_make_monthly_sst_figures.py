#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_ANALYSIS_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis"


def _finite_percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanpercentile(finite, q))


def _area_weights(mask: xr.DataArray) -> xr.DataArray:
    lat_weights = np.cos(np.deg2rad(mask["lat"])).astype("float32")
    return lat_weights.broadcast_like(mask).where(mask == 1, 0.0)


def _save(fig: plt.Figure, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(path), "bytes": path.stat().st_size}


def _plot_area_timeseries(df: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    data = df.copy()
    data["time"] = pd.to_datetime(data["time"])

    trend = summary["trend_summary"]["ssta_area_mean"]
    slope = float(trend["slope_c_per_year"])
    intercept = float(trend["intercept_at_start_c"])
    t0 = data["time"].iloc[0]
    years_since_start = (data["time"] - t0).dt.days / 365.2425
    trend_line = intercept + slope * years_since_start

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True)
    axes[0].plot(data["time"], data["sst_area_mean_c"], color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("SST (deg C)")
    axes[0].set_title("South China Sea Area-Mean Monthly SST")
    axes[0].grid(True, color="0.88", linewidth=0.8)

    axes[1].plot(data["time"], data["ssta_area_mean_c"], color="#d62728", linewidth=1.3, label="SSTA")
    axes[1].plot(data["time"], trend_line, color="#222222", linewidth=1.4, linestyle="--", label="Linear trend")
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_ylabel("SSTA (deg C)")
    axes[1].set_title(f"Monthly SST Anomaly Trend: {trend['slope_c_per_decade']:.3f} deg C/decade")
    axes[1].grid(True, color="0.88", linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left")

    missing = summary.get("missing_area_mean_months", [])
    for item in missing:
        ts = pd.Timestamp(item["time"])
        for ax in axes:
            ax.axvspan(ts, ts + pd.offsets.MonthEnd(1), color="0.72", alpha=0.35, linewidth=0)

    axes[1].xaxis.set_major_locator(mdates.YearLocator(base=5))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    return _save(fig, out_dir / "scs_area_mean_sst_ssta_timeseries.png")


def _plot_climatology_cycle(products: xr.Dataset, out_dir: Path) -> dict[str, Any]:
    weights = _area_weights(products["ocean_mask"])
    clim = products["sst_climatology"].weighted(weights).mean(("lat", "lon"), skipna=True).compute()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    months = np.arange(1, 13)
    ax.plot(months, clim.values, color="#1b6f8a", marker="o", linewidth=2.0)
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Climatological SST (deg C)")
    ax.set_title("South China Sea Monthly SST Climatology")
    ax.grid(True, color="0.88", linewidth=0.8)
    fig.tight_layout()
    return _save(fig, out_dir / "scs_monthly_sst_climatology_cycle.png")


def _plot_map(
    da: xr.DataArray,
    out_dir: Path,
    filename: str,
    title: str,
    label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center_zero: bool = False,
) -> dict[str, Any]:
    arr = da.values
    if center_zero:
        bound = max(abs(_finite_percentile(arr, 2.0)), abs(_finite_percentile(arr, 98.0)))
        if np.isfinite(bound) and bound > 0:
            vmin = -bound
            vmax = bound
    else:
        if vmin is None:
            vmin = _finite_percentile(arr, 2.0)
        if vmax is None:
            vmax = _finite_percentile(arr, 98.0)

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    mesh = ax.pcolormesh(da["lon"], da["lat"], arr, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label(label)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title(title)
    ax.set_xlim(float(da["lon"].min()), float(da["lon"].max()))
    ax.set_ylim(float(da["lat"].min()), float(da["lat"].max()))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.78", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    return _save(fig, out_dir / filename)


def make_figures(args: argparse.Namespace) -> None:
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = analysis_dir / "monthly_sst_summary.json"
    csv_path = analysis_dir / "scs_monthly_area_mean_sst_ssta.csv"
    products_path = analysis_dir / "monthly_ssta.zarr"
    trend_path = analysis_dir / "monthly_sst_trend.zarr"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    df = pd.read_csv(csv_path)
    products = xr.open_zarr(products_path, consolidated=True)
    trend = xr.open_zarr(trend_path, consolidated=True)

    manifest: dict[str, Any] = {
        "analysis_dir": str(analysis_dir),
        "output_dir": str(out_dir),
        "figures": {},
        "missing_months": summary.get("missing_area_mean_months", []),
        "trend_summary": summary.get("trend_summary", {}),
    }

    manifest["figures"]["area_mean_timeseries"] = _plot_area_timeseries(df, summary, out_dir)
    manifest["figures"]["climatology_cycle"] = _plot_climatology_cycle(products, out_dir)
    manifest["figures"]["mean_sst_map"] = _plot_map(
        trend["sst_mean_c"],
        out_dir,
        "scs_mean_sst_1991_2021.png",
        "Mean Monthly SST, 1991-2021",
        "SST (deg C)",
        "turbo",
    )
    manifest["figures"]["ssta_trend_map"] = _plot_map(
        trend["ssta_slope_c_per_decade"],
        out_dir,
        "scs_ssta_trend_per_decade.png",
        "Monthly SSTA Linear Trend, 1991-2021",
        "Trend (deg C/decade)",
        "RdBu_r",
        center_zero=True,
    )

    manifest_path = out_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {manifest_path}")
    for item in manifest["figures"].values():
        print(f"[ok] {item['path']} ({item['bytes']} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make monthly South China Sea SST statistical figures.")
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    make_figures(args)


if __name__ == "__main__":
    main()
