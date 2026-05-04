#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    from scipy.stats import linregress
except Exception:  # pragma: no cover - 目标机器已安装 scipy。
    linregress = None


DEFAULT_ANALYSIS_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis"


METRIC_LABELS = {
    "mhw_frequency": ("Frequency", "events/year"),
    "mhw_total_days": ("Total days", "days/year"),
    "mhw_mean_duration": ("Mean duration", "days/event"),
    "mhw_max_duration": ("Max duration", "days"),
    "mhw_max_intensity": ("Max intensity", "deg C"),
    "mhw_cumulative_intensity": ("Cumulative intensity", "deg C days/year"),
    "valid_days": ("Valid days", "days/year"),
}


def _finite_percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanpercentile(finite, q))


def _save(fig: plt.Figure, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(path), "bytes": path.stat().st_size}


def _linear_trend_1d(values: np.ndarray, years: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(values) & np.isfinite(years)
    x = years[mask].astype("float64")
    y = values[mask].astype("float64")
    if len(y) < 3:
        return {
            "n": int(len(y)),
            "slope_per_year": float("nan"),
            "slope_per_decade": float("nan"),
            "linear_change": float("nan"),
            "intercept_at_start": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
            "stderr": float("nan"),
            "mean": float(np.nanmean(y)) if len(y) else float("nan"),
            "first": float(y[0]) if len(y) else float("nan"),
            "last": float(y[-1]) if len(y) else float("nan"),
        }

    if linregress is not None:
        fit = linregress(x, y)
        slope = float(fit.slope)
        return {
            "n": int(len(y)),
            "slope_per_year": slope,
            "slope_per_decade": slope * 10.0,
            "linear_change": slope * float(x[-1] - x[0]),
            "intercept_at_start": float(fit.intercept),
            "r_value": float(fit.rvalue),
            "p_value": float(fit.pvalue),
            "stderr": float(fit.stderr),
            "mean": float(np.nanmean(y)),
            "first": float(y[0]),
            "last": float(y[-1]),
        }

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_value = float(np.sqrt(max(0.0, 1.0 - ss_res / ss_tot))) if ss_tot else 0.0
    return {
        "n": int(len(y)),
        "slope_per_year": float(slope),
        "slope_per_decade": float(slope * 10.0),
        "linear_change": float(slope * (x[-1] - x[0])),
        "intercept_at_start": float(intercept),
        "r_value": r_value,
        "p_value": float("nan"),
        "stderr": float("nan"),
        "mean": float(np.nanmean(y)),
        "first": float(y[0]),
        "last": float(y[-1]),
    }


def _linear_trend_map(da: xr.DataArray, years: xr.DataArray, min_obs: int) -> xr.DataArray:
    valid = xr.where(da.notnull(), 1.0, 0.0)
    n = valid.sum("year")
    n_safe = n.where(n > 0)
    t_mean = (years * valid).sum("year") / n_safe
    y_mean = da.fillna(0.0).sum("year") / n_safe
    dt = years - t_mean
    dy = da - y_mean
    denominator = ((dt**2) * valid).sum("year")
    slope = (dt * dy).where(da.notnull()).sum("year") / denominator.where(denominator > 0)
    return slope.where(n >= min_obs)


def _write_trend_summary(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    years = df["year"].to_numpy(dtype="float64")
    rows = []
    summary: dict[str, Any] = {}
    for metric, (label, unit) in METRIC_LABELS.items():
        if metric not in df:
            continue
        trend = _linear_trend_1d(df[metric].to_numpy(dtype="float64"), years)
        row = {"metric": metric, "label": label, "unit": unit, **trend}
        rows.append(row)
        summary[metric] = row

    trend_df = pd.DataFrame(rows)
    csv_path = out_dir / "mhw_area_mean_trend_summary.csv"
    json_path = out_dir / "mhw_area_mean_trend_summary.json"
    trend_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {csv_path}")
    print(f"[write] {json_path}")
    return trend_df, summary


def _plot_area_timeseries(
    df: pd.DataFrame,
    trend_summary: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    plot_specs = [
        ("mhw_frequency", "#1b6f8a"),
        ("mhw_total_days", "#d95f02"),
        ("mhw_max_intensity", "#7570b3"),
        ("mhw_cumulative_intensity", "#1b9e77"),
    ]
    years = df["year"].to_numpy(dtype="float64")

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.2), sharex=True)
    for ax, (metric, color) in zip(axes, plot_specs, strict=True):
        label, unit = METRIC_LABELS[metric]
        y = df[metric].to_numpy(dtype="float64")
        trend = trend_summary[metric]
        trend_line = trend["intercept_at_start"] + trend["slope_per_year"] * years
        p_value = trend["p_value"]
        p_text = f", p={p_value:.3g}" if np.isfinite(p_value) else ""

        ax.plot(years, y, color=color, linewidth=1.7, marker="o", markersize=3.5, label=label)
        ax.plot(years, trend_line, color="#222222", linewidth=1.3, linestyle="--", label="Linear trend")
        ax.set_ylabel(unit)
        ax.set_title(f"{label}: {trend['slope_per_decade']:.3g} {unit}/decade{p_text}")
        ax.grid(True, color="0.88", linewidth=0.8)
        ax.legend(frameon=False, loc="upper left")

    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(np.arange(int(years.min()), int(years.max()) + 1, 5))
    fig.suptitle("South China Sea Area-Mean Annual Marine Heatwave Metrics", y=0.995)
    fig.tight_layout()
    return _save(fig, out_dir / "scs_mhw_area_mean_annual_timeseries.png")


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

    metrics_path = analysis_dir / "mhw_annual_metrics.zarr"
    csv_path = analysis_dir / "mhw_annual_area_mean.csv"
    summary_path = analysis_dir / "daily_mhw_summary.json"

    metrics = xr.open_zarr(metrics_path, consolidated=True)
    df = pd.read_csv(csv_path)
    daily_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    trend_df, trend_summary = _write_trend_summary(df, out_dir)
    years = xr.DataArray(
        metrics["year"].astype("float64"),
        dims="year",
        coords={"year": metrics["year"]},
        name="year",
    )

    total_days_mean = metrics["mhw_total_days"].where(metrics["ocean_mask"] == 1).mean("year", skipna=True)
    total_days_trend = _linear_trend_map(metrics["mhw_total_days"], years, args.min_obs) * 10.0
    cumulative_intensity_trend = _linear_trend_map(metrics["mhw_cumulative_intensity"], years, args.min_obs) * 10.0
    max_intensity_mean = metrics["mhw_max_intensity"].where(metrics["ocean_mask"] == 1).mean("year", skipna=True)

    manifest: dict[str, Any] = {
        "analysis_dir": str(analysis_dir),
        "output_dir": str(out_dir),
        "source_files": {
            "metrics_zarr": str(metrics_path),
            "area_mean_csv": str(csv_path),
            "daily_mhw_summary": str(summary_path),
        },
        "summary": {
            "year_start": int(df["year"].min()),
            "year_end": int(df["year"].max()),
            "year_count": int(len(df)),
            "min_obs_for_trend_maps": int(args.min_obs),
            "missing_calendar_month_counts": daily_summary.get("missing_calendar_month_counts", {}),
            "zero_valid_ocean_month_counts": daily_summary.get("zero_valid_ocean_month_counts", {}),
        },
        "trend_summary_csv": str(out_dir / "mhw_area_mean_trend_summary.csv"),
        "trend_summary_json": str(out_dir / "mhw_area_mean_trend_summary.json"),
        "figures": {},
    }

    manifest["figures"]["area_mean_timeseries"] = _plot_area_timeseries(df, trend_summary, out_dir)
    manifest["figures"]["mean_total_days_map"] = _plot_map(
        total_days_mean,
        out_dir,
        "scs_mhw_mean_total_days_1991_2021.png",
        "Mean Annual Marine Heatwave Total Days, 1991-2021",
        "MHW total days (days/year)",
        "YlOrRd",
        vmin=0.0,
    )
    manifest["figures"]["total_days_trend_map"] = _plot_map(
        total_days_trend,
        out_dir,
        "scs_mhw_total_days_trend_per_decade.png",
        "Marine Heatwave Total Days Linear Trend, 1991-2021",
        "Trend (days/decade)",
        "RdBu_r",
        center_zero=True,
    )
    manifest["figures"]["cumulative_intensity_trend_map"] = _plot_map(
        cumulative_intensity_trend,
        out_dir,
        "scs_mhw_cumulative_intensity_trend_per_decade.png",
        "Marine Heatwave Cumulative Intensity Linear Trend, 1991-2021",
        "Trend (deg C days/decade)",
        "RdBu_r",
        center_zero=True,
    )
    manifest["figures"]["mean_max_intensity_map"] = _plot_map(
        max_intensity_mean,
        out_dir,
        "scs_mhw_mean_max_intensity_1991_2021.png",
        "Mean Annual Marine Heatwave Maximum Intensity, 1991-2021",
        "MHW max intensity (deg C)",
        "magma",
    )

    manifest["area_mean_trends"] = {
        row["metric"]: {
            key: (None if pd.isna(value) else value)
            for key, value in row.items()
        }
        for row in trend_df.to_dict(orient="records")
    }
    manifest_path = out_dir / "mhw_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {manifest_path}")
    for item in manifest["figures"].values():
        print(f"[ok] {item['path']} ({item['bytes']} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make South China Sea marine heatwave figures and trend tables.")
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--min-obs", type=int, default=25)
    args = parser.parse_args()
    make_figures(args)


if __name__ == "__main__":
    main()
