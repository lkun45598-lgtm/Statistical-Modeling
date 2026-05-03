#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
except Exception:  # pragma: no cover - scipy is available on the target machine.
    linregress = None


DEFAULT_ANALYSIS_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis"
DEFAULT_OUTPUT_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis/subregions"
DEFAULT_PAPER_DIR = "paper"


@dataclass(frozen=True)
class Subregion:
    region_id: str
    name_en: str
    name_zh: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


SUBREGIONS = [
    Subregion("north_shelf", "North Shelf", "北部陆架", 15.0, 25.0, 105.0, 122.0),
    Subregion("central_basin", "Central Basin", "中部海盆", 8.0, 18.0, 110.0, 118.0),
    Subregion("southern_scs", "Southern SCS", "南部南海", -5.0, 8.0, 105.0, 118.0),
    Subregion("western_nearshore", "Western Nearshore", "西部近岸", 5.0, 22.0, 100.0, 110.0),
    Subregion("eastern_offshore", "Eastern Offshore", "东部外海", 5.0, 22.0, 110.0, 125.0),
]

MHW_METRICS = [
    "mhw_frequency",
    "mhw_total_days",
    "mhw_mean_duration",
    "mhw_max_duration",
    "mhw_max_intensity",
    "mhw_cumulative_intensity",
    "valid_days",
]


def _area_weights(mask: xr.DataArray) -> xr.DataArray:
    lat_weights = np.cos(np.deg2rad(mask["lat"])).astype("float32")
    return lat_weights.broadcast_like(mask).where(mask == 1, 0.0)


def _region_mask(mask: xr.DataArray, region: Subregion) -> xr.DataArray:
    lat_ok = (mask["lat"] >= region.lat_min) & (mask["lat"] <= region.lat_max)
    lon_ok = (mask["lon"] >= region.lon_min) & (mask["lon"] <= region.lon_max)
    return (lat_ok.broadcast_like(mask) & lon_ok.broadcast_like(mask) & (mask == 1)).astype("uint8")


def _years_since_start(time_values: np.ndarray) -> np.ndarray:
    times = np.asarray(time_values).astype("datetime64[ns]")
    return ((times - times[0]) / np.timedelta64(1, "D")).astype("float64") / 365.2425


def _linear_trend(values: np.ndarray, years: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(values) & np.isfinite(years)
    x = years[mask].astype("float64")
    y = values[mask].astype("float64")
    if len(y) < 3:
        return {
            "n": int(len(y)),
            "slope_per_year": float("nan"),
            "slope_per_decade": float("nan"),
            "linear_change": float("nan"),
            "intercept": float("nan"),
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
            "intercept": float(fit.intercept),
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
    r_value = float(np.sqrt(max(0.0, 1.0 - ss_res / ss_tot))) if ss_tot else float("nan")
    return {
        "n": int(len(y)),
        "slope_per_year": float(slope),
        "slope_per_decade": float(slope * 10.0),
        "linear_change": float(slope * float(x[-1] - x[0])),
        "intercept": float(intercept),
        "r_value": r_value,
        "p_value": float("nan"),
        "stderr": float("nan"),
        "mean": float(np.nanmean(y)),
        "first": float(y[0]),
        "last": float(y[-1]),
    }


def _sig_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _write_table(summary_df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"分区 & SSTA趋势 & MHW总天数趋势 & MHW累计强度趋势 & 显著性判断 \\",
        r" & ($^\circ$C/10年) & (天/10年) & ($^\circ$C$\cdot$天/10年) &  \\",
        r"\hline",
    ]
    for row in summary_df.itertuples(index=False):
        sig = (
            f"SSTA {row.ssta_sig}; "
            f"天数 {row.mhw_total_days_sig}; "
            f"强度 {row.mhw_cumulative_intensity_sig}"
        )
        lines.append(
            f"{row.region_name_zh} & "
            f"{_fmt(row.ssta_slope_per_decade)} & "
            f"{_fmt(row.mhw_total_days_slope_per_decade)} & "
            f"{_fmt(row.mhw_cumulative_intensity_slope_per_decade)} & "
            f"{sig} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path}")


def _save(fig: plt.Figure, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(path), "bytes": path.stat().st_size}


def _plot_subregions(
    monthly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    colors = {
        "north_shelf": "#1b6f8a",
        "central_basin": "#d95f02",
        "southern_scs": "#1b9e77",
        "western_nearshore": "#7570b3",
        "eastern_offshore": "#a6761d",
    }
    name_map = dict(zip(summary_df["region_id"], summary_df["region_name_en"], strict=True))

    monthly = monthly_df.copy()
    monthly["time"] = pd.to_datetime(monthly["time"])
    monthly["ssta_smooth"] = monthly.groupby("region_id")["ssta_c"].transform(
        lambda s: s.rolling(12, center=True, min_periods=6).mean()
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))

    ax = axes[0, 0]
    for region_id, group in monthly.groupby("region_id", sort=False):
        ax.plot(group["time"], group["ssta_smooth"], color=colors[region_id], linewidth=1.5, label=name_map[region_id])
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_title("(a) 12-month smoothed SSTA")
    ax.set_ylabel("SSTA (deg C)")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncols=2)

    ax = axes[0, 1]
    for region_id, group in annual_df.groupby("region_id", sort=False):
        ax.plot(
            group["year"],
            group["mhw_total_days"],
            color=colors[region_id],
            linewidth=1.5,
            marker="o",
            markersize=3.0,
            label=name_map[region_id],
        )
    ax.set_title("(b) Annual MHW total days")
    ax.set_ylabel("days/year")
    ax.grid(True, color="0.88", linewidth=0.8)

    x = np.arange(len(summary_df))
    ax = axes[1, 0]
    ax.bar(x, summary_df["ssta_slope_per_decade"], color=[colors[r] for r in summary_df["region_id"]])
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_title("(c) SSTA trend by subregion")
    ax.set_ylabel("deg C/decade")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["region_name_en"], rotation=25, ha="right")
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)

    ax = axes[1, 1]
    ax.bar(x, summary_df["mhw_total_days_slope_per_decade"], color=[colors[r] for r in summary_df["region_id"]])
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_title("(d) MHW total-days trend by subregion")
    ax.set_ylabel("days/decade")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["region_name_en"], rotation=25, ha="right")
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)

    fig.tight_layout()
    return _save(fig, path)


def _weighted_mean(da: xr.DataArray, weights: xr.DataArray, dims: tuple[str, str] = ("lat", "lon")) -> xr.DataArray:
    return da.weighted(weights).mean(dims, skipna=True)


def build_subregion_analysis(args: argparse.Namespace) -> None:
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly = xr.open_zarr(analysis_dir / "monthly_ssta.zarr", consolidated=True)
    mhw = xr.open_zarr(analysis_dir / "mhw_annual_metrics.zarr", consolidated=True)

    monthly_mask = monthly["ocean_mask"]
    mhw_mask = mhw["ocean_mask"]
    monthly_weights_all = _area_weights(monthly_mask)
    mhw_weights_all = _area_weights(mhw_mask)

    monthly_years = _years_since_start(monthly["time"].values)
    annual_years = mhw["year"].values.astype("float64")

    monthly_records: list[pd.DataFrame] = []
    annual_records: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []

    for region in SUBREGIONS:
        print(f"[compute] {region.name_en}")
        m_mask = _region_mask(monthly_mask, region)
        a_mask = _region_mask(mhw_mask, region)
        m_weights = monthly_weights_all.where(m_mask == 1, 0.0)
        a_weights = mhw_weights_all.where(a_mask == 1, 0.0)

        ssta_mean = _weighted_mean(monthly["ssta"], m_weights).compute()
        month_time = pd.DatetimeIndex(pd.to_datetime(monthly["time"].values))
        region_monthly = pd.DataFrame(
            {
                "region_id": region.region_id,
                "region_name_en": region.name_en,
                "region_name_zh": region.name_zh,
                "time": month_time.strftime("%Y-%m-%d"),
                "year": month_time.year,
                "month": month_time.month,
                "ssta_c": ssta_mean.values.astype("float64"),
            }
        )
        monthly_records.append(region_monthly)

        annual_data: dict[str, Any] = {
            "region_id": region.region_id,
            "region_name_en": region.name_en,
            "region_name_zh": region.name_zh,
            "year": mhw["year"].values.astype("int64"),
        }
        for metric in MHW_METRICS:
            annual_data[metric] = _weighted_mean(mhw[metric], a_weights).compute().values.astype("float64")
        region_annual = pd.DataFrame(annual_data)
        annual_records.append(region_annual)

        ssta_trend = _linear_trend(region_monthly["ssta_c"].to_numpy(), monthly_years)
        frequency_trend = _linear_trend(region_annual["mhw_frequency"].to_numpy(), annual_years)
        total_days_trend = _linear_trend(region_annual["mhw_total_days"].to_numpy(), annual_years)
        cumulative_trend = _linear_trend(region_annual["mhw_cumulative_intensity"].to_numpy(), annual_years)
        max_intensity_trend = _linear_trend(region_annual["mhw_max_intensity"].to_numpy(), annual_years)

        ocean_cell_count = int(a_mask.sum().compute().item())
        weight_sum = float(a_weights.sum().compute().item())
        summary_records.append(
            {
                "region_id": region.region_id,
                "region_name_en": region.name_en,
                "region_name_zh": region.name_zh,
                "lat_min": region.lat_min,
                "lat_max": region.lat_max,
                "lon_min": region.lon_min,
                "lon_max": region.lon_max,
                "ocean_cell_count": ocean_cell_count,
                "area_weight_sum": weight_sum,
                "ssta_slope_per_decade": ssta_trend["slope_per_decade"],
                "ssta_p_value": ssta_trend["p_value"],
                "ssta_n": ssta_trend["n"],
                "ssta_sig": _sig_label(ssta_trend["p_value"]),
                "mhw_frequency_slope_per_decade": frequency_trend["slope_per_decade"],
                "mhw_frequency_p_value": frequency_trend["p_value"],
                "mhw_frequency_n": frequency_trend["n"],
                "mhw_frequency_sig": _sig_label(frequency_trend["p_value"]),
                "mhw_total_days_slope_per_decade": total_days_trend["slope_per_decade"],
                "mhw_total_days_p_value": total_days_trend["p_value"],
                "mhw_total_days_n": total_days_trend["n"],
                "mhw_total_days_sig": _sig_label(total_days_trend["p_value"]),
                "mhw_cumulative_intensity_slope_per_decade": cumulative_trend["slope_per_decade"],
                "mhw_cumulative_intensity_p_value": cumulative_trend["p_value"],
                "mhw_cumulative_intensity_n": cumulative_trend["n"],
                "mhw_cumulative_intensity_sig": _sig_label(cumulative_trend["p_value"]),
                "mhw_max_intensity_slope_per_decade": max_intensity_trend["slope_per_decade"],
                "mhw_max_intensity_p_value": max_intensity_trend["p_value"],
                "mhw_max_intensity_n": max_intensity_trend["n"],
                "mhw_max_intensity_sig": _sig_label(max_intensity_trend["p_value"]),
            }
        )

    monthly_df = pd.concat(monthly_records, ignore_index=True)
    annual_df = pd.concat(annual_records, ignore_index=True)
    summary_df = pd.DataFrame(summary_records)

    monthly_csv = output_dir / "subregion_monthly_ssta.csv"
    annual_csv = output_dir / "subregion_mhw_annual.csv"
    summary_csv = output_dir / "subregion_summary.csv"
    summary_json = output_dir / "subregion_summary.json"

    monthly_df.to_csv(monthly_csv, index=False)
    annual_df.to_csv(annual_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "analysis_dir": str(analysis_dir),
                "output_dir": str(output_dir),
                "subregions": summary_records,
                "notes": {
                    "area_weighting": "cos(latitude) weights with ocean_mask == 1",
                    "missing_policy": "missing monthly values remain NaN and are excluded from trend fits",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[write] {monthly_csv}")
    print(f"[write] {annual_csv}")
    print(f"[write] {summary_csv}")
    print(f"[write] {summary_json}")

    table_path = paper_dir / "tables" / "subregion_summary_table.tex"
    figure_path = paper_dir / "figures" / "subregion_trend_panel.png"
    _write_table(summary_df, table_path)
    figure_info = _plot_subregions(monthly_df, annual_df, summary_df, figure_path)
    print(f"[ok] {figure_info['path']} ({figure_info['bytes']} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build South China Sea subregion SST and MHW summaries.")
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-dir", default=DEFAULT_PAPER_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Accepted for workflow compatibility; files are overwritten.")
    args = parser.parse_args()
    build_subregion_analysis(args)


if __name__ == "__main__":
    main()
