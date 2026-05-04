#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    from scipy.stats import kendalltau, linregress, theilslopes
except Exception as exc:  # pragma: no cover - 目标机器已安装 scipy。
    raise RuntimeError("scipy is required for robustness analysis.") from exc


DEFAULT_ANALYSIS_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis"
DEFAULT_OUTPUT_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis/robustness"
DEFAULT_DAILY_ZARR = "/data1/user/lz/osita_data/scs_5s25n/ostia_scs_daily.zarr"
DEFAULT_PAPER_DIR = "paper"


@dataclass(frozen=True)
class TrendTarget:
    key: str
    label_zh: str
    label_en: str
    unit: str
    source: str
    column: str


TREND_TARGETS = [
    TrendTarget("monthly_ssta", "区域平均SSTA", "Monthly SSTA", "$^\\circ$C/10年", "monthly", "ssta_area_mean_c"),
    TrendTarget("mhw_frequency", "MHW频次", "MHW frequency", "次/10年", "annual", "mhw_frequency"),
    TrendTarget("mhw_total_days", "MHW总天数", "MHW total days", "天/10年", "annual", "mhw_total_days"),
    TrendTarget(
        "mhw_cumulative_intensity",
        "MHW累计强度",
        "MHW cumulative intensity",
        "$^\\circ$C$\\cdot$天/10年",
        "annual",
        "mhw_cumulative_intensity",
    ),
    TrendTarget(
        "mhw_max_intensity",
        "MHW最大强度",
        "MHW max intensity",
        "$^\\circ$C/10年",
        "annual",
        "mhw_max_intensity",
    ),
]


def _years_since_start(time_values: np.ndarray) -> np.ndarray:
    times = np.asarray(time_values).astype("datetime64[ns]")
    return ((times - times[0]) / np.timedelta64(1, "D")).astype("float64") / 365.2425


def _trend_stats(values: np.ndarray, years: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(values) & np.isfinite(years)
    x = years[mask].astype("float64")
    y = values[mask].astype("float64")
    if len(y) < 3:
        return {
            "n": int(len(y)),
            "ols_slope_per_decade": float("nan"),
            "ols_p_value": float("nan"),
            "theil_sen_slope_per_decade": float("nan"),
            "theil_sen_low_per_decade": float("nan"),
            "theil_sen_high_per_decade": float("nan"),
            "kendall_tau": float("nan"),
            "kendall_p_value": float("nan"),
            "direction": "insufficient",
        }

    fit = linregress(x, y)
    ts = theilslopes(y, x, 0.95)
    kt = kendalltau(x, y, nan_policy="omit")
    slope = float(fit.slope * 10.0)
    ts_slope = float(ts.slope * 10.0)
    if np.sign(slope) == np.sign(ts_slope) and np.isfinite(kt.pvalue) and kt.pvalue < 0.05:
        direction = "consistent significant increase" if slope > 0 else "consistent significant decrease"
    elif np.sign(slope) == np.sign(ts_slope):
        direction = "consistent direction"
    else:
        direction = "mixed direction"

    return {
        "n": int(len(y)),
        "ols_slope_per_decade": slope,
        "ols_p_value": float(fit.pvalue),
        "ols_r_value": float(fit.rvalue),
        "ols_stderr_per_decade": float(fit.stderr * 10.0),
        "theil_sen_slope_per_decade": ts_slope,
        "theil_sen_low_per_decade": float(ts.low_slope * 10.0),
        "theil_sen_high_per_decade": float(ts.high_slope * 10.0),
        "kendall_tau": float(kt.statistic),
        "kendall_p_value": float(kt.pvalue),
        "direction": direction,
    }


def _sig_text(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "$p<0.001$"
    if p_value < 0.01:
        return "$p<0.01$"
    if p_value < 0.05:
        return "$p<0.05$"
    return "$p\\geq0.05$"


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _conclusion_zh(direction: str) -> str:
    if direction == "consistent significant increase":
        return "显著增加"
    if direction == "consistent significant decrease":
        return "显著降低"
    if direction == "consistent direction":
        return "方向一致"
    if direction == "mixed direction":
        return "方向不一致"
    return "样本不足"


def _load_main_series(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(analysis_dir / "scs_monthly_area_mean_sst_ssta.csv")
    monthly["time"] = pd.to_datetime(monthly["time"])
    annual = pd.read_csv(analysis_dir / "mhw_annual_area_mean.csv")
    return monthly, annual


def _build_robust_trends(analysis_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly, annual = _load_main_series(analysis_dir)
    monthly_years = _years_since_start(monthly["time"].to_numpy(dtype="datetime64[ns]"))
    annual_years = annual["year"].to_numpy(dtype="float64")

    rows: list[dict[str, Any]] = []
    for target in TREND_TARGETS:
        if target.source == "monthly":
            values = monthly[target.column].to_numpy(dtype="float64")
            years = monthly_years
        else:
            values = annual[target.column].to_numpy(dtype="float64")
            years = annual_years
        stats = _trend_stats(values, years)
        rows.append(
            {
                "target": target.key,
                "label_zh": target.label_zh,
                "label_en": target.label_en,
                "unit": target.unit,
                **stats,
            }
        )

    robust_df = pd.DataFrame(rows)

    complete = annual.loc[annual["valid_days"] >= 360].copy()
    complete_years = complete["year"].to_numpy(dtype="float64")
    missing_rows: list[dict[str, Any]] = []
    for target in TREND_TARGETS:
        if target.source != "annual":
            continue
        main_row = robust_df.loc[robust_df["target"] == target.key].iloc[0]
        complete_stats = _trend_stats(complete[target.column].to_numpy(dtype="float64"), complete_years)
        missing_rows.append(
            {
                "target": target.key,
                "label_zh": target.label_zh,
                "main_n": int(main_row["n"]),
                "complete_year_n": int(complete_stats["n"]),
                "main_ols_slope_per_decade": float(main_row["ols_slope_per_decade"]),
                "complete_year_ols_slope_per_decade": complete_stats["ols_slope_per_decade"],
                "main_kendall_p_value": float(main_row["kendall_p_value"]),
                "complete_year_kendall_p_value": complete_stats["kendall_p_value"],
                "direction_consistent": bool(
                    np.sign(main_row["ols_slope_per_decade"])
                    == np.sign(complete_stats["ols_slope_per_decade"])
                ),
            }
        )

    missing_df = pd.DataFrame(missing_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    robust_df.to_csv(output_dir / "robust_trend_summary.csv", index=False)
    missing_df.to_csv(output_dir / "missing_month_sensitivity.csv", index=False)
    print(f"[write] {output_dir / 'robust_trend_summary.csv'}")
    print(f"[write] {output_dir / 'missing_month_sensitivity.csv'}")
    return robust_df, missing_df


def _threshold_output_dir(output_dir: Path, percentile: float) -> Path:
    return output_dir / f"mhw_percentile_{int(percentile)}"


def _ensure_threshold_product(args: argparse.Namespace, percentile: float) -> Path:
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir)
    if float(percentile) == 90.0:
        return analysis_dir

    product_dir = _threshold_output_dir(output_dir, percentile)
    annual_csv = product_dir / "mhw_annual_area_mean.csv"
    metrics_zarr = product_dir / "mhw_annual_metrics.zarr"
    if annual_csv.exists() and metrics_zarr.exists() and not args.rebuild_threshold_products:
        return product_dir

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("32_build_daily_mhw_products.py")),
        "--input-zarr",
        str(args.daily_zarr),
        "--output-dir",
        str(product_dir),
        "--percentile",
        str(float(percentile)),
        "--lat-block-size",
        str(int(args.lat_block_size)),
        "--workers",
        str(int(args.threshold_workers)),
        "--overwrite",
    ]
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return product_dir


def _build_threshold_sensitivity(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    reference_signs: tuple[float, float] | None = None

    for percentile in args.thresholds:
        product_dir = _ensure_threshold_product(args, float(percentile))
        annual = pd.read_csv(product_dir / "mhw_annual_area_mean.csv")
        years = annual["year"].to_numpy(dtype="float64")
        frequency = annual["mhw_frequency"].to_numpy(dtype="float64")
        total_days = annual["mhw_total_days"].to_numpy(dtype="float64")
        cumulative = annual["mhw_cumulative_intensity"].to_numpy(dtype="float64")
        total_trend = _trend_stats(total_days, years)
        cumulative_trend = _trend_stats(cumulative, years)
        frequency_trend = _trend_stats(frequency, years)

        if float(percentile) == 90.0:
            reference_signs = (
                float(np.sign(total_trend["ols_slope_per_decade"])),
                float(np.sign(cumulative_trend["ols_slope_per_decade"])),
            )

        rows.append(
            {
                "percentile": float(percentile),
                "product_dir": str(product_dir),
                "mean_frequency": float(np.nanmean(frequency)),
                "mean_total_days": float(np.nanmean(total_days)),
                "frequency_trend_per_decade": frequency_trend["ols_slope_per_decade"],
                "total_days_trend_per_decade": total_trend["ols_slope_per_decade"],
                "total_days_kendall_p_value": total_trend["kendall_p_value"],
                "cumulative_intensity_trend_per_decade": cumulative_trend["ols_slope_per_decade"],
                "cumulative_intensity_kendall_p_value": cumulative_trend["kendall_p_value"],
            }
        )

    df = pd.DataFrame(rows).sort_values("percentile")
    if reference_signs is None:
        reference = df.loc[df["percentile"].sub(90.0).abs().idxmin()]
        reference_signs = (
            float(np.sign(reference["total_days_trend_per_decade"])),
            float(np.sign(reference["cumulative_intensity_trend_per_decade"])),
        )
    df["conclusion_consistent"] = (
        (np.sign(df["total_days_trend_per_decade"]) == reference_signs[0])
        & (np.sign(df["cumulative_intensity_trend_per_decade"]) == reference_signs[1])
    )
    path = output_dir / "mhw_threshold_sensitivity.csv"
    df.to_csv(path, index=False)
    print(f"[write] {path}")
    return df


def _write_robust_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"指标 & 单位 & OLS趋势 & Theil-Sen趋势 & Kendall $\tau$ & $p$ 值 & 结论 \\",
        r"\hline",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.label_zh} & "
            f"{row.unit} & "
            f"{_fmt(row.ols_slope_per_decade)} & "
            f"{_fmt(row.theil_sen_slope_per_decade)} & "
            f"{_fmt(row.kendall_tau)} & "
            f"{_sig_text(row.kendall_p_value)} & "
            f"{_conclusion_zh(row.direction)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path}")


def _write_threshold_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"阈值 & 平均频次 & 平均总天数 & 总天数趋势 & 累计强度趋势 & 结论是否一致 \\",
        r" & (次/年) & (天/年) & (天/10年) & ($^\circ$C$\cdot$天/10年) & \\",
        r"\hline",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{int(row.percentile)}\\% & "
            f"{_fmt(row.mean_frequency)} & "
            f"{_fmt(row.mean_total_days)} & "
            f"{_fmt(row.total_days_trend_per_decade)} & "
            f"{_fmt(row.cumulative_intensity_trend_per_decade)} & "
            f"{'一致' if row.conclusion_consistent else '不一致'} \\\\"
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


def _plot_robust_trends(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    labels = df["label_en"].tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(x - width / 2, df["ols_slope_per_decade"], width, label="OLS", color="#1b6f8a")
    ax.bar(x + width / 2, df["theil_sen_slope_per_decade"], width, label="Theil-Sen", color="#d95f02")
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Trend per decade in native units")
    ax.set_title("Robust Trend Comparison")
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def _plot_threshold_sensitivity(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    x = df["percentile"].to_numpy(dtype="float64")
    ax1.plot(
        x,
        df["total_days_trend_per_decade"],
        color="#d95f02",
        marker="o",
        linewidth=2.0,
        label="MHW total-days trend",
    )
    ax1.set_xlabel("MHW threshold percentile")
    ax1.set_ylabel("Total-days trend (days/decade)", color="#d95f02")
    ax1.tick_params(axis="y", labelcolor="#d95f02")
    ax1.grid(True, color="0.88", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["cumulative_intensity_trend_per_decade"],
        color="#1b6f8a",
        marker="s",
        linewidth=2.0,
        label="Cumulative-intensity trend",
    )
    ax2.set_ylabel("Cumulative-intensity trend (deg C days/decade)", color="#1b6f8a")
    ax2.tick_params(axis="y", labelcolor="#1b6f8a")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")
    ax1.set_title("MHW Threshold Sensitivity")
    fig.tight_layout()
    return _save(fig, path)


def build_robustness(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    robust_df, missing_df = _build_robust_trends(Path(args.analysis_dir), output_dir)
    threshold_df = _build_threshold_sensitivity(args)

    _write_robust_table(robust_df, paper_dir / "tables" / "robust_trend_table.tex")
    _write_threshold_table(threshold_df, paper_dir / "tables" / "threshold_sensitivity_table.tex")
    robust_fig = _plot_robust_trends(robust_df, paper_dir / "figures" / "robust_trend_comparison.png")
    threshold_fig = _plot_threshold_sensitivity(threshold_df, paper_dir / "figures" / "mhw_threshold_sensitivity.png")

    summary = {
        "analysis_dir": str(args.analysis_dir),
        "output_dir": str(output_dir),
        "daily_zarr": str(args.daily_zarr),
        "thresholds": [float(v) for v in args.thresholds],
        "robust_trends": robust_df.to_dict(orient="records"),
        "missing_month_sensitivity": missing_df.to_dict(orient="records"),
        "threshold_sensitivity": threshold_df.to_dict(orient="records"),
        "figures": {
            "robust_trend_comparison": robust_fig,
            "mhw_threshold_sensitivity": threshold_fig,
        },
    }
    summary_path = output_dir / "robustness_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build robustness checks for South China Sea SST/MHW paper.")
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--daily-zarr", default=DEFAULT_DAILY_ZARR)
    parser.add_argument("--paper-dir", default=DEFAULT_PAPER_DIR)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[85.0, 90.0, 95.0])
    parser.add_argument("--lat-block-size", type=int, default=1)
    parser.add_argument("--threshold-workers", type=int, default=128)
    parser.add_argument("--rebuild-threshold-products", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for workflow compatibility; summary files are overwritten.")
    args = parser.parse_args()
    build_robustness(args)


if __name__ == "__main__":
    main()
