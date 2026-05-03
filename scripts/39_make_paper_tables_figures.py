#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_PAPER_DIR = "paper"


FIGURES = [
    ("fig:workflow", "figures/workflow_diagram.png", "研究技术路线"),
    ("fig:sst_series", "figures/sst_area_mean_timeseries.png", "南海区域平均月尺度 SST 与 SSTA 时间序列"),
    ("fig:sst_climatology", "figures/sst_climatology_cycle.png", "南海区域平均月尺度 SST 气候态季节循环"),
    ("fig:sst_maps", "figures/sst_mean_map.png", "南海多年平均 SST 空间分布"),
    ("fig:ssta_trend", "figures/ssta_trend_map.png", "南海月尺度 SSTA 趋势空间分布"),
    ("fig:subregion", "figures/subregion_trend_panel.png", "南海分区增暖与海洋热浪变化"),
    ("fig:robust", "figures/robust_trend_comparison.png", "主要指标趋势稳健性比较"),
    ("fig:mhw_series", "figures/mhw_area_mean_timeseries.png", "南海区域平均年度海洋热浪指标时间序列"),
    ("fig:mhw_maps", "figures/mhw_mean_total_days_map.png", "多年平均年度 MHW 总天数空间分布"),
    ("fig:mhw_total_days_trend", "figures/mhw_total_days_trend_map.png", "年度 MHW 总天数趋势空间分布"),
    ("fig:mhw_cumulative_trend", "figures/mhw_cumulative_intensity_trend_map.png", "年度 MHW 累计强度趋势空间分布"),
    ("fig:threshold_sensitivity", "figures/mhw_threshold_sensitivity.png", "MHW 阈值敏感性检验"),
    ("fig:hri", "figures/heat_risk_index_map.png", "南海相对综合海洋热风险指数空间分布"),
    ("fig:driver_heatmap", "figures/driver_correlation_heatmap.png", "南海 SSTA 与 MHW 指标的驱动因子相关热图"),
    ("fig:monthly_driver_fit", "figures/monthly_ssta_driver_fit.png", "月尺度 SSTA 驱动因子回归拟合"),
    ("fig:annual_driver_fit", "figures/annual_mhw_driver_fit.png", "年度 MHW 总天数驱动因子回归拟合"),
]

TABLES = [
    ("tab:data", "tables/data_summary_table.tex", "数据来源与变量说明"),
    ("tab:trend", "tables/trend_summary_table.tex", "区域平均增暖和海洋热浪风险趋势摘要"),
    ("tab:subregion", "tables/subregion_summary_table.tex", "南海分区增暖与海洋热浪趋势摘要"),
    ("tab:robust", "tables/robust_trend_table.tex", "主要指标趋势稳健性检验"),
    ("tab:threshold", "tables/threshold_sensitivity_table.tex", "MHW 阈值敏感性检验"),
    ("tab:hri", "tables/heat_risk_index_table.tex", "南海分区相对综合海洋热风险指数"),
    ("tab:driver_lag", "tables/monthly_driver_best_lag_table.tex", "月尺度 SSTA 与驱动因子的最强滞后相关"),
    ("tab:monthly_driver", "tables/monthly_driver_regression_table.tex", "月尺度 SSTA 标准化解释模型"),
    ("tab:annual_driver", "tables/annual_driver_regression_table.tex", "年度 MHW 总天数标准化解释模型"),
    ("tab:driver_summary", "tables/driver_model_summary_table.tex", "驱动因子解释模型拟合优度"),
]


def _draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 2.2, height: float = 0.72) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1,
        edgecolor="#2f3b52",
        facecolor="#eef3f8",
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9.5, color="#182033")


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.0, color="#374151")
    ax.add_patch(arrow)


def make_workflow(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        ((0.5, 4.55), "OSTIA daily SST\n1991-2021"),
        ((3.1, 4.55), "SCS crop\nquality control"),
        ((5.7, 4.55), "Monthly SSTA\ntrend model"),
        ((8.3, 4.55), "Daily MHW\nevent recognition"),
        ((3.1, 2.75), "Subregional\nstatistics"),
        ((5.7, 2.75), "Robustness\nand sensitivity"),
        ((8.3, 2.75), "Composite heat-risk\nindex"),
        ((5.7, 0.95), "Driver explanation\nmodel"),
        ((8.3, 0.95), "Monitoring and\nmanagement use"),
    ]
    for xy, text in boxes:
        _draw_box(ax, xy, text)

    arrows = [
        ((2.7, 4.91), (3.08, 4.91)),
        ((5.3, 4.91), (5.68, 4.91)),
        ((7.9, 4.91), (8.28, 4.91)),
        ((6.8, 4.55), (4.2, 3.47)),
        ((6.8, 4.55), (6.8, 3.47)),
        ((9.4, 4.55), (9.4, 3.47)),
        ((5.3, 3.11), (5.68, 3.11)),
        ((7.9, 3.11), (8.28, 3.11)),
        ((6.8, 2.75), (6.8, 1.67)),
        ((9.4, 2.75), (9.4, 1.67)),
        ((7.9, 1.31), (8.28, 1.31)),
    ]
    for start, end in arrows:
        _draw_arrow(ax, start, end)

    ax.text(
        6.0,
        5.72,
        "Reproducible statistical modeling workflow",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(path), "bytes": path.stat().st_size}


def build_manifest(args: argparse.Namespace) -> None:
    paper_dir = Path(args.paper_dir)
    workflow_info = make_workflow(paper_dir / "figures" / "workflow_diagram.png")

    figures: list[dict[str, Any]] = []
    for fig_id, rel_path, caption in FIGURES:
        path = paper_dir / rel_path
        figures.append(
            {
                "id": fig_id,
                "path": str(path),
                "caption": caption,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    tables: list[dict[str, Any]] = []
    for table_id, rel_path, caption in TABLES:
        path = paper_dir / rel_path
        tables.append(
            {
                "id": table_id,
                "path": str(path),
                "caption": caption,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    manifest = {
        "paper_dir": str(paper_dir),
        "workflow_diagram": workflow_info,
        "figures": figures,
        "tables": tables,
        "missing": [item for item in figures + tables if not item["exists"]],
    }
    manifest_path = paper_dir / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {paper_dir / 'figures' / 'workflow_diagram.png'}")
    print(f"[write] {manifest_path}")
    if manifest["missing"]:
        raise FileNotFoundError(f"Missing paper assets: {manifest['missing']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final paper workflow figure and asset manifest.")
    parser.add_argument("--paper-dir", default=DEFAULT_PAPER_DIR)
    args = parser.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
