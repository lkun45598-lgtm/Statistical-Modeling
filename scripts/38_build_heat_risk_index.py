#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from numcodecs import Blosc


DEFAULT_ANALYSIS_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis"
DEFAULT_OUTPUT_DIR = "/data1/user/lz/osita_data/scs_5s25n/analysis/heat_risk"
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

COMPONENT_LABELS = {
    "ssta_trend_scaled": "增暖趋势",
    "mhw_total_days_mean_scaled": "热浪暴露均值",
    "mhw_total_days_trend_scaled": "总天数趋势",
    "mhw_cumulative_intensity_trend_scaled": "累计强度趋势",
}


def _maybe_remove(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _area_weights(mask: xr.DataArray) -> xr.DataArray:
    lat_weights = np.cos(np.deg2rad(mask["lat"])).astype("float32")
    return lat_weights.broadcast_like(mask).where(mask == 1, 0.0)


def _region_mask(mask: xr.DataArray, region: Subregion) -> xr.DataArray:
    lat_ok = (mask["lat"] >= region.lat_min) & (mask["lat"] <= region.lat_max)
    lon_ok = (mask["lon"] >= region.lon_min) & (mask["lon"] <= region.lon_max)
    return (lat_ok.broadcast_like(mask) & lon_ok.broadcast_like(mask) & (mask == 1)).astype("uint8")


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


def _scale_component(da: xr.DataArray, ocean_mask: xr.DataArray) -> tuple[xr.DataArray, dict[str, Any]]:
    valid = da.where(ocean_mask == 1)
    valid_min = float(valid.min(skipna=True).compute().item())
    valid_max = float(valid.max(skipna=True).compute().item())
    if not np.isfinite(valid_min) or not np.isfinite(valid_max) or np.isclose(valid_max, valid_min):
        scaled = xr.full_like(da, np.nan, dtype="float32")
        return scaled, {
            "valid": False,
            "min": valid_min,
            "max": valid_max,
            "reason": "no finite range",
        }
    scaled = ((da - valid_min) / (valid_max - valid_min)).where(ocean_mask == 1).clip(0.0, 1.0).astype("float32")
    return scaled, {
        "valid": True,
        "min": valid_min,
        "max": valid_max,
        "reason": "",
    }


def _weighted_mean(da: xr.DataArray, weights: xr.DataArray) -> float:
    value = da.weighted(weights).mean(("lat", "lon"), skipna=True).compute().item()
    return float(value) if np.isfinite(value) else float("nan")


def _weighted_share(mask: xr.DataArray, weights: xr.DataArray) -> float:
    numerator = (mask.astype("float32") * weights).sum(("lat", "lon"), skipna=True).compute().item()
    denominator = weights.sum(("lat", "lon"), skipna=True).compute().item()
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _application_text(mean_hri: float, high_share: float) -> str:
    if high_share >= 0.35 or mean_hri >= 0.66:
        return "优先监测"
    if high_share >= 0.15 or mean_hri >= 0.50:
        return "重点跟踪"
    return "常规监测"


def _write_zarr(ds: xr.Dataset, path: Path, overwrite: bool) -> None:
    _maybe_remove(path, overwrite)
    chunks = {dim: size for dim, size in {"lat": 100, "lon": 100}.items() if dim in ds.dims}
    if chunks:
        ds = ds.chunk(chunks)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        var_chunks = tuple(chunks.get(dim, ds.sizes[dim]) for dim in da.dims)
        enc: dict[str, Any] = {"compressor": compressor}
        if var_chunks:
            enc["chunks"] = var_chunks
        encoding[name] = enc
    print(f"[write] {path}")
    ds.to_zarr(path, mode="w", consolidated=True, encoding=encoding)


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _write_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"分区 & 平均HRI & 高风险格点占比 & 主导风险来源 & 应用含义 \\",
        r"\hline",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.region_name_zh} & "
            f"{_fmt(row.mean_hri)} & "
            f"{_fmt(row.high_risk_share * 100.0, 1)}\\% & "
            f"{row.dominant_component_zh} & "
            f"{row.application_meaning} \\\\"
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


def _plot_hri(hri: xr.DataArray, path: Path) -> dict[str, Any]:
    cmap = ListedColormap(["#4daf4a", "#fdae61", "#d73027"])
    norm = BoundaryNorm([0.0, 0.33, 0.66, 1.0], cmap.N)
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    mesh = ax.pcolormesh(hri["lon"], hri["lat"], hri.values, cmap=cmap, norm=norm, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.86, pad=0.02, ticks=[0.165, 0.495, 0.83])
    cbar.ax.set_yticklabels(["Low", "Medium", "High"])
    cbar.set_label("Relative marine heat-risk index")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("Composite Marine Heat-Risk Index")
    ax.set_xlim(float(hri["lon"].min()), float(hri["lon"].max()))
    ax.set_ylim(float(hri["lat"].min()), float(hri["lat"].max()))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.78", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    return _save(fig, path)


def build_heat_risk_index(args: argparse.Namespace) -> None:
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trend = xr.open_zarr(analysis_dir / "monthly_sst_trend.zarr", consolidated=True)
    mhw = xr.open_zarr(analysis_dir / "mhw_annual_metrics.zarr", consolidated=True)
    ocean_mask = mhw["ocean_mask"]
    years = xr.DataArray(
        mhw["year"].astype("float64"),
        dims="year",
        coords={"year": mhw["year"]},
        name="year",
    )

    components_raw = {
        "ssta_trend": trend["ssta_slope_c_per_decade"].where(ocean_mask == 1),
        "mhw_total_days_mean": mhw["mhw_total_days"].where(ocean_mask == 1).mean("year", skipna=True),
        "mhw_total_days_trend": (_linear_trend_map(mhw["mhw_total_days"], years, args.min_obs) * 10.0).where(
            ocean_mask == 1
        ),
        "mhw_cumulative_intensity_trend": (
            _linear_trend_map(mhw["mhw_cumulative_intensity"], years, args.min_obs) * 10.0
        ).where(ocean_mask == 1),
    }

    scaled_components: dict[str, xr.DataArray] = {}
    scale_metadata: dict[str, Any] = {}
    for name, da in components_raw.items():
        scaled, meta = _scale_component(da, ocean_mask)
        scaled_name = f"{name}_scaled"
        scaled_components[scaled_name] = scaled
        scale_metadata[scaled_name] = meta

    valid_names = [name for name, meta in scale_metadata.items() if meta["valid"]]
    if not valid_names:
        raise ValueError("No finite HRI components are available.")
    weight_value = 1.0 / len(valid_names)
    component_weights = {name: (weight_value if name in valid_names else 0.0) for name in scaled_components}

    hri = None
    for name in valid_names:
        term = scaled_components[name] * component_weights[name]
        hri = term if hri is None else hri + term
    assert hri is not None
    hri = hri.where(ocean_mask == 1).astype("float32")
    risk_class = xr.where(hri >= 0.66, 3, xr.where(hri >= 0.33, 2, xr.where(hri.notnull(), 1, 0))).astype("uint8")

    output_ds = xr.Dataset(
        {
            "heat_risk_index": hri,
            "heat_risk_class": risk_class,
            "ocean_mask": ocean_mask.astype("uint8"),
            **{name: da.astype("float32") for name, da in scaled_components.items()},
            **{name: da.astype("float32") for name, da in components_raw.items()},
        },
        attrs={
            "title": "South China Sea relative composite marine heat-risk index",
            "definition": (
                "Equal-weight min-max composite of SSTA trend, mean MHW total days, "
                "MHW total-days trend, and MHW cumulative-intensity trend."
            ),
            "risk_classes": "Low: HRI < 0.33; Medium: 0.33 <= HRI < 0.66; High: HRI >= 0.66",
            "component_weights": json.dumps(component_weights, ensure_ascii=False),
        },
    )
    zarr_path = output_dir / "heat_risk_index.zarr"
    _write_zarr(output_ds, zarr_path, args.overwrite)

    hri_summary = hri.compute()
    weights_all = _area_weights(ocean_mask)
    high_mask = hri_summary >= 0.66
    rows: list[dict[str, Any]] = []
    for region in SUBREGIONS:
        r_mask = _region_mask(ocean_mask, region)
        r_weights = weights_all.where(r_mask == 1, 0.0)
        component_means = {
            name: _weighted_mean(scaled_components[name], r_weights)
            for name in valid_names
        }
        dominant_name = max(component_means, key=lambda key: (-np.inf if not np.isfinite(component_means[key]) else component_means[key]))
        mean_hri = _weighted_mean(hri_summary, r_weights)
        median_values = hri_summary.where(r_mask == 1).values
        median_hri = float(np.nanmedian(median_values))
        high_share = _weighted_share(high_mask.where(r_mask == 1, False), r_weights)
        rows.append(
            {
                "region_id": region.region_id,
                "region_name_en": region.name_en,
                "region_name_zh": region.name_zh,
                "mean_hri": mean_hri,
                "median_hri": median_hri,
                "high_risk_share": high_share,
                "dominant_component": dominant_name,
                "dominant_component_zh": COMPONENT_LABELS[dominant_name],
                "application_meaning": _application_text(mean_hri, high_share),
                **{f"{name}_mean": value for name, value in component_means.items()},
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_csv = output_dir / "heat_risk_subregion_summary.csv"
    summary_json = output_dir / "heat_risk_index_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "analysis_dir": str(analysis_dir),
                "output_dir": str(output_dir),
                "zarr": str(zarr_path),
                "component_weights": component_weights,
                "scale_metadata": scale_metadata,
                "subregions": summary_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[write] {summary_csv}")
    print(f"[write] {summary_json}")

    table_path = paper_dir / "tables" / "heat_risk_index_table.tex"
    figure_path = paper_dir / "figures" / "heat_risk_index_map.png"
    _write_table(summary_df, table_path)
    fig_info = _plot_hri(hri_summary, figure_path)
    print(f"[ok] {fig_info['path']} ({fig_info['bytes']} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build composite South China Sea marine heat-risk index.")
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-dir", default=DEFAULT_PAPER_DIR)
    parser.add_argument("--min-obs", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_heat_risk_index(args)


if __name__ == "__main__":
    main()
