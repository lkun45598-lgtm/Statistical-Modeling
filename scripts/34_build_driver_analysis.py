#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ANALYSIS_DIR = Path("/data1/user/lz/osita_data/scs_5s25n/analysis")
DEFAULT_EXTERNAL_DIR = Path("/data1/user/lz/osita_data/external_drivers")
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "driver_analysis"

CLIMATE_VARS = ["nino34_cpc_3m_centered", "pdo", "dmi", "soi"]
WIND_VARS = ["u10", "v10", "wind_speed", "wind_speed_anom"]
MONTHLY_DRIVER_VARS = CLIMATE_VARS + WIND_VARS
ANNUAL_DRIVER_VARS = CLIMATE_VARS + ["u10", "v10", "wind_speed_anom"]
MHW_TARGETS = ["mhw_frequency", "mhw_total_days", "mhw_max_intensity", "mhw_cumulative_intensity"]


def _maybe_reset_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    mean = values.mean(skipna=True)
    std = values.std(skipna=True, ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return values * np.nan
    return (values - mean) / std


def _pearson_row(x: pd.Series, y: pd.Series, min_obs: int = 12) -> dict[str, Any]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    n = int(len(data))
    if n < min_obs or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return {"n": n, "r": np.nan, "p_value": np.nan}
    r, p = stats.pearsonr(data["x"], data["y"])
    return {"n": n, "r": float(r), "p_value": float(p)}


def _ols_standardized(df: pd.DataFrame, target: str, predictors: list[str]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    cols = [target] + predictors
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    for col in cols:
        data[col] = _standardize(data[col])
    data = data.dropna()

    y = data[target].to_numpy(dtype="float64")
    x = data[predictors].to_numpy(dtype="float64")
    n = len(y)
    p = len(predictors)
    x_design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    fitted = x_design @ beta
    resid = y - fitted

    df_resid = n - p - 1
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid > 0 and np.isfinite(r2) else np.nan
    sigma2 = sse / df_resid if df_resid > 0 else np.nan
    cov = sigma2 * np.linalg.pinv(x_design.T @ x_design)
    stderr = np.sqrt(np.diag(cov))
    t_stats = beta / stderr
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=df_resid) if df_resid > 0 else np.full_like(beta, np.nan)

    coef_rows = []
    for i, name in enumerate(["intercept", *predictors]):
        coef_rows.append(
            {
                "term": name,
                "coef_standardized": float(beta[i]),
                "stderr": float(stderr[i]),
                "t_stat": float(t_stats[i]),
                "p_value": float(p_values[i]),
            }
        )

    vif_rows = []
    for i, name in enumerate(predictors):
        others = [j for j in range(p) if j != i]
        if not others:
            vif = 1.0
        else:
            xi = x[:, i]
            xo = np.column_stack([np.ones(n), x[:, others]])
            b, *_ = np.linalg.lstsq(xo, xi, rcond=None)
            pred = xo @ b
            ss_res = float(np.sum((xi - pred) ** 2))
            ss_tot = float(np.sum((xi - xi.mean()) ** 2))
            r2_i = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            vif = 1.0 / (1.0 - r2_i) if np.isfinite(r2_i) and r2_i < 1.0 else np.inf
        vif_rows.append({"term": name, "vif": float(vif)})

    fitted_df = data.reset_index(drop=True).copy()
    fitted_df[f"{target}_fitted_std"] = fitted
    fitted_df[f"{target}_residual_std"] = resid
    model_summary = {
        "target": target,
        "predictors": predictors,
        "n": int(n),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "sse": float(sse),
        "df_resid": int(df_resid),
    }
    coef_df = pd.DataFrame(coef_rows).merge(pd.DataFrame(vif_rows), on="term", how="left")
    return coef_df, model_summary, fitted_df


def _load_monthly_inputs(analysis_dir: Path, external_dir: Path) -> pd.DataFrame:
    sst = pd.read_csv(analysis_dir / "scs_monthly_area_mean_sst_ssta.csv")
    climate = pd.read_csv(external_dir / "climate_indices_monthly_1991_2021.csv")
    wind = pd.read_csv(external_dir / "ncep_wind_scs_region_mean.csv")

    for df in (sst, climate, wind):
        df["time"] = pd.to_datetime(df["time"])

    merged = sst.merge(climate, on="time", how="left").merge(wind, on="time", how="left")
    merged = merged.sort_values("time").reset_index(drop=True)
    wind_clim = merged.groupby("month")["wind_speed"].transform("mean")
    merged["wind_speed_anom"] = merged["wind_speed"] - wind_clim
    return merged


def _monthly_lag_correlations(monthly: pd.DataFrame, output_dir: Path, max_lag: int) -> pd.DataFrame:
    rows = []
    for var in MONTHLY_DRIVER_VARS:
        if var not in monthly:
            continue
        for lag in range(max_lag + 1):
            driver = monthly[var].shift(lag)
            row = {
                "target": "ssta_area_mean_c",
                "driver": var,
                "lag_months": lag,
                **_pearson_row(driver, monthly["ssta_area_mean_c"], min_obs=36),
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    out["abs_r"] = out["r"].abs()
    out = out.sort_values(["driver", "lag_months"]).reset_index(drop=True)
    out.to_csv(output_dir / "monthly_driver_correlations.csv", index=False)
    print(f"[write] {output_dir / 'monthly_driver_correlations.csv'}")
    return out


def _annual_driver_table(monthly: pd.DataFrame, mhw: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    annual_drivers = (
        monthly.assign(year=monthly["time"].dt.year)
        .groupby("year")[ANNUAL_DRIVER_VARS]
        .mean(numeric_only=True)
        .reset_index()
    )
    annual = mhw.merge(annual_drivers, on="year", how="left")
    annual.to_csv(output_dir / "annual_driver_dataset.csv", index=False)
    print(f"[write] {output_dir / 'annual_driver_dataset.csv'}")
    return annual


def _annual_correlations(annual: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for target in MHW_TARGETS:
        for driver in ANNUAL_DRIVER_VARS:
            rows.append({"target": target, "driver": driver, **_pearson_row(annual[driver], annual[target], min_obs=20)})
    out = pd.DataFrame(rows)
    out["abs_r"] = out["r"].abs()
    out.to_csv(output_dir / "annual_mhw_driver_correlations.csv", index=False)
    print(f"[write] {output_dir / 'annual_mhw_driver_correlations.csv'}")
    return out


def _plot_fit(
    fitted: pd.DataFrame,
    target: str,
    fitted_col: str,
    x_values: pd.Series | np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(x_values, fitted[target], color="#1f77b4", linewidth=1.6, marker="o", markersize=3.2, label="Observed")
    ax.plot(x_values, fitted[fitted_col], color="#d62728", linewidth=1.5, linestyle="--", label="Fitted")
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(path), "bytes": path.stat().st_size}


def _plot_correlation_heatmap(monthly_corr: pd.DataFrame, annual_corr: pd.DataFrame, path: Path) -> dict[str, Any]:
    monthly_best = (
        monthly_corr.sort_values("abs_r", ascending=False)
        .dropna(subset=["r"])
        .drop_duplicates("driver")
        .assign(target="monthly_ssta_best_lag")
    )
    combined = pd.concat(
        [
            monthly_best[["target", "driver", "r"]],
            annual_corr[annual_corr["target"].isin(["mhw_frequency", "mhw_total_days", "mhw_cumulative_intensity"])][
                ["target", "driver", "r"]
            ],
        ],
        ignore_index=True,
    )
    heat = combined.pivot_table(index="target", columns="driver", values="r", aggfunc="first")
    heat = heat.reindex(
        ["monthly_ssta_best_lag", "mhw_frequency", "mhw_total_days", "mhw_cumulative_intensity"],
        columns=MONTHLY_DRIVER_VARS,
    )

    arr = heat.to_numpy(dtype="float64")
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    mesh = ax.imshow(arr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Pearson r")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Driver Correlations with SCS SSTA and Marine Heatwave Metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(path), "bytes": path.stat().st_size}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_driver_analysis(args: argparse.Namespace) -> None:
    analysis_dir = Path(args.analysis_dir)
    external_dir = Path(args.external_dir)
    output_dir = Path(args.output_dir)
    _maybe_reset_output_dir(output_dir, args.overwrite)

    monthly = _load_monthly_inputs(analysis_dir, external_dir)
    monthly.to_csv(output_dir / "monthly_driver_dataset.csv", index=False)
    print(f"[write] {output_dir / 'monthly_driver_dataset.csv'}")

    mhw = pd.read_csv(analysis_dir / "mhw_annual_area_mean.csv")
    monthly_corr = _monthly_lag_correlations(monthly, output_dir, args.max_lag)
    annual = _annual_driver_table(monthly, mhw, output_dir)
    annual_corr = _annual_correlations(annual, output_dir)

    monthly_model = monthly.copy()
    monthly_model["nino34_cpc_3m_centered_lag3"] = monthly_model["nino34_cpc_3m_centered"].shift(3)
    monthly_model["pdo_lag0"] = monthly_model["pdo"]
    monthly_model["dmi_lag0"] = monthly_model["dmi"]
    monthly_model["wind_speed_anom_lag0"] = monthly_model["wind_speed_anom"]
    monthly_predictors = ["nino34_cpc_3m_centered_lag3", "pdo_lag0", "dmi_lag0", "wind_speed_anom_lag0"]
    monthly_coef, monthly_summary, monthly_fitted = _ols_standardized(
        monthly_model,
        "ssta_area_mean_c",
        monthly_predictors,
    )
    monthly_coef.to_csv(output_dir / "monthly_ssta_driver_regression.csv", index=False)
    monthly_fitted[["ssta_area_mean_c", "ssta_area_mean_c_fitted_std", "ssta_area_mean_c_residual_std"]].to_csv(
        output_dir / "monthly_ssta_driver_fitted.csv",
        index=False,
    )
    print(f"[write] {output_dir / 'monthly_ssta_driver_regression.csv'}")

    annual_predictors = ["nino34_cpc_3m_centered", "pdo", "dmi", "wind_speed_anom"]
    annual_coef, annual_summary, annual_fitted = _ols_standardized(annual, "mhw_total_days", annual_predictors)
    annual_coef.to_csv(output_dir / "annual_mhw_total_days_driver_regression.csv", index=False)
    annual_fitted[["mhw_total_days", "mhw_total_days_fitted_std", "mhw_total_days_residual_std"]].to_csv(
        output_dir / "annual_mhw_total_days_driver_fitted.csv",
        index=False,
    )
    print(f"[write] {output_dir / 'annual_mhw_total_days_driver_regression.csv'}")

    figures = {
        "monthly_ssta_driver_fit": _plot_fit(
            monthly_fitted,
            "ssta_area_mean_c",
            "ssta_area_mean_c_fitted_std",
            np.arange(len(monthly_fitted)),
            "Valid monthly sample index",
            "Standardized SSTA",
            f"Monthly SSTA Driver Regression (adjusted R2={monthly_summary['adj_r2']:.3f})",
            output_dir / "monthly_ssta_driver_fit.png",
        ),
        "annual_mhw_driver_fit": _plot_fit(
            annual_fitted,
            "mhw_total_days",
            "mhw_total_days_fitted_std",
            annual.loc[annual_fitted.index, "year"].to_numpy(dtype=int),
            "Year",
            "Standardized MHW total days",
            f"Annual MHW Total Days Driver Regression (adjusted R2={annual_summary['adj_r2']:.3f})",
            output_dir / "annual_mhw_driver_fit.png",
        ),
        "driver_correlation_heatmap": _plot_correlation_heatmap(
            monthly_corr,
            annual_corr,
            output_dir / "driver_correlation_heatmap.png",
        ),
    }

    best_monthly = (
        monthly_corr.dropna(subset=["r"])
        .sort_values("abs_r", ascending=False)
        .drop_duplicates("driver")
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )
    best_monthly.to_csv(output_dir / "monthly_driver_best_lag_correlations.csv", index=False)
    print(f"[write] {output_dir / 'monthly_driver_best_lag_correlations.csv'}")

    summary = {
        "analysis_dir": str(analysis_dir),
        "external_dir": str(external_dir),
        "output_dir": str(output_dir),
        "monthly_model": monthly_summary,
        "annual_mhw_total_days_model": annual_summary,
        "monthly_predictors": monthly_predictors,
        "annual_predictors": annual_predictors,
        "top_monthly_correlations": best_monthly.head(8).to_dict(orient="records"),
        "top_annual_correlations": annual_corr.sort_values("abs_r", ascending=False).head(12).to_dict(orient="records"),
        "figures": figures,
    }
    summary_path = output_dir / "driver_analysis_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {summary_path}")
    for item in figures.values():
        print(f"[ok] {item['path']} ({item['bytes']} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build climate-driver explanatory statistics for SCS SST/MHW paper.")
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--external-dir", default=str(DEFAULT_EXTERNAL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_driver_analysis(args)


if __name__ == "__main__":
    main()
