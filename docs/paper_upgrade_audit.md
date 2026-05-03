# Paper Upgrade Audit

## Claims Already Supported

- Claim: 1991--2021 年南海区域平均 SST 和 SSTA 呈显著增暖。
  Evidence: `paper/tables/trend_summary_table.tex`, `paper/figures/sst_area_mean_timeseries.png`, `paper/figures/ssta_trend_map.png`.
  Current numeric hook: SST trend is about `0.191 °C/10a`; SSTA trend is about `0.174 °C/10a`.

- Claim: 南海 MHW 频次、总天数和累计强度呈上升趋势。
  Evidence: `paper/tables/trend_summary_table.tex`, `paper/figures/mhw_area_mean_timeseries.png`, `paper/figures/mhw_total_days_trend_map.png`, `paper/figures/mhw_cumulative_intensity_trend_map.png`.
  Current numeric hook: MHW frequency trend is about `1.308 events/10a`; total-days trend is about `13.381 days/10a`; cumulative-intensity trend is about `15.176 °C·d/10a`.

- Claim: 海洋热浪指标比平均海温更直接刻画持续性热风险。
  Evidence: `paper/figures/mhw_area_mean_timeseries.png`, `paper/figures/mhw_mean_total_days_map.png`, `paper/figures/mhw_total_days_trend_map.png`.
  Current gap: the wording is reasonable, but the paper should explicitly distinguish frequency, exposure duration, maximum intensity, and accumulated thermal stress.

- Claim: 风速距平是当前解释模型中最稳定的解释变量。
  Evidence: `paper/tables/monthly_driver_best_lag_table.tex`, `paper/tables/monthly_driver_regression_table.tex`, `paper/tables/annual_driver_regression_table.tex`, `paper/tables/driver_model_summary_table.tex`, `paper/figures/driver_correlation_heatmap.png`, `paper/figures/monthly_ssta_driver_fit.png`, `paper/figures/annual_mhw_driver_fit.png`.
  Current numeric hook: monthly SSTA regression coefficient for wind-speed anomaly is about `-0.363`; annual MHW total-days regression coefficient is about `-0.558`.

- Claim: 缺失月份没有被插值，而是被保守排除或用于打断事件。
  Evidence: `paper/main.tex` data-quality section; `README.md` notes; monthly and daily analysis scripts.
  Current gap: this is described as a data-quality policy, but it should be connected to robustness analysis rather than left as a limitation only.

## Claims Needing Stronger Evidence

- Claim: 热风险具有分区差异。
  Needed evidence: `paper/tables/subregion_summary_table.tex`, `paper/figures/subregion_trend_panel.png`.
  Reason: the current draft states that nearshore, shelf, and central/southern regions differ, but this is mostly visual interpretation from maps. A competition paper should quantify subregional trends.

- Claim: 趋势结论具有稳健性。
  Needed evidence: `paper/tables/robust_trend_table.tex`, `paper/figures/robust_trend_comparison.png`.
  Reason: the current methods mention Mann-Kendall and Theil-Sen only as future extensions. This weakens the current claim that trends are stable.

- Claim: MHW 结果不依赖单一阈值选择。
  Needed evidence: `paper/tables/threshold_sensitivity_table.tex`, `paper/figures/mhw_threshold_sensitivity.png`.
  Reason: the paper uses the standard 90th-percentile Hobday definition, but a stronger paper should show that the conclusion is directionally stable under 85th and 95th percentile thresholds.

- Claim: 模型能够识别重点风险海区。
  Needed evidence: `paper/tables/heat_risk_index_table.tex`, `paper/figures/heat_risk_index_map.png`.
  Reason: current results report separate SST/MHW fields. A composite relative index would turn the analysis into a more complete statistical risk-identification model.

- Claim: 当前统计建模不是预测优先，而是测度、识别、检验和解释。
  Needed evidence: stronger introduction and methods narrative.
  Reason: the current paper says it is not forecasting-first, but the structure still reads close to a result report. The upgraded paper should explicitly state the statistical questions answered by each model component.

## Writing Passages To Rewrite

- Abstract: replace report-style result listing with purpose-method-result-conclusion-innovation structure. The abstract should state why average-state warming and extreme-event exposure are modeled jointly.

- Introduction: add problem abstraction and why this is statistical modeling rather than pure prediction. The current introduction is acceptable but too short for an award-level paper.

- Research contribution paragraph: expand from three general contributions to four explicit contributions: unified daily/monthly data scale, multi-layer indicator system, robustness/subregion/index evidence, and interpretable driver model.

- Data and methods: expand indicator definitions, model-validation design, and missing-data policy. The present methods describe formulas but do not yet explain how the modeling pieces work together.

- Trend section: add subregional results and nonparametric robustness checks. The current section relies on OLS trend and maps.

- MHW section: add threshold sensitivity and composite HRI. The current MHW section reports event indicators but does not yet translate them into a ranked risk-priority model.

- Driver section: clarify that the regression is explanatory rather than causal or forecasting. Keep the wind-speed result, but state the limitation of regional climate indices.

- Discussion: strengthen the evidence chain from SSTA trend to longer MHW exposure and then to cumulative heat stress. The current discussion is accurate but generic.

- Conclusions and recommendations: every conclusion and recommendation should cite a computed number or table. Avoid broad policy language not derived from this dataset.

## Assets Available Before Upgrade

### Figures

- `paper/figures/sst_area_mean_timeseries.png`
- `paper/figures/sst_climatology_cycle.png`
- `paper/figures/sst_mean_map.png`
- `paper/figures/ssta_trend_map.png`
- `paper/figures/mhw_area_mean_timeseries.png`
- `paper/figures/mhw_mean_total_days_map.png`
- `paper/figures/mhw_total_days_trend_map.png`
- `paper/figures/mhw_cumulative_intensity_trend_map.png`
- `paper/figures/driver_correlation_heatmap.png`
- `paper/figures/monthly_ssta_driver_fit.png`
- `paper/figures/annual_mhw_driver_fit.png`

### Tables

- `paper/tables/data_summary_table.tex`
- `paper/tables/trend_summary_table.tex`
- `paper/tables/monthly_driver_best_lag_table.tex`
- `paper/tables/monthly_driver_regression_table.tex`
- `paper/tables/annual_driver_regression_table.tex`
- `paper/tables/driver_model_summary_table.tex`

## Upgrade Priority

1. Build subregional analysis first, because it directly supports the claim that warming and MHW risk are spatially heterogeneous.
2. Build robustness and threshold-sensitivity analysis second, because it upgrades current limitations into tested evidence.
3. Build HRI third, because it converts separate indicators into a competition-style composite statistical model.
4. Rewrite the paper only after the new numerical outputs exist, so the final text is evidence-driven rather than decorative.
