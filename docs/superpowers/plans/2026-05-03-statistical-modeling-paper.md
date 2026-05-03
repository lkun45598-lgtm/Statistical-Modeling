# Statistical Modeling Paper Implementation Plan

> Superseded for the paper-upgrade phase by `docs/superpowers/plans/2026-05-03-statistical-modeling-paper-upgrade.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete first-version competition paper source for the South China Sea SST warming and marine heatwave statistical-modeling project.

**Architecture:** The paper is generated from a LaTeX main source under `paper/`, with figures copied from the analysis output directory and tables generated from CSV/JSON products. A separate driver-analysis script computes climate-index and wind explanatory statistics so the paper text only states results supported by local data.

**Tech Stack:** Python in `/home/lz/miniconda3/envs/pytorch`, NumPy, SciPy, pandas, xarray, Matplotlib, XeLaTeX/latexmk, BibTeX-style `.bib` references.

---

### Task 1: Driver Analysis Products

**Files:**
- Create: `scripts/34_build_driver_analysis.py`
- Write outputs to: `/data1/user/lz/osita_data/scs_5s25n/analysis/driver_analysis/`

- [ ] **Step 1: Load local monthly and annual datasets**

Read:
```text
/data1/user/lz/osita_data/scs_5s25n/analysis/scs_monthly_area_mean_sst_ssta.csv
/data1/user/lz/osita_data/scs_5s25n/analysis/mhw_annual_area_mean.csv
/data1/user/lz/osita_data/external_drivers/climate_indices_monthly_1991_2021.csv
/data1/user/lz/osita_data/external_drivers/ncep_wind_scs_region_mean.csv
```

- [ ] **Step 2: Compute monthly driver correlations**

Use South China Sea area-mean monthly SSTA as the target. Standardize predictors, evaluate lags 0-6 months for `nino34_cpc_3m_centered`, `pdo`, `dmi`, `soi`, `u10`, `v10`, and `wind_speed_anom`. Output `monthly_driver_correlations.csv`.

- [ ] **Step 3: Compute annual MHW driver correlations**

Aggregate climate indices and wind variables to annual means, merge with `mhw_annual_area_mean.csv`, and compute Pearson correlations against `mhw_frequency`, `mhw_total_days`, `mhw_max_intensity`, and `mhw_cumulative_intensity`. Output `annual_mhw_driver_correlations.csv`.

- [ ] **Step 4: Fit compact standardized regression models**

Fit ordinary least squares by `numpy.linalg.lstsq` with standardized predictors. Use a small predictor set to avoid overfitting:
```text
monthly_ssta ~ nino34_cpc_3m_centered_lag3 + pdo_lag0 + dmi_lag0 + wind_speed_anom_lag0
annual_mhw_total_days ~ nino34_cpc_3m_centered + pdo + dmi + wind_speed_anom
```
Compute coefficient, t statistic, p value, R2, adjusted R2, and VIF. Output CSV and JSON summaries.

- [ ] **Step 5: Generate driver figures**

Write:
```text
monthly_ssta_driver_fit.png
annual_mhw_driver_fit.png
driver_correlation_heatmap.png
```

- [ ] **Step 6: Verify**

Run:
```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/34_build_driver_analysis.py --overwrite
```
Expected: output CSV/JSON/PNG files exist and are readable.

### Task 2: Paper Assets

**Files:**
- Create or update: `paper/figures/*.png`
- Create: `paper/tables/*.tex`
- Create: `paper/references.bib`

- [ ] **Step 1: Copy selected figures**

Copy existing monthly, MHW, and driver-analysis PNG files into `paper/figures/` with stable names used by LaTeX.

- [ ] **Step 2: Generate table snippets**

Create LaTeX tables for data products, data-quality notes, area-mean trends, MHW trends, and driver regressions under `paper/tables/`.

- [ ] **Step 3: Add bibliography**

Create `paper/references.bib` with references for OSTIA, marine heatwave definition, NCEP/NCAR reanalysis, climate indices/data sources, Mann-Kendall/Theil-Sen trend methods, and related South China Sea warming/MHW context.

### Task 3: LaTeX Main Draft

**Files:**
- Create: `paper/main.tex`

- [ ] **Step 1: Build document structure**

Use `ctexart` with A4 paper and margins matching the competition requirement. Include title, Chinese abstract, keywords, table of contents, list of tables, list of figures, body sections, references, appendix, and optional acknowledgement.

- [ ] **Step 2: Write full first-version body**

Write sections:
```text
摘要
一、引言
二、研究区、数据来源与质量控制
三、指标体系与统计建模方法
四、南海海表温度增暖的统计测度
五、海洋热浪风险的时空演化
六、气候驱动因子解释模型
七、讨论、结论与建议
参考文献
附录
```

- [ ] **Step 3: Insert figures and tables**

Reference every table/figure in the text and make captions competition-style.

- [ ] **Step 4: Verify compilation**

Run:
```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=paper/build paper/main.tex
```
Expected: `paper/build/main.pdf` exists.

### Task 4: Version Control

**Files:**
- Add: `docs/superpowers/plans/2026-05-03-statistical-modeling-paper.md`
- Add: `scripts/34_build_driver_analysis.py`
- Add: `paper/main.tex`
- Add: `paper/references.bib`
- Add: `paper/figures/*`
- Add: `paper/tables/*`

- [ ] **Step 1: Check worktree**

Run:
```bash
git status --short
```

- [ ] **Step 2: Commit and push**

Run:
```bash
git add docs/superpowers/plans/2026-05-03-statistical-modeling-paper.md scripts/34_build_driver_analysis.py paper
git commit -m "Draft statistical modeling paper"
git push origin main
```

- [ ] **Step 3: Final verification**

Run `git status --short` and confirm no unexpected files remain.
