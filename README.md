# 南海海温增暖与海洋热浪风险评估

本仓库用于支撑论文《基于高分辨率海温资料的南海增暖及海洋热浪风险评估研究》的数据处理、统计分析、图表生成和论文导出。当前研究重点为南海增暖与海洋热浪风险评估；仓库中只保留与本次论文复现直接相关的脚本和文档。

作者：Leizheng。

## 项目结构

```text
paper/
  main.tex                         # LaTeX 论文主文件
  figures/                         # 论文使用的图件
  tables/                          # 论文使用的 LaTeX 表格
scripts/
  20_prepare_ostia_scs.py          # 裁剪 OSTIA 南海日/月尺度数据
  21_download_external_drivers.py  # 下载并整理气候指数和风场资料
  30_build_monthly_sst_products.py # 构建月尺度 SST/SSTA 与趋势产品
  31_make_monthly_sst_figures.py   # 生成月尺度图件
  32_build_daily_mhw_products.py   # 识别日尺度海洋热浪事件
  33_make_mhw_figures.py           # 生成 MHW 图件和趋势表
  34_build_driver_analysis.py      # 构建驱动因子解释分析
  35_build_word_paper.py           # 由 LaTeX 主文件导出 Word 论文
  36_build_subregion_analysis.py   # 构建分区统计结果
  37_build_robustness_analysis.py  # 构建稳健性和阈值敏感性分析
  38_build_heat_risk_index.py      # 构建综合海洋热风险指数
  39_make_paper_tables_figures.py  # 汇总论文图表资产
docs/
  final_submission_checklist.md    # 当前提交物检查清单
  paper_style_reference_notes.md   # 优秀论文和格式参考说明
```

大型原始数据、裁剪数据和分析输出不提交到 git，默认保存在 `/data1/user/lz/osita_data/`。

## 环境

本机直接使用已有 conda 环境：

```bash
/home/lz/miniconda3/envs/pytorch/bin/python
```

若需要单独安装依赖：

```bash
pip install -r requirements.txt
```

## 数据范围

当前论文使用 OSTIA 日尺度高分辨率海表温度资料，研究区为：

```text
5°S--25°N, 100°E--125°E
```

本地原始数据位置：

```text
/data/sst_data/sst_missing_value_imputation/copernicus_data/copernicus_sst_monthly_1991_2021.nc
```

本地裁剪和分析输出位置：

```text
/data1/user/lz/osita_data/scs_5s25n/
```

当前本地源数据存在三个不可用月份：`2014-07`、`2015-11`、`2015-12`。论文和脚本均采用保守处理：月尺度分析排除这些月份，日尺度 MHW 识别中缺失日打断连续事件，不进行插值。

`scripts/20_prepare_ostia_scs.py` 既可以直接接收上述 `.nc` 文件，也可以接收包含该文件的目录；如果你传入的是 `/data/sst_data/sst_missing_value_imputation/ostia/`，脚本会自动回退到同级 `copernicus_data/` 下的原始文件。`ostia/` 目录本身主要是辅助脚本、Notebook 和示意图，不是这份原始 nc 文件本体。

## 复现流程

### 1. 裁剪 OSTIA 南海数据

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/20_prepare_ostia_scs.py \
  --output-dir /data1/user/lz/osita_data/scs_5s25n \
  --lat-min -5 \
  --lat-max 25 \
  --lon-min 100 \
  --lon-max 125 \
  --workers 128
```

主要输出：

```text
/data1/user/lz/osita_data/scs_5s25n/ostia_scs_daily.zarr
/data1/user/lz/osita_data/scs_5s25n/ostia_scs_monthly.zarr
/data1/user/lz/osita_data/scs_5s25n/metadata.json
```

### 2. 构建月尺度 SST/SSTA 产品

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/30_build_monthly_sst_products.py \
  --input-zarr /data1/user/lz/osita_data/scs_5s25n/ostia_scs_monthly.zarr \
  --output-dir /data1/user/lz/osita_data/scs_5s25n/analysis \
  --overwrite
```

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/31_make_monthly_sst_figures.py \
  --analysis-dir /data1/user/lz/osita_data/scs_5s25n/analysis
```

### 3. 构建日尺度 MHW 产品

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/32_build_daily_mhw_products.py \
  --input-zarr /data1/user/lz/osita_data/scs_5s25n/ostia_scs_daily.zarr \
  --output-dir /data1/user/lz/osita_data/scs_5s25n/analysis \
  --lat-block-size 1 \
  --workers 128 \
  --overwrite
```

主模型采用 Hobday 方法，使用逐日 90% 分位阈值，连续不少于 5 个有效日识别为一次海洋热浪事件。

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/33_make_mhw_figures.py \
  --analysis-dir /data1/user/lz/osita_data/scs_5s25n/analysis
```

### 4. 下载外部驱动因子

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/21_download_external_drivers.py \
  --output-dir /data1/user/lz/osita_data/external_drivers
```

该脚本整理 NOAA/PSL 气候指数和 NCEP/NCAR 10 m 风场资料，供解释型回归分析使用。

### 5. 构建分区、稳健性和综合风险结果

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/34_build_driver_analysis.py \
  --analysis-dir /data1/user/lz/osita_data/scs_5s25n/analysis \
  --drivers-dir /data1/user/lz/osita_data/external_drivers
```

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/36_build_subregion_analysis.py \
  --analysis-dir /data1/user/lz/osita_data/scs_5s25n/analysis
```

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/37_build_robustness_analysis.py \
  --analysis-dir /data1/user/lz/osita_data/scs_5s25n/analysis
```

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/38_build_heat_risk_index.py \
  --analysis-dir /data1/user/lz/osita_data/scs_5s25n/analysis
```

### 6. 汇总论文图表

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/39_make_paper_tables_figures.py \
  --analysis-dir /data1/user/lz/osita_data/scs_5s25n/analysis \
  --paper-dir paper
```

## 构建论文

完整 PDF：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=paper/build paper/main.tex
```

匿名 PDF：

```bash
xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=paper/build \
  -jobname=main_anonymous \
  '\def\ANONYMOUS{1}\input{paper/main.tex}'
xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=paper/build \
  -jobname=main_anonymous \
  '\def\ANONYMOUS{1}\input{paper/main.tex}'
```

Word 版本：

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/35_build_word_paper.py
```

当前生成文件：

```text
paper/build/main.pdf
paper/build/main_anonymous.pdf
paper/build/作品全文-组别-作品编号.pdf
paper/build/匿名作品-组别-作品编号.pdf
paper/build/作品全文-组别-作品编号.docx
paper/build/full_paper.docx
```

Word 目录为可更新域。最终提交前需要在 Word/WPS 中更新目录域，刷新页码。

## 仓库提交原则

- 只提交支撑当前论文的数据处理、分析、图表和导出代码。
- 不提交旧预测模型、训练脚本、无关基线代码或大体积生成数据。
- 代码注释使用中文，说明“为什么这样处理”和“关键口径是什么”，避免无意义注释。
- 大型数据、模型输出、LaTeX 编译产物和本地比赛附件不进入 git。
