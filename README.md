# ReefCastNet: South China Sea Coral-Reef SST Forecasting

This repository contains a complete, runnable codebase for:

**Task**: weekly South China Sea SST anomaly long-lead forecasting and coral-bleaching heat-stress evaluation.

**Main model**: `ReefCastNet-SimVP`, a SimVPv2-inspired spatiotemporal backbone modified for coral-reef heat-stress forecasting:
- SST anomaly forecasting rather than raw SST only
- reef-aware and reef-buffer-weighted loss
- differentiable HotSpot / DHW loss
- optional spatial Alert Level classification head
- weekly 1/2/4/8/12/16 lead evaluation

This codebase is self-contained. It does **not** vendor OpenSTL source code. Use OpenSTL/TAU/SwinLSTM/CAS-Canglong as external baselines if needed; see `docs/external_sota_baselines.md`.

## 0. Install

```bash
conda create -n reefcast python=3.10 -y
conda activate reefcast
pip install -r requirements.txt
pip install -e .
```

For the current statistical-modeling workflow on this machine, use the existing
conda environment:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python
```

## Current competition direction

The current paper direction is **statistical modeling of South China Sea SST
warming and marine heatwave risk**, not only SST forecasting. The forecasting
code remains in the repository, but the active data products are:

- daily South China Sea OSTIA SST crop
- monthly South China Sea OSTIA SST crop
- climate-driver indices and NCEP/NCAR monthly wind fields

Large generated data products are intentionally not committed to git.

## Prepare OSTIA South China Sea data

Raw OSTIA/Copernicus file used locally:

```text
/data/sst_data/sst_missing_value_imputation/copernicus_data/copernicus_sst_monthly_1991_2021.nc
```

Generate daily and monthly South China Sea Zarr products:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/20_prepare_ostia_scs.py \
  --output-dir /data1/user/lz/osita_data \
  --workers 128
```

Main local outputs:

```text
/data1/user/lz/osita_data/ostia_scs_daily.zarr
/data1/user/lz/osita_data/ostia_scs_monthly.zarr
/data1/user/lz/osita_data/metadata.json
```

Default crop is `0-25N, 100-125E`, with SST converted from Kelvin to
`degree_Celsius`.

## Download external climate drivers

Download and align the lightweight driver data used for explanatory modeling:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/21_download_external_drivers.py \
  --output-dir /data1/user/lz/osita_data/external_drivers
```

This fetches NOAA/PSL climate indices and NCEP/NCAR monthly 10 m winds, then
builds:

```text
/data1/user/lz/osita_data/external_drivers/climate_indices_monthly_1991_2021.csv
/data1/user/lz/osita_data/external_drivers/ncep_wind_scs_monthly.zarr
/data1/user/lz/osita_data/external_drivers/ncep_wind_scs_region_mean.csv
```

## 1. Smoke test without NOAA data

```bash
python scripts/00_make_toy_data.py --config configs/toy.yaml
python scripts/05_baselines.py --config configs/toy.yaml --baseline persistence --split test
python scripts/03_train.py --config configs/toy.yaml
python scripts/04_evaluate.py --config configs/toy.yaml --checkpoint outputs_toy/reefcastnet_simvp/best.pt --split test --save-npz
python scripts/06_make_figures.py --npz outputs_toy/reefcastnet_simvp/examples_test.npz
```

## 2. Download NOAA CRW CoralTemp SST

First test with a short period:

```bash
python scripts/01_download_noaa_crw.py --config configs/south_china_sea.yaml --years 2020 2021
```

Full run:

```bash
python scripts/01_download_noaa_crw.py --config configs/south_china_sea.yaml
```

## 3. Build weekly dataset

```bash
python scripts/02_build_weekly_dataset.py --config configs/south_china_sea.yaml
```

Output:

```text
data/processed/scs_weekly.zarr
data/processed/metadata.json
```

## 4. Baselines

```bash
python scripts/05_baselines.py --config configs/south_china_sea.yaml --baseline persistence --split test
python scripts/05_baselines.py --config configs/south_china_sea.yaml --baseline climatology --split test
```

## 5. Train ReefCastNet

Single GPU:

```bash
python scripts/03_train.py --config configs/south_china_sea.yaml
```

8×4090 DDP:

```bash
torchrun --nproc_per_node=8 scripts/03_train.py --config configs/south_china_sea.yaml
```

## 6. Evaluate and plot

```bash
python scripts/04_evaluate.py \
  --config configs/south_china_sea.yaml \
  --checkpoint outputs/reefcastnet_simvp/best.pt \
  --split test \
  --save-npz

python scripts/06_make_figures.py --npz outputs/reefcastnet_simvp/examples_test.npz
```

## What is actually original here?

Do **not** claim that the generic backbone is invented from scratch. The safe statement is:

> We use a SimVPv2-inspired spatiotemporal backbone and modify it into ReefCastNet by adding SST-anomaly formulation, reef-aware static inputs, reef-weighted loss, HotSpot/DHW-aware loss, and Alert Level risk supervision for coral bleaching heat-stress forecasting.

## Final paper model table

Recommended:
- Persistence
- Climatology
- ConvLSTM
- SimVPv2 / OpenSTL baseline
- TAU / OpenSTL baseline
- SwinLSTM baseline
- ReefCastNet-SimVP (main)
- CAS-Canglong as coarse S2S external reference only
