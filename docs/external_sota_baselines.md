# External SOTA Baselines

Use these as comparison models, not as the paper's only contribution.

## OpenSTL

OpenSTL is useful for ConvLSTM, PredRNN variants, SimVP/SimVPv2, TAU, SwinLSTM, etc.

```bash
git clone https://github.com/chengtan9907/OpenSTL external/OpenSTL
cd external/OpenSTL
pip install -e .
```

Recommended baselines:
- ConvLSTM
- PredRNN.V2
- SimVPv2
- TAU
- SwinLSTM

Convert this repo's zarr data into OpenSTL custom dataset format if you want exact OpenSTL runs. For the competition paper, ReefCastNet-SimVP can be the main model; OpenSTL models are the controlled benchmark.

## CAS-Canglong

Use only as coarse-resolution external SST S2S reference, not as the 5km coral-reef model.

```bash
git clone https://github.com/GISWLH/CAS-Canglong external/CAS-Canglong
```

It is global 0.25-degree SST S2S forecasting; compare on regional mean or downsampled fields only.

## Time-Series-Library

Use for reef-region mean time series only.

```bash
git clone https://github.com/thuml/Time-Series-Library external/Time-Series-Library
```

Good baselines:
- PatchTST
- iTransformer
- TimeMixer
- TimesNet
