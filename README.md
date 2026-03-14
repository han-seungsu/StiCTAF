# StiCTAF

This repository contains the code used to reproduce the experiments in the paper:

**Stick-breaking Component-wise Tail Adaptive Flow for Variational Inference**

The repository is organized so that the core implementation lives in `src/`, while experiment-specific reproduction scripts live in `reproduce/`.

## Repository structure

```text
StiCTAF/
├── README.md
├── requirements.txt
├── src/
│   ├── base.py
│   ├── config.py
│   ├── core.py
│   ├── HeavyTarget.py
│   ├── train.py
│   ├── transforms.py
│   └── utils.py
├── reproduce/
│   ├── gaussian_inverse_gamma.py
│   ├── complex_mixture_target.py
│   └── gpd_real_data.py
├── exports/
└── outputs/
```

- `src/` contains the main implementation of StiCTAF and related utilities.
- `reproduce/` contains scripts for reproducing the supplementary figures and tables.
- `exports/` should contain saved posterior samples or benchmark outputs used by the reproduction scripts.
- `outputs/` will be created automatically and will store generated figures and tables.

## Environment

The code was developed for Python 3.10+ and uses PyTorch-based normalizing-flow models.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

The current repository is set up to reproduce the supplementary full figures and summary tables from saved outputs.

### 1. Gaussian--Inverse--Gamma experiment

This script reads saved benchmark samples from `exports/` and generates the full comparison figure and summary table.

```bash
python reproduce/gaussian_inverse_gamma.py \
  --exports-dir exports \
  --output-dir outputs/gaussian_inverse_gamma
```

Expected outputs include:
- a full benchmark comparison figure
- a percentile summary table

### 2. Complex multimodal target experiment

This script reconstructs the target distribution, loads saved benchmark samples from `exports/`, and generates the full comparison figure and summary table.

```bash
python reproduce/complex_mixture_target.py \
  --exports-dir exports \
  --output-dir outputs/complex_mixture
```

Expected outputs include:
- a full benchmark comparison figure
- a summary CSV table

### 3. Real-data analysis

This script loads saved posterior draws together with the MCMC reference output and produces representative posterior plots, the full posterior grid, and the summary table.

```bash
python reproduce/gpd_real_data.py \
  --mcmc-rds /path/to/gpd_wind_2024_10_3_burned_mcmc.rds \
  --exports-dir exports \
  --output-dir outputs/gpd
```

Expected outputs include:
- representative posterior figures
- a full posterior comparison figure
- a summary CSV table with posterior modes and 99% intervals

## Data and saved outputs

The reproduction scripts assume that saved benchmark samples are already available in `exports/`.
These files are produced by the training code and are used here to generate the final figures and tables efficiently.

For the real-data analysis, the MCMC reference file must also be provided separately in `.rds` format.

## Notes

- The current repository focuses on reproducing the figures and tables reported in the paper.
- Some external baselines, such as gTAF, may be added separately.
- If file-name patterns in `exports/` differ from those assumed by the reproduction scripts, the relevant path patterns in `reproduce/` may need small adjustments.

