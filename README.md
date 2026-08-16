# Reffuser

Open-source implementation of **Reffuser** for multivariate time-series forecasting.

Reffuser couples a forecasting actor and critic with adaptive diffusion time and noise schedules for multivariate time-series forecasting. This repository provides the model implementation, nine-dataset configurations, preprocessing code, environment specification, and executable scripts used to reproduce the principal Reffuser results.

## Reproducibility Snapshot

The repository records the software and experimental protocol used for the reported results. For archival reproducibility, record the Git commit or release tag used for an experiment rather than the moving `main` branch.

Reference environment:

- Python 3.8.19
- PyTorch 1.10.0 + CUDA 11.1
- NVIDIA GeForce RTX 4090 GPU
- Intel Xeon Platinum 8383C CPU

See [INSTALL.md](docs/INSTALL.md) for exact installation instructions, [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete protocol, and [ARTIFACT_MAP.md](docs/ARTIFACT_MAP.md) for the command-to-output mapping.

## Repository Structure

```text
Reffuser/
├── baselines/
│   ├── Reffuser/              # model and nine dataset configurations
│   ├── PatchTST/arch/         # default actor backbone
│   └── DLinear/arch/          # backbone-sensitivity implementation
├── basicts/                   # forecasting runner, scaler, metrics, and dataset API
├── experiments/               # training and evaluation entry points
├── scripts/
│   ├── data_preparation/      # preprocessing for the nine datasets
│   ├── run_reffuser.py        # run one configuration
│   ├── reproduce_main_table.sh
│   └── smoke_test.sh
├── src/reffuser/              # framework-independent data utilities
├── tests/                     # executable smoke tests
├── docs/
│   ├── ARTIFACT_MAP.md        # manuscript artifact to command/output mapping
│   └── ...
├── environment.yml
└── requirements.txt
```

## Installation

```bash
conda env create -f environment.yml
conda activate Reffuser
bash scripts/smoke_test.sh
```

The pip alternative and CUDA compatibility notes are documented in [INSTALL.md](docs/INSTALL.md).

## Data Preparation

Place each raw CSV at:

```text
datasets/raw_data/<Dataset>/<Dataset>.csv
```

Supported dataset names are:

```text
Electricity ETTh1 ETTh2 ETTm1 ETTm2 ExchangeRate Illness Traffic Weather
```

Then preprocess one forecasting setting, for example:

```bash
python scripts/data_preparation/Weather/generate_training_data.py \
  --input_len 96 --output_len 336
```

The split ratios, normalization scope, expected filenames, and dataset sources are specified in [DATA.md](docs/DATA.md).

## Run One Experiment

The wrapper prepares the selected window configuration and invokes the same training entry point used in the experiments:

```bash
python scripts/run_reffuser.py \
  --dataset Weather \
  --input-len 96 \
  --output-len 336 \
  --gpu 0
```

Use `--skip-prepare` when the matching processed dataset already exists.

Results are written under:

```text
checkpoints/Reffuser/<Dataset>_100_<input>_<output>/<run>/0.1_0.9_0.1/
```

## Reproduce the Main Reffuser Runs

The complete protocol contains four horizons for each of nine datasets:

```bash
bash scripts/reproduce_main_table.sh
```

The full command is computationally expensive. The script supports environment overrides for a smaller verification run:

```bash
DATASETS=Weather HORIZONS=96 GPU=0 \
  bash scripts/reproduce_main_table.sh
```

See [RESULTS.md](docs/RESULTS.md) for output files and metric aggregation.

## Archival Version

The revision reproducibility snapshot is tagged `v1.0-r1`. Resolve the exact immutable commit with:

```bash
git rev-list -n 1 v1.0-r1
```

Use this tag rather than the moving `main` branch when reproducing the archived manuscript results.

## Evaluation

Evaluate a saved checkpoint with the matching configuration:

```bash
python experiments/evaluate.py \
  --config baselines/Reffuser/Weather.py \
  --checkpoint /path/to/Reffuser_best_val_MAE.pt \
  --gpus 0 \
  --device_type gpu \
  --batch_size 64
```

The configuration and processed dataset description must use the same input/output lengths as the checkpoint.

## Experimental Protocol

- Historical window: 96 for all datasets except ILI (`Illness` in code), which uses 24.
- Forecast horizons: `{96, 192, 336, 720}`; ILI uses `{24, 36, 48, 60}`.
- Chronological split: `0.6/0.2/0.2` for ETT datasets and `0.7/0.1/0.2` otherwise.
- Feature-wise z-score statistics are fitted on the training split only.
- Main-table metrics are MSE and MAE in the normalized forecasting protocol.
- Training uses 100 epochs with deterministic execution settings.
- Reffuser coefficients are `alpha_aloss=0.1`, `alpha_csim=0.9`, and `alpha_rsim=0.1`.

Dataset-specific optimizer, scheduler, batch-size, and clipping settings are preserved in `baselines/Reffuser/<Dataset>.py` and summarized in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Citation

```bibtex
@article{reffuser2026,
  title   = {Reffuser},
  author  = {To be updated},
  journal = {Manuscript under revision},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE). Third-party-derived components retain their original notices in the corresponding source files.

## Contact

For implementation questions, please open a GitHub issue.
