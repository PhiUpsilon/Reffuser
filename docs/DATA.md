# Dataset Preparation

## Raw-data Layout

Download the public long-term forecasting datasets and place the CSV files under the following exact names:

```text
datasets/raw_data/
├── Electricity/Electricity.csv
├── ETTh1/ETTh1.csv
├── ETTh2/ETTh2.csv
├── ETTm1/ETTm1.csv
├── ETTm2/ETTm2.csv
├── ExchangeRate/ExchangeRate.csv
├── Illness/Illness.csv
├── Traffic/Traffic.csv
└── Weather/Weather.csv
```

The datasets follow the commonly used Long-Term Series Forecasting collection. ETT data are available from the [ETDataset repository](https://github.com/zhouhaoyi/ETDataset); the remaining benchmark files are distributed by established forecasting repositories such as [Autoformer](https://github.com/thuml/Autoformer) and [PatchTST](https://github.com/yuqinie98/PatchTST). Preserve the original chronological ordering and column names.

The code uses `Illness` as the directory/configuration name for the ILI dataset.

## Forecasting Protocol

| Dataset group | History | Horizons | Train/validation/test |
|---|---:|---|---|
| ETTh1, ETTh2, ETTm1, ETTm2 | 96 | 96, 192, 336, 720 | 0.6 / 0.2 / 0.2 |
| Electricity, ExchangeRate, Traffic, Weather | 96 | 96, 192, 336, 720 | 0.7 / 0.1 / 0.2 |
| ILI (`Illness`) | 24 | 24, 36, 48, 60 | 0.7 / 0.1 / 0.2 |

Splits are chronological. No future observations are shuffled into training or validation splits.

## Preprocessing

Run the dataset-specific preprocessing script after placing the raw CSV. For example:

```bash
python scripts/data_preparation/ETTh1/generate_training_data.py \
  --input_len 96 --output_len 336
```

The script writes:

```text
datasets/<Dataset>/data.dat
datasets/<Dataset>/index_in_<input>_out_<output>.pkl
datasets/<Dataset>/desc.json
```

`desc.json` records the generated shape, sampling frequency, split ratios, normalization flags, missing-value convention, and input/output lengths. Regenerate the processed index whenever the forecasting horizon changes.

## Normalization and Metrics

- Z-score statistics are fitted using only the chronological training split.
- `NORM_EACH_CHANNEL=True` applies feature-wise/channel-wise normalization.
- `RESCALE=False` is used for the normalized main-table forecasting protocol.
- Missing values are represented by `NaN`; masked metrics exclude invalid entries.
- MSE and MAE are computed over the selected target feature after applying the same mask.

Experiments using a denormalized MAE protocol are not numerically interchangeable with the normalized forecasting protocol described above.

## Data Integrity Check

Before a long run, inspect the generated description:

```bash
python -m json.tool datasets/Weather/desc.json
```

Confirm that `INPUT_LEN`, `OUTPUT_LEN`, split ratios, and node count match the intended configuration.
