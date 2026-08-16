# Result Files and Aggregation

## Output Mapping

For a run such as Weather-96-336, outputs are stored below:

```text
checkpoints/Reffuser/Weather_100_96_336/<run>/0.1_0.9_0.1/<config-hash>/
```

The hashed run directory contains the configuration snapshot and generated artifacts. Depending on the runner stage, relevant files include:

- best validation checkpoint;
- `test_metrics.json`;
- saved predictions and targets;
- training/evaluation logs;
- `agent_checkpoint.pth`;
- `training_metrics.pkl`.

## Reported Main Metrics

The normalized main-table protocol reports MSE and MAE. For each dataset/horizon pair:

1. locate the completed `test_metrics.json` files;
2. verify dataset, input length, output length, coefficients, and metric space;
3. compute the arithmetic mean across repeated run-level metrics;
4. compute the corresponding standard deviation;
5. report `mean +/- standard deviation`.

Do not mix normalized main-table metrics with the denormalized RL-baseline protocol.

## Minimal Validation

After one run, locate metrics with:

```bash
find checkpoints/Reffuser -name test_metrics.json -print
```

Inspect a file with:

```bash
python -m json.tool /path/to/test_metrics.json
```

The run wrapper prints the expected checkpoint root before launching training, making it possible to map every command to its result directory.
