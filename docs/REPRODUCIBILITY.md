# Reproducibility Protocol

## Main Experiment Grid

The main Reffuser experiment grid covers:

- nine datasets;
- four forecast horizons per dataset.

The complete grid is executable through:

```bash
bash scripts/reproduce_main_table.sh
```

The manuscript-artifact, command, and output mapping is provided in [ARTIFACT_MAP.md](ARTIFACT_MAP.md).

## Fixed Reffuser Coefficients

| Parameter | Value | Role |
|---|---:|---|
| `ALPHA_ACTOR_LOSS` | 0.1 | Q-driven/supervised actor-loss trade-off |
| `ALPHA_CRITIC_SIMILARITY` | 0.9 | critic similarity weighting |
| `ALPHA_REWARDS_SIMILARITY` | 0.1 | Fourier reward magnitude/phase weighting |
| Epochs | 100 | training duration |
| Target metric | MAE | validation checkpoint selection |

The coefficients remain fixed during each run. They are not dynamically adapted across epochs.

## Architecture

The default actor is PatchTST with three encoder layers, 16 attention heads, model dimension 128, feed-forward dimension 256, patch length 16, stride 8, and dropout 0.2. The Reffuser configuration also constructs the critic, learned time schedule, and learned noise schedule. Dataset-specific node counts and forecast lengths are obtained from the generated `desc.json`.

## Dataset-specific Optimization

The authoritative settings are the executable configuration files in `baselines/Reffuser/`. They record:

- Adam learning rate and weight decay;
- scheduler milestones and decay factor;
- gradient clipping norm;
- training, validation, and test batch size;
- deterministic execution settings;
- checkpoint output path.

The configurations are preserved rather than replaced by one global optimizer setting because the experiments use dataset-specific optimization settings.

## Hyperparameter Search

The sensitivity analysis varies one coefficient at a time over `{0.1, 0.2, ..., 0.9}`, with the other two coefficients fixed to `0.5`. The final main-run coefficients are `(0.1, 0.9, 0.1)` for actor-loss, critic-similarity, and reward-similarity coefficients, respectively.

## Determinism

Each configuration sets:

```python
CFG.ENV.DETERMINISTIC = True
CFG.ENV.CUDNN.DETERMINISTIC = True
```

Exact bitwise equality across different GPUs, CUDA drivers, or PyTorch releases is not guaranteed. For numerical replication, use the pinned environment and report the aggregation procedure used for repeated runs.

## Outputs

Each run writes its checkpoint, logs, configuration snapshot, predictions, targets, and metric JSON below `checkpoints/Reffuser/`. See [RESULTS.md](RESULTS.md) for the directory mapping and aggregation procedure.

## Reproduction Checklist

1. Check out the archival tag `v1.0-r1` and record its resolved Git commit.
2. Create the pinned environment and run `scripts/smoke_test.sh`.
3. Download the raw datasets and verify their filenames.
4. Run one small setting before starting the complete grid.
5. Preserve generated `desc.json` and configuration snapshots with the results.
6. Aggregate repeated runs using a consistent, explicitly reported procedure.
7. Report software, GPU, and CUDA versions with reproduced metrics.
