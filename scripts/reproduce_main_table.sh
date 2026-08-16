#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STRING="${DATASETS:-Electricity ETTh1 ETTh2 ETTm1 ETTm2 ExchangeRate Illness Traffic Weather}"
SEEDS_STRING="${SEEDS:-0}"
GPU="${GPU:-0}"

read -r -a DATASET_LIST <<< "$DATASETS_STRING"
read -r -a SEED_LIST <<< "$SEEDS_STRING"

failures=()
for dataset in "${DATASET_LIST[@]}"; do
    if [[ "$dataset" == "Illness" ]]; then
        input_len=24
        default_horizons="24 36 48 60"
    else
        input_len=96
        default_horizons="96 192 336 720"
    fi
    read -r -a HORIZON_LIST <<< "${HORIZONS:-$default_horizons}"

    for output_len in "${HORIZON_LIST[@]}"; do
        prepared=0
        for seed in "${SEED_LIST[@]}"; do
            args=(
                python scripts/run_reffuser.py
                --dataset "$dataset"
                --input-len "$input_len"
                --output-len "$output_len"
                --seed "$seed"
                --gpu "$GPU"
            )
            if [[ "$prepared" -eq 1 ]]; then
                args+=(--skip-prepare)
            fi
            echo "[RUN] dataset=$dataset input=$input_len output=$output_len seed=$seed gpu=$GPU"
            if "${args[@]}"; then
                prepared=1
            else
                failures+=("$dataset,$input_len,$output_len,$seed")
            fi
        done
    done
done

if (( ${#failures[@]} > 0 )); then
    printf '[FAILED] %s\n' "${failures[@]}" >&2
    exit 1
fi

echo "All requested Reffuser runs completed."
