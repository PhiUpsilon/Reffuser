#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH=src:. python tests/data_smoke_test.py
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tests/model_smoke_test.py
python scripts/run_reffuser.py --help >/dev/null
python scripts/run_reffuser.py \
  --dataset Weather --input-len 96 --output-len 96 --seed 0 --gpu 0 --dry-run >/dev/null

echo "All public reproducibility smoke tests passed."
