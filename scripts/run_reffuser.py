"""Prepare and run one Reffuser dataset/horizon/seed configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


DATASETS = {
    "Electricity",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "ExchangeRate",
    "Illness",
    "Traffic",
    "Weather",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--input-len", required=True, type=int)
    parser.add_argument("--output-len", required=True, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default="0", help="GPU ID passed to BasicTS")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse an already generated dataset index for this window setting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool = False) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def seed_specific_config(source: Path, seed: int) -> str:
    text = source.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^SEED\s*=\s*\d+\s*$",
        f"SEED = {seed}",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Could not replace SEED in {source}")
    return updated


def main() -> None:
    args = parse_args()
    if args.input_len <= 0 or args.output_len <= 0:
        raise SystemExit("--input-len and --output-len must be positive")

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    preprocessing = root / "scripts" / "data_preparation" / args.dataset / "generate_training_data.py"
    source_config = root / "baselines" / "Reffuser" / f"{args.dataset}.py"
    if not preprocessing.exists() or not source_config.exists():
        raise FileNotFoundError(f"Missing public configuration for {args.dataset}")

    if not args.skip_prepare:
        run(
            [
                sys.executable,
                str(preprocessing),
                "--input_len",
                str(args.input_len),
                "--output_len",
                str(args.output_len),
            ],
            args.dry_run,
        )

    if args.dry_run:
        print(
            "+ temporary config:",
            source_config,
            f"with SEED={args.seed}",
            flush=True,
        )
        run(
            [sys.executable, "experiments/train.py", "-c", "<temporary-config>", "-g", args.gpu],
            True,
        )
        return

    config_text = seed_specific_config(source_config, args.seed)
    config_dir = root / "baselines" / "Reffuser"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"{args.dataset}_public_",
        suffix=".py",
        dir=config_dir,
        delete=False,
    ) as temporary:
        temporary.write(config_text)
        temporary_config = Path(temporary.name)

    try:
        relative_config = temporary_config.relative_to(root)
        run(
            [
                sys.executable,
                "experiments/train.py",
                "-c",
                str(relative_config),
                "-g",
                args.gpu,
            ]
        )
    finally:
        temporary_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
