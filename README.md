# Reffuser

Official repository for **Reffuser**.

This repository is prepared for the revised manuscript. The current release is an initial public version, and the codebase will be continuously updated with cleaned implementation, documentation, and reproducibility materials.

## News

- **2026-06-09**: Initial public repository created for the revised manuscript.

## Current Status

This repository is under active development. We are gradually organizing and releasing:

- Core implementation of Reffuser
- Training and inference scripts
- Evaluation code
- Configuration files
- Dataset preparation instructions
- Environment setup instructions
- Checkpoint or pretrained model instructions, when available
- Reproducibility notes for the experiments reported in the paper

## Repository Structure

```text
Reffuser/
├── README.md
├── docs/
│   └── release_plan.md
├── src/reffuser/
│   ├── __init__.py
│   └── data.py
├── tests/
│   └── data_smoke_test.py
├── requirements.txt
└── .gitignore
```

The structure will be expanded as the cleaned code is released.

## Code Release Status

The Reffuser implementation is currently being cleaned, documented, and prepared for
public release.

We will progressively release the implementation, configuration files, environment
specification, and reproducibility materials during the manuscript revision process.

## Initial Utilities

The initial release includes framework-independent utilities for standardizing
multivariate time series and creating historical/future sliding windows. These utilities
support common forecasting data preparation workflows and do not require the full model
implementation.

```bash
pip install -r requirements.txt
PYTHONPATH=src python tests/data_smoke_test.py
```

## Installation

Installation instructions will be added with the full implementation release.

## Usage

Example commands for training, inference, and evaluation will be added in future updates.

## Citation

If this work is useful for your research, please cite our paper. The final BibTeX entry will be added once the bibliographic information is available.

```bibtex
@article{reffuser2026,
  title   = {Reffuser},
  author  = {To be updated},
  journal = {Manuscript under revision},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Contact

Please open an issue for questions once the code and documentation are released.
