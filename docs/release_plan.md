# Reffuser Release Checklist

## Public Reproducibility Package

- [x] Core Reffuser implementation
- [x] Default PatchTST actor implementation
- [x] DLinear backbone-sensitivity implementation
- [x] Nine dataset configurations
- [x] BasicTS training and evaluation runner
- [x] Dataset preprocessing scripts
- [x] Exact Python/package environment
- [x] Installation instructions
- [x] Main experiment execution script
- [x] Result-directory and aggregation documentation
- [x] Data and model smoke tests

## Release Procedure

- [x] Run all smoke tests in the pinned Reffuser environment
- [x] Confirm documented command paths and dry-run wiring
- [x] Record the release Git commit through the immutable tag
- [x] Create the immutable `v1.0-r1` tag
- [x] Publish tag-resolution instructions in the repository documentation

This checklist is intentionally separated from experimental results. Checkpoints and raw datasets are not committed to Git and must be obtained or generated according to the documentation.
