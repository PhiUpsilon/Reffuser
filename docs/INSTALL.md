# Installation

## Tested Environment

The reported experiments used:

| Component | Version |
|---|---:|
| Python | 3.8.19 |
| PyTorch | 1.10.0+cu111 |
| CUDA toolkit used by PyTorch | 11.1 |
| EasyTorch | 1.3.2 |
| NumPy | 1.22.4 |
| pandas | 1.3.5 |
| SciPy | 1.7.3 |
| scikit-learn | 1.0.2 |
| sktime | 0.29.1 |

Reference hardware was an NVIDIA GeForce RTX 4090 GPU and an Intel Xeon Platinum 8383C CPU. Exact runtime and memory values may vary with the GPU, CUDA driver, and host system.

## Conda Installation

```bash
git clone https://github.com/PhiUpsilon/Reffuser.git
cd Reffuser
conda env create -f environment.yml
conda activate Reffuser
bash scripts/smoke_test.sh
```

The NVIDIA driver must support CUDA 11.1 binaries. The CUDA toolkit is provided inside the Conda environment; a separate system toolkit is not required for ordinary PyTorch execution.

## Pip Installation

Python 3.8 is required for the exact environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/smoke_test.sh
```

If the archived CUDA 11.1 PyTorch wheel is unavailable for the current platform, create the Conda environment instead. Changing PyTorch or CUDA versions can alter deterministic behavior and should be reported when reproducing numerical results.

## Verification

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
PYTHONPATH=src:. python tests/data_smoke_test.py
PYTHONPATH=. python tests/model_smoke_test.py
python scripts/run_reffuser.py --help
```

The model smoke test checks the forecasting actor, time schedule, and noise schedule on synthetic tensors; it does not train a forecasting model.
