# Benchmarking Custom Model w.r.t. scCausalVI Model

This repository contains the code to compare Custom Model (Model 1) w.r.t. scCausalVI on a few RNA sequencing datasets.

## Model 1 : Framework

Model 1 is inspired from scCausalVI model with few potential improvements :-

- A unified encoder for all treatment values, reducing the number of model parameters and thus improving model scalability over a large number of treatment values while maintaining performance similar to scCausalVI.

- Minimizing Total Correlation to ensure proper disentanglement in background latent variable ($z_{bg}$), treatment effect latent variable ($\tilde e_{t}$) and batch covariate ($c$)

## Evaluation Metrics

1. $\| \hat{x} - x \|_2$ : Euclidian distance between observed gene expression values and reconstructed gene expression values. Lower value indicates better reconstruction.

2. $Cov(z_{bg}, \tilde e_{t})$ : Pearson Correlation Coefficient between background latent variable and treatment effect variable (after multiplying with attention). Lower value indicates better disentanglement

3. $Cov(c, \tilde e_{t})$

4. $Cov(z_{bg}, c)$

5. Average Silhoutte Width over $z_{bg}$ : Lower value indicates that background latent variables follow similar distribution regardless of the treatment value ($t$)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sccvi_impl.git
cd sccvi_impl

# Install the package in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"
```

Note: The package uses `pyproject.toml` for configuration. No `setup.py` is needed.

## Package Structure

The package is organized as follows:

```
sccvi_impl/                 # Project root
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
├── test_imports.py         # Test script for imports
├── test_import_usage.py    # Test script for import usage
├── src/
│   └── sccvi_impl/         # Main package
│       ├── cloud/          # Cloud deployment utilities
│       ├── data/           # Data loading and preprocessing
│       │   └── dataloaders/   # Data loaders
│       ├── model/          # Model implementations
│       │   └── base/       # Base model utilities
│       ├── module/         # PyTorch modules
│       ├── results/        # Benchmark results
│       └── scripts/        # Benchmarking scripts
```

## Project Root

The project root is the `sccvi_impl` directory containing the `pyproject.toml` file. All paths and commands should be run from this directory.

## Import Usage

After installation, you can import the main components directly from the package:

```python
from sccvi_impl.model import Model1, scCausalVI

# Or import specific components
from sccvi_impl.model.model_1 import Model1
from sccvi_impl.model.scCausalVI import scCausalVI
from sccvi_impl.data.utils import preprocess
```

## Testing Imports

To ensure all package components can be imported correctly, run the test script from the project root:

```bash
python test_imports.py
```

For testing import usage in your own code:

```bash
python test_import_usage.py
```

These scripts will check all imports in the package and report any issues.

## Requirements

See `requirements.txt` for a list of dependencies.

## Useful Links

The official implementation of scCausalVI can be found at the following link. - https://github.com/ShaokunAn/scCausalVI.git