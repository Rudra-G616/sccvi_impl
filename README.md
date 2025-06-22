# scCausalVI - Single-Cell Causal Variational Inference

This package implements the scCausalVI framework for analyzing single-cell RNA sequencing data with causal inference capabilities.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sccvi_impl.git
cd sccvi_impl

# Install the package
pip install -e .
```

## Package Structure

The package is organized as follows:

```
sccvi_impl/
├── src/
│   └── sccvi_impl/         # Main package
│       ├── data/           # Data loading and preprocessing
│       ├── model/          # Model implementations
│       ├── module/         # PyTorch modules
│       └── scripts/        # Analysis scripts
```

## Import Usage

You can import the main components directly from the package:

```python
from src.sccvi_impl import Model1, scCausalVI

# Or import specific components
from src.sccvi_impl.model import Model1, scCausalVI
from src.sccvi_impl.data.utils import preprocess
```

## Testing Imports

To ensure all package components can be imported correctly, run:

```bash
python sccvi_impl/test_imports.py
```

This will check all imports in the package and report any issues.

## Requirements

See `requirements.txt` for a list of dependencies.

## License

[License information]