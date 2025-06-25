# SCCVI_IMPL Testing Guide

This guide provides instructions for testing the import functionality of your SCCVI_IMPL project.

## Basic Import Testing

The `test_imports.py` script checks if all modules in the project can be successfully imported:

```bash
python test_imports.py
```

### Options:

- `--verbose`: Show detailed error messages and tracebacks
- `--fix`: Attempt to automatically fix common import issues (e.g., installing missing dependencies)

## Comprehensive Usage Testing

The `test_import_usage.py` script performs more comprehensive testing by:

1. Discovering all modules in the project
2. Testing if modules can be imported
3. Testing if basic functionality (functions and classes) can be executed

```bash
python test_import_usage.py
```

### Options:

- `--verbose`: Show detailed error messages and tracebacks

## Common Issues and Solutions

1. **Module not found errors**:
   - Ensure your PYTHONPATH includes the project root
   - Check if your package is properly installed (e.g., in development mode)
   - Verify package structure matches `pyproject.toml` configuration

2. **Import errors in specific modules**:
   - Check if all dependencies are installed
   - Look for circular dependencies between modules
   - Ensure relative imports are used correctly

3. **Errors during function execution**:
   - This often indicates missing dependencies or incorrect configuration
   - Check function parameters and default values

## Installation and Development

For development, install the package in editable mode:

```bash
pip install -e .
```

This command should be run from the root directory of the project (where `pyproject.toml` is located). It ensures that your package structure is correctly recognized by Python, and any changes you make to the code will be immediately available without reinstalling the package.

## Project Structure

The SCCVI_IMPL project is organized as follows:

```
sccvi_impl/
├── pyproject.toml
├── README.md
├── requirements.txt
├── test_import_usage.py
├── test_imports.py
├── TESTING_GUIDE.md
└── src/
    └── sccvi_impl/
        ├── __init__.py
        ├── cloud/
        ├── data/
        ├── model/
        ├── module/
        ├── results/
        └── scripts/
```

Testing scripts are located at the root of the project for easy access.
