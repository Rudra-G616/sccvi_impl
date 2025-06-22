# sccvi_impl Import Tests

This directory contains a test script to verify that all modules in the `sccvi_impl` project can be successfully imported. This test helps identify any import-related issues in the codebase.

## Using the Import Test Script

### Basic Usage

```bash
python test_all_imports.py
```

This script automatically discovers all Python modules in the project by traversing the directory structure and tests importing each one. It provides a simple pass/fail report and error messages for any modules that fail to import.

### Detailed Output

For more detailed output, including full tracebacks:

```bash
python test_all_imports.py --verbose
```

### Auto-Fix Mode

The script can attempt to automatically fix common issues, such as missing dependencies:

```bash
python test_all_imports.py --fix
```

## What the Script Checks For

The import test identifies various issues including:

1. **Missing dependencies**: Modules that aren't installed in your environment
2. **Circular imports**: Modules that import each other creating dependency cycles
3. **Syntax errors**: Invalid Python code in modules
4. **Path configuration issues**: Problems with PYTHONPATH or package structure

## Fixing Import Errors

If you encounter import errors, here are some common solutions:

1. **Missing dependencies**: Install the required package using pip
   ```bash
   pip install <package-name>
   ```

2. **Path issues**: Make sure your `PYTHONPATH` includes the project root
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/project/root
   ```

3. **Circular imports**: Refactor your code to eliminate circular dependencies between modules

4. **Syntax errors**: Fix any syntax errors in the imported modules
