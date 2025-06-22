#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Import Test for sccvi_impl

This script tests importing all Python modules in the project to verify that imports work correctly.
It helps identify issues with:
- Missing dependencies
- Circular imports
- Syntax errors
- Path configuration issues

Usage:
    python test_all_imports.py             # Run basic import tests
    python test_all_imports.py --verbose   # Include full tracebacks
    python test_all_imports.py --fix       # Attempt to fix common issues
"""

import os
import sys
import importlib
import traceback
import subprocess
import argparse

# Try to import colorama, install if missing
try:
    from colorama import Fore, Style, init
    init(autoreset=True)  # Initialize colorama for cross-platform colored terminal output
except ImportError:
    print("Installing required package: colorama")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import Fore, Style, init
    init(autoreset=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test all imports in the project")
    parser.add_argument("--verbose", action="store_true", help="Show full tracebacks for errors")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    return parser.parse_args()

def print_success(message):
    """Print success message in green"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print error message in red"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_warning(message):
    """Print warning message in yellow"""
    print(f"{Fore.YELLOW}! {message}{Style.RESET_ALL}")

def print_info(message):
    """Print info message in cyan"""
    print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")

def print_header(message):
    """Print header message in blue"""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}{message}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'-' * len(message)}{Style.RESET_ALL}")

def test_import_module(module_path):
    """
    Test importing a Python module given its file path.
    
    Args:
        module_path: The file path to the Python module
        
    Returns:
        Tuple of (success_status, error_message)
    """
    # Skip files in __pycache__ directories
    if "__pycache__" in module_path:
        return True, None
        
    # Extract module name from path
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    
    # Skip __init__.py files
    if module_name == "__init__":
        return True, None
    
    # Skip this test script itself
    if "test_all_imports.py" in module_path:
        return True, None
    
    # Add directory to sys.path temporarily for the import
    module_dir = os.path.dirname(module_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    try:
        # Attempt to import the module
        module = importlib.import_module(module_name)
        return True, None
    except Exception as e:
        error_msg = traceback.format_exc()
        return False, (str(e), error_msg)
    finally:
        # Clean up sys.path
        if module_dir in sys.path:
            sys.path.remove(module_dir)

def attempt_fix_missing_dependency(missing_module):
    """Attempt to install a missing dependency."""
    print_info(f"Attempting to install missing dependency: {missing_module}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", missing_module])
        print_success(f"Successfully installed {missing_module}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install {missing_module}: {str(e)}")
        return False

def find_all_python_files(start_dir):
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, dirs, files in os.walk(start_dir):
        # Skip __pycache__ and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('__') and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    args = parse_arguments()
    
    # Get the script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Add directories to Python path
    sys.path.append(project_root)
    sys.path.append(script_dir)
    
    if os.path.exists(os.path.join(script_dir, 'src')):
        sys.path.append(os.path.join(script_dir, 'src'))
    
    # Print Python path configuration
    print_header("Python Path Configuration")
    for path in sys.path:
        print(f"- {path}")
    print()
    
    # Find all Python files in the project
    print_header("Finding Python files")
    python_files = find_all_python_files(script_dir)
    print(f"Found {len(python_files)} Python files")
    
    # Test importing each Python file
    print_header("Testing imports")
    successful_imports = []
    failed_imports = []
    
    for module_path in python_files:
        rel_path = os.path.relpath(module_path, project_root)
        success, error = test_import_module(module_path)
        
        if success:
            successful_imports.append(rel_path)
            print_success(f"Successfully imported: {rel_path}")
        else:
            failed_imports.append((rel_path, error))
            print_error(f"Failed to import: {rel_path}")
            print(f"    Error: {error[0]}")
            
            # Attempt to fix if requested
            if args.fix and "No module named" in error[0]:
                missing_module = error[0].split("No module named ")[-1].strip("'")
                if attempt_fix_missing_dependency(missing_module):
                    # Try again after fixing
                    success, error = test_import_module(module_path)
                    if success:
                        successful_imports.append(rel_path)
                        failed_imports.pop()  # Remove from failed list
                        print_success(f"Successfully imported after fix: {rel_path}")
    
    # Summary
    print_header("Import Test Summary")
    print(f"Total modules tested: {len(successful_imports) + len(failed_imports)}")
    print(f"Successfully imported: {len(successful_imports)}")
    print(f"Failed to import: {len(failed_imports)}")
    
    if failed_imports:
        print_header("Failed Imports")
        for module, (error, traceback_msg) in failed_imports:
            print_error(module)
            print(f"    Error: {error}")
            
            # Provide more detailed error information for common issues
            if "No module named" in error:
                missing_module = error.split("No module named ")[-1].strip("'")
                print_warning(f"    Missing dependency: {missing_module}")
                print(f"    Try installing with: pip install {missing_module}")
            elif "cannot import name" in error:
                print_warning("    Possible circular import detected")
            elif "IndentationError" in error or "SyntaxError" in error:
                print_warning("    Syntax error in the module")
            
            # Print full traceback if verbose mode is enabled
            if args.verbose:
                print("\nTraceback:")
                print(traceback_msg)
    
    # Provide recommendations
    if failed_imports:
        print_header("Recommendations")
        if any("No module named" in error[0] for _, error in failed_imports):
            print_info("Install missing dependencies:")
            print("    pip install -r requirements.txt")
            
        if any("cannot import name" in error[0] for _, error in failed_imports):
            print_info("Fix circular imports:")
            print("    1. Identify the circular dependency chain")
            print("    2. Refactor one of the modules to break the cycle")
            print("    3. Consider using late imports inside functions")
            
        if any("IndentationError" in error[0] or "SyntaxError" in error[0] for _, error in failed_imports):
            print_info("Fix syntax errors in the affected modules")
    
    return len(failed_imports)

if __name__ == "__main__":
    sys.exit(main())
