#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Import Test for sccvi_impl - Updated Version

This script tests importing all modules in the sccvi_impl package properly.
"""

import os
import sys
import importlib
import traceback
import subprocess
import argparse
from colorama import Fore, Style, init

init(autoreset=True)  # Initialize colorama for cross-platform colored terminal output

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test all imports in the project")
    parser.add_argument("--verbose", action="store_true", help="Show full tracebacks for errors")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues automatically")
    return parser.parse_args()

def print_success(message):
    """Print success message in green"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print error message in red"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_header(message):
    """Print header message in blue"""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}{message}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'-' * len(message)}{Style.RESET_ALL}")

def attempt_to_fix_imports(failed_modules):
    """Try to fix common import issues automatically."""
    print_header("Attempting to fix import issues")
    
    fixed_count = 0
    for name, error, _ in failed_modules:
        error_str = str(error)
        
        # Check for common missing package errors
        if "No module named" in error_str:
            missing_pkg = error_str.split("'")[1]
            # Try to extract the top-level package name
            top_pkg = missing_pkg.split('.')[0]
            
            print(f"Attempting to install missing package: {top_pkg}")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", top_pkg],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print_success(f"Successfully installed {top_pkg}")
                    fixed_count += 1
                else:
                    print_error(f"Failed to install {top_pkg}: {result.stderr}")
            except Exception as e:
                print_error(f"Error during package installation: {str(e)}")
    
    return fixed_count

def main():
    args = parse_arguments()
    
    # Ensure the project root is in the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print_header("Python Path Configuration")
    for p in sys.path:
        print(f"- {p}")
    
    # Automatically discover all modules to test
    print_header("Discovering modules")
    modules_to_test = []
    
    def find_modules(package_path, package_name):
        """Recursively find all modules in the package."""
        if not os.path.isdir(package_path):
            return
            
        # Add the package itself
        if package_name and package_name not in modules_to_test:
            modules_to_test.append(package_name)
            print(f"- Found module: {package_name}")
            
        # Find all Python files and directories
        for item in os.listdir(package_path):
            item_path = os.path.join(package_path, item)
            
            # Skip __pycache__ directories and hidden files
            if item.startswith('__pycache__') or item.startswith('.'):
                continue
                
            # If it's a Python file, add it as a module
            if os.path.isfile(item_path) and item.endswith('.py') and item != '__init__.py':
                module_name = f"{package_name}.{item[:-3]}" if package_name else item[:-3]
                if module_name not in modules_to_test:
                    modules_to_test.append(module_name)
                    print(f"- Found module: {module_name}")
                    
            # If it's a directory with an __init__.py, it's a package
            elif os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py')):
                subpackage_name = f"{package_name}.{item}" if package_name else item
                find_modules(item_path, subpackage_name)
    
    # Find all modules in the sccvi_impl package
    src_dir = os.path.join(project_root, "sccvi_impl", "src")
    if os.path.exists(src_dir):
        package_dir = os.path.join(src_dir, "sccvi_impl")
        find_modules(package_dir, "sccvi_impl")
    else:
        # If there's no src directory, look directly in the sccvi_impl directory
        package_dir = os.path.join(project_root, "sccvi_impl")
        package_src_dir = os.path.join(package_dir, "src", "sccvi_impl")
        if os.path.exists(package_src_dir):
            find_modules(package_src_dir, "sccvi_impl")
    
    # Add the base package if not already included
    if "sccvi_impl" not in modules_to_test:
        modules_to_test.insert(0, "sccvi_impl")
    
    print(f"\nTotal modules discovered: {len(modules_to_test)}")
    
    print_header("Testing imports")
    
    success_count = 0
    failed_modules = []
    
    for module_name in modules_to_test:
        try:
            # Try to import the module
            importlib.import_module(module_name)
            print_success(f"Successfully imported: {module_name}")
            success_count += 1
        except Exception as e:
            print_error(f"Failed to import: {module_name}")
            print(f"    Error: {str(e)}")
            if args.verbose:
                tb = traceback.format_exc()
                failed_modules.append((module_name, str(e), tb))
            else:
                failed_modules.append((module_name, str(e), None))
    
    # Print summary
    print_header("Import Test Summary")
    print(f"Total modules tested: {len(modules_to_test)}")
    print(f"Successfully imported: {success_count}")
    print(f"Failed to import: {len(failed_modules)}")
    
    # Try to fix issues if requested
    if args.fix and failed_modules:
        fixed_count = attempt_to_fix_imports(failed_modules)
        
        if fixed_count > 0:
            print_header("Re-testing after fixes")
            # Re-test the previously failed modules
            still_failed = []
            newly_fixed = 0
            
            for name, _, _ in failed_modules:
                try:
                    importlib.import_module(name)
                    print_success(f"Fixed: {name}")
                    newly_fixed += 1
                except Exception as e:
                    print_error(f"Still failing: {name}")
                    print(f"    Error: {str(e)}")
                    still_failed.append((name, str(e), traceback.format_exc() if args.verbose else None))
            
            print(f"\nFixed {newly_fixed} out of {len(failed_modules)} failing modules")
            failed_modules = still_failed
    
    if failed_modules:
        print_header("Failed Imports")
        for name, error, tb in failed_modules:
            print_error(f"{name}")
            print(f"    Error: {error}")
            if tb and args.verbose:
                print("\nTraceback:")
                print(tb)
    
    # Return non-zero exit code if any imports failed
    return len(failed_modules)

if __name__ == "__main__":
    sys.exit(main())
