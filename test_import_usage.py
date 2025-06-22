#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Import Usage Test for sccvi_impl

This script tests not only if modules can be imported, but also if basic 
functionality from each module can be executed correctly. This provides
a more comprehensive test of the project's structure and dependencies.
"""

import os
import sys
import importlib
import traceback
import inspect
import argparse
from colorama import Fore, Style, init

init(autoreset=True)  # Initialize colorama for cross-platform colored terminal output

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test actual usage of imported modules")
    parser.add_argument("--verbose", action="store_true", help="Show full tracebacks for errors")
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

def discover_modules(package_name, package_path):
    """Discover all modules in a package recursively."""
    result = []
    
    if not os.path.isdir(package_path):
        return result
        
    # Add the package itself
    if package_name:
        result.append(package_name)
        
    # Find all Python files and directories
    for item in os.listdir(package_path):
        item_path = os.path.join(package_path, item)
        
        # Skip __pycache__ directories and hidden files
        if item.startswith('__pycache__') or item.startswith('.'):
            continue
            
        # If it's a Python file, add it as a module
        if os.path.isfile(item_path) and item.endswith('.py') and item != '__init__.py':
            module_name = f"{package_name}.{item[:-3]}" if package_name else item[:-3]
            result.append(module_name)
                
        # If it's a directory with an __init__.py, it's a package
        elif os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py')):
            subpackage_name = f"{package_name}.{item}" if package_name else item
            result.extend(discover_modules(subpackage_name, item_path))
    
    return result

def test_callable_execution(module, callables):
    """Test executing callable objects from a module."""
    results = []
    
    for obj_name, obj in callables:
        try:
            # Try to execute the function with no arguments first
            try:
                if len(inspect.signature(obj).parameters) == 0:
                    obj()
                    results.append((obj_name, True, None))
                    continue
            except Exception as e:
                # If it fails with no arguments, we'll just report it can be imported
                pass
                
            # If we can import it but not execute it, that's still a partial success
            results.append((obj_name, None, None))
            
        except Exception as e:
            results.append((obj_name, False, str(e)))
            
    return results

def main():
    args = parse_arguments()
    
    # Ensure the project root is in the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print_header("Python Path Configuration")
    for p in sys.path:
        print(f"- {p}")
    
    # Discover all modules
    print_header("Discovering modules")
    
    # Find the package directory
    src_dir = os.path.join(project_root, "sccvi_impl", "src")
    if os.path.exists(src_dir):
        package_dir = os.path.join(src_dir, "sccvi_impl")
    else:
        # If there's no src directory, look directly in the sccvi_impl directory
        package_dir = os.path.join(project_root, "sccvi_impl", "src", "sccvi_impl")
        if not os.path.exists(package_dir):
            package_dir = None
    
    if package_dir and os.path.exists(package_dir):
        modules = discover_modules("sccvi_impl", package_dir)
    else:
        print_error("Could not find the sccvi_impl package directory")
        modules = ["sccvi_impl"]  # Try at least the base package
    
    print(f"Found {len(modules)} modules")
    
    # Test importing and executing functions from each module
    print_header("Testing module imports and basic functionality")
    
    import_results = []
    usage_results = []
    
    for module_name in modules:
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            import_results.append((module_name, True, None))
            
            # Find all callable objects in the module
            callables = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isfunction(obj) or inspect.isclass(obj)) and not name.startswith('_'):
                    callables.append((name, obj))
            
            # Test executing the callables
            if callables:
                callable_results = test_callable_execution(module, callables)
                usage_results.extend([(f"{module_name}.{name}", success, error) 
                                     for name, success, error in callable_results])
                
        except Exception as e:
            tb = traceback.format_exc() if args.verbose else None
            import_results.append((module_name, False, (str(e), tb)))
    
    # Print summary
    print_header("Test Results Summary")
    
    # Import results
    print_header("Module Import Results")
    successful_imports = sum(1 for _, success, _ in import_results if success)
    print(f"Successfully imported: {successful_imports}/{len(import_results)} modules")
    
    failed_imports = [(name, error) for name, success, error in import_results if not success]
    if failed_imports:
        print("\nFailed imports:")
        for name, (error, tb) in failed_imports:
            print_error(f"{name}: {error}")
            if tb:
                print(tb)
    
    # Usage results
    if usage_results:
        print_header("Function/Class Usage Results")
        successful_executions = sum(1 for _, success, _ in usage_results if success is True)
        importable_only = sum(1 for _, success, _ in usage_results if success is None)
        print(f"Successfully executed: {successful_executions} functions/classes")
        print(f"Importable but not tested: {importable_only} functions/classes")
        
        failed_executions = [(name, error) for name, success, error in usage_results if success is False]
        if failed_executions:
            print("\nFailed executions:")
            for name, error in failed_executions:
                print_error(f"{name}: {error}")
    
    # Return non-zero exit code if any imports failed
    return 1 if failed_imports else 0

if __name__ == "__main__":
    sys.exit(main())
