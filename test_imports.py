#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Import Test Script for sccvi_impl

This script attempts to import all modules in the sccvi_impl project
to check for import errors. It will report which modules successfully
import and which ones fail.
"""

import os
import sys
import importlib
import traceback
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

def print_success(message):
    """Print success message in green"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print error message in red"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_warning(message):
    """Print warning message in yellow"""
    print(f"{Fore.YELLOW}! {message}{Style.RESET_ALL}")

def print_header(message):
    """Print header message in blue"""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}{message}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'-' * len(message)}{Style.RESET_ALL}")

# Ensure src is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Try to import the main package
try:
    import src.sccvi_impl
    print_success("Successfully imported main package: src.sccvi_impl")
except ImportError as e:
    print_error(f"Failed to import main package: src.sccvi_impl - {e}")

# List of all modules to test importing
modules_to_test = [
    # Core modules
    "src.sccvi_impl",
    
    # Model modules
    "src.sccvi_impl.model",
    "src.sccvi_impl.model.model_1", 
    "src.sccvi_impl.model.scCausalVI",
    "src.sccvi_impl.model.base._utils",
    "src.sccvi_impl.model.base.model_1_training_mixin",
    
    # Module implementation
    "src.sccvi_impl.module",
    "src.sccvi_impl.module.model_1",
    "src.sccvi_impl.module.scCausalVI",
    
    # Data modules
    "src.sccvi_impl.data",
    "src.sccvi_impl.data.utils",
    "src.sccvi_impl.data.Download_Dataset",
    "src.sccvi_impl.data.dataloaders.data_splitting",
    "src.sccvi_impl.data.dataloaders.scCausalVI_dataloader",
    
    # Script modules
    "src.sccvi_impl.scripts.simulated_benchmarking",
    "src.sccvi_impl.scripts.covid_epithelial_benchmarking",
    "src.sccvi_impl.scripts.covid_pbmc_benchmarking",
    "src.sccvi_impl.scripts.ifn_beta_benchmarking",
]

# Results tracking
successful_imports = []
failed_imports = []
import_details = {}

# Test each module
print_header("Testing module imports")

for module_name in modules_to_test:
    try:
        module = importlib.import_module(module_name)
        successful_imports.append(module_name)
        print_success(f"Successfully imported: {module_name}")
        
        # Get module attributes to verify deeper imports
        try:
            attributes = dir(module)
            import_details[module_name] = {"status": "success", "attributes": attributes}
        except Exception as e:
            import_details[module_name] = {"status": "success", "attributes_error": str(e)}
            
    except Exception as e:
        error_msg = traceback.format_exc()
        failed_imports.append(module_name)
        print_error(f"Failed to import: {module_name}")
        print(f"    Error: {str(e)}")
        import_details[module_name] = {"status": "failed", "error": str(e), "traceback": error_msg}

# Summary
print_header("Import Test Summary")
print(f"Total modules tested: {len(modules_to_test)}")
print(f"Successfully imported: {len(successful_imports)}")
print(f"Failed to import: {len(failed_imports)}")

if failed_imports:
    print_header("Failed Imports")
    for module in failed_imports:
        print_error(module)
        error_details = import_details[module]["error"]
        print(f"    Error: {error_details}")
        
        # Provide more detailed error information for common issues
        if "No module named" in error_details:
            missing_module = error_details.split("No module named ")[-1].strip("'")
            print_warning(f"    Missing dependency: {missing_module}")
            print(f"    Try installing with: pip install {missing_module}")

# Detailed report option
if "--verbose" in sys.argv:
    print_header("Detailed Import Information")
    for module_name, details in import_details.items():
        if details["status"] == "success":
            print_success(f"{module_name}")
            if "attributes" in details:
                imported_modules = [attr for attr in details["attributes"] 
                                  if not attr.startswith("_") and attr not in dir(object)]
                if imported_modules:
                    print("    Imported components:", ", ".join(imported_modules[:5]))
                    if len(imported_modules) > 5:
                        print(f"    ... and {len(imported_modules) - 5} more")
        else:
            print_error(f"{module_name}")
            print(f"    Error: {details['error']}")

print("\nTest completed.")

# Return non-zero exit code if any imports failed
sys.exit(len(failed_imports))
