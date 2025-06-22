#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pytest-based Import Tests for sccvi_impl

This script uses pytest to test importing all modules in the sccvi_impl project.
"""

import os
import sys
import importlib
import pytest

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Main package test
def test_main_package_import():
    """Test importing the main package."""
    import src.sccvi_impl
    assert src.sccvi_impl is not None

# Model module tests
def test_model_base_imports():
    """Test importing model base modules."""
    from src.sccvi_impl.model.base import _utils
    assert _utils is not None
    
    from src.sccvi_impl.model.base import model_1_training_mixin
    assert model_1_training_mixin is not None
    
    assert hasattr(_utils, 'SCCAUSALVI_REGISTRY_KEYS')
    assert hasattr(_utils, '_invert_dict')

def test_model_modules():
    """Test importing model modules."""
    from src.sccvi_impl.model import model_1, scCausalVI
    assert model_1 is not None
    assert scCausalVI is not None
    
    assert hasattr(model_1, 'Model1')
    assert hasattr(scCausalVI, 'scCausalVIModel')

# Module implementation tests
def test_module_implementations():
    """Test importing module implementations."""
    from src.sccvi_impl.module import model_1, scCausalVI
    assert model_1 is not None
    assert scCausalVI is not None
    
    assert hasattr(model_1, 'Model1Module')
    assert hasattr(scCausalVI, 'scCausalVIModule')
    
    # Test class attribute access
    assert hasattr(model_1.Model1Module, 'inference')
    assert hasattr(model_1.Model1Module, 'generative')
    assert hasattr(model_1.Model1Module, 'loss')
    
    assert hasattr(scCausalVI.scCausalVIModule, 'inference')
    assert hasattr(scCausalVI.scCausalVIModule, 'generative')
    assert hasattr(scCausalVI.scCausalVIModule, 'loss')

# Data module tests
def test_data_modules():
    """Test importing data modules."""
    from src.sccvi_impl.data import utils, Download_Dataset
    assert utils is not None
    assert Download_Dataset is not None
    
    assert hasattr(utils, 'preprocess')
    assert hasattr(utils, 'get_library_log_means_and_vars')
    assert hasattr(utils, 'gram_matrix')
    
    assert hasattr(Download_Dataset, 'download_dataset')
    assert hasattr(Download_Dataset, 'download_all_datasets')

def test_data_loader_modules():
    """Test importing data loader modules."""
    from src.sccvi_impl.data.dataloaders import data_splitting, scCausalVI_dataloader
    assert data_splitting is not None
    assert scCausalVI_dataloader is not None
    
    assert hasattr(data_splitting, 'scCausalVIDataSplitter')
    assert hasattr(scCausalVI_dataloader, 'scCausalDataLoader')

# Script module tests - these might be incomplete but should at least import
def test_script_modules():
    """Test importing script modules."""
    # These scripts appear to be mostly empty or pseudocode,
    # but they should still import without errors
    import src.sccvi_impl.scripts.simulated_benchmarking
    import src.sccvi_impl.scripts.covid_epithelial_benchmarking
    import src.sccvi_impl.scripts.covid_pbmc_benchmarking
    import src.sccvi_impl.scripts.ifn_beta_benchmarking
    
    # Just checking that import doesn't raise an exception is sufficient here
    assert True

# Module dependency and integration tests
def test_module_dependencies():
    """Test importing modules with their dependencies."""
    # Test importing a model that depends on its module implementation
    from src.sccvi_impl.model import model_1
    from src.sccvi_impl.module import model_1 as module_model_1
    
    # Check that model classes correctly import their module counterparts
    assert model_1.Model1Module == module_model_1.Model1Module
    
    # Test the same for scCausalVI
    from src.sccvi_impl.model import scCausalVI
    from src.sccvi_impl.module import scCausalVI as module_scCausalVI
    
    assert scCausalVI.scCausalVIModule == module_scCausalVI.scCausalVIModule

if __name__ == "__main__":
    # This allows running the tests directly with python
    pytest.main(["-v", __file__])
