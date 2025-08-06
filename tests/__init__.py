# Import necessary modules for test setup
import pytest
import torch
import numpy as np
import random

# Set random seeds for reproducibility across all tests
def pytest_configure(config):
    """
    Pytest hook to configure the test environment.
    Sets fixed seeds for torch, numpy, and random to ensure consistent results.
    This is called automatically when pytest runs.
    """
    seed = 42  # Fixed seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Optional: Define shared fixtures or markers
# For example, a marker for GPU tests
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")

# Function to run all tests (convenience wrapper)
def run_all_tests():
    """
    Runs all tests in the package using pytest.
    This can be called from scripts or CI for comprehensive testing.
    
    Returns:
        int: Exit code (0 for success, non-zero for failures).
    
    Note: Equivalent to running 'pytest' from the command line in the tests directory.
    """
    import sys
    exit_code = pytest.main([__file__.rsplit('/', 1)[0]])  # Run on tests dir
    return exit_code

# No __all__ needed for tests; they are executed directly
