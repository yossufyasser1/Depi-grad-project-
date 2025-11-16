"""
PyTest configuration file for test fixtures and setup.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_models_dir(tmp_path_factory):
    """Create a temporary directory for test models."""
    return tmp_path_factory.mktemp("test_models")


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment before each test."""
    # Any cleanup or setup needed before each test
    yield
    # Cleanup after test
