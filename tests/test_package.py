"""Basic package tests."""

import harmony_trainer


def test_version():
    assert harmony_trainer.__version__ == "0.1.0"


def test_package_importable():
    """Verify the package can be imported without errors."""
    assert hasattr(harmony_trainer, "__version__")
