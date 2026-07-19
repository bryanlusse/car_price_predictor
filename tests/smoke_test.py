"""Smoke tests to verify the package imports and CI is wired up correctly."""

import model_training


def test_package_importable():
    assert model_training.__version__ == "0.1.0"
