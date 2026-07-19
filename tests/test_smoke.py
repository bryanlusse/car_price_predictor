"""Smoke tests: the package imports and exposes what CI expects."""

from __future__ import annotations

import model_training


def test_package_importable():
    assert model_training.__version__ == "0.1.0"
