"""Tests for champion-config loading and the file/flag override precedence."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from model_training.config import REPO_ROOT, TrainingConfig
from model_training.train import _parse_args


def test_from_dict_applies_partial_overrides():
    config = TrainingConfig.from_dict({"model_type": "ridge", "model_params": {"alpha": 2.0}})
    assert config.model_type == "ridge"
    assert config.model_params == {"alpha": 2.0}
    # Untouched fields keep their defaults.
    assert config.currency == "EUR"


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown config keys"):
        TrainingConfig.from_dict({"modeltype": "ridge"})  # typo


def test_from_dict_coerces_paths():
    config = TrainingConfig.from_dict({"data_path": "some/dir/cars.csv"})
    assert isinstance(config.data_path, Path)


def test_from_file_reads_yaml(tmp_path):
    path = tmp_path / "champion.yaml"
    path.write_text(
        textwrap.dedent(
            """
            model_type: gradient_boosting
            model_params:
              learning_rate: 0.05
              n_estimators: 300
            random_state: 7
            """
        )
    )
    config = TrainingConfig.from_file(path)
    assert config.model_type == "gradient_boosting"
    assert config.model_params == {"learning_rate": 0.05, "n_estimators": 300}
    assert config.random_state == 7


def test_shipped_champion_file_is_valid():
    """The committed champion config must always load -- CD depends on it."""
    config = TrainingConfig.from_file(REPO_ROOT / "model_training" / "champion.yaml")
    assert config.model_type in {
        "linear_regression",
        "ridge",
        "lasso",
        "random_forest",
        "gradient_boosting",
    }


def test_cli_flags_override_config_file(tmp_path):
    path = tmp_path / "champion.yaml"
    path.write_text("model_type: gradient_boosting\nrandom_state: 1\n")

    # File sets gradient_boosting; the flag must win, while unflagged fields come from file.
    config, gate = _parse_args(["--config", str(path), "--model-type", "ridge"])
    assert config.model_type == "ridge"
    assert config.random_state == 1
    assert gate is None


def test_cli_without_config_uses_defaults():
    config, gate = _parse_args([])
    assert config.model_type == TrainingConfig().model_type
    assert gate is None
