"""Tests for configurable model type / hyperparameters (MODEL_REGISTRY, _build_model,
CLI parsing) added to support comparing multiple models via MLflow.

Reuses the synthetic-data pattern from test_train.py so these stay fast and don't
require `dvc pull`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from model_training.config import TrainingConfig
from model_training.train import MODEL_REGISTRY, _build_model, _parse_args, build_pipeline, train


def _make_raw_frame(n: int = 200) -> pd.DataFrame:
    """Build a synthetic raw frame mirroring the columns train.py reads."""
    rng = np.random.default_rng(0)
    makes = np.array(["BMW", "Audi", "Suzuki", "Porsche"])
    return pd.DataFrame(
        {
            "price": rng.uniform(2_000, 80_000, n),
            "price_currency": "EUR",
            "registration_date": "2020-01-01",
            "mileage_km_raw": rng.uniform(0, 200_000, n),
            "power_hp": rng.uniform(60, 500, n),
            "power_kw": rng.uniform(45, 370, n),
            "nr_seats": rng.integers(2, 8, n),
            "nr_doors": rng.integers(2, 6, n),
            "gears": rng.integers(4, 9, n),
            "cylinders": rng.integers(3, 8, n),
            "cylinders_volume_cc": rng.uniform(900, 4000, n),
            "co2_emission_grper_km": rng.uniform(80, 300, n),
            "fuel_cons_comb_l100_km": rng.uniform(3, 12, n),
            "make": rng.choice(makes, n),
            "transmission": rng.choice(["Manual", "Automatic"], n),
            "drive_train": rng.choice(["Front Wheel Drive", "4WD"], n),
            "fuel_category": rng.choice(["Gasoline", "Diesel"], n),
            "body_type": rng.choice(["Compact", "Coupe", "Off-Road/Pick-up"], n),
            "vehicle_type": "Car",
            "seller_type": rng.choice(["Dealer", "PrivateSeller"], n),
        }
    )


# ---------------------------------------------------------------------------
# MODEL_REGISTRY / _build_model
# ---------------------------------------------------------------------------


def test_registry_contains_expected_model_types():
    assert set(MODEL_REGISTRY) == {
        "linear_regression",
        "ridge",
        "lasso",
        "random_forest",
        "gradient_boosting",
    }


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("linear_regression", LinearRegression),
        ("ridge", Ridge),
        ("lasso", Lasso),
        ("random_forest", RandomForestRegressor),
        ("gradient_boosting", GradientBoostingRegressor),
    ],
)
def test_build_model_returns_configured_estimator_type(model_type, expected_cls):
    config = TrainingConfig(model_type=model_type)
    model = _build_model(config)
    assert isinstance(model, expected_cls)


def test_build_model_applies_hyperparameters():
    config = TrainingConfig(model_type="ridge", model_params={"alpha": 5.0})
    model = _build_model(config)
    assert model.alpha == 5.0


def test_build_model_injects_random_state_when_supported():
    config = TrainingConfig(model_type="random_forest", random_state=123)
    model = _build_model(config)
    assert model.random_state == 123


def test_build_model_explicit_random_state_param_wins_over_config():
    # model_params should take precedence over the auto-injected config.random_state.
    config = TrainingConfig(
        model_type="random_forest", random_state=123, model_params={"random_state": 7}
    )
    model = _build_model(config)
    assert model.random_state == 7


def test_build_model_does_not_error_for_model_without_random_state():
    # LinearRegression has no random_state param; injection must be skipped, not error.
    config = TrainingConfig(model_type="linear_regression", random_state=123)
    model = _build_model(config)
    assert not hasattr(model, "random_state")


def test_build_model_unknown_model_type_raises_value_error():
    config = TrainingConfig(model_type="not_a_real_model")
    with pytest.raises(ValueError, match="Unknown model_type"):
        _build_model(config)


# ---------------------------------------------------------------------------
# build_pipeline / train end-to-end with a non-default model
# ---------------------------------------------------------------------------


def test_pipeline_fits_and_predicts_with_random_forest():
    config = TrainingConfig(
        model_type="random_forest", model_params={"n_estimators": 10, "max_depth": 3}
    )
    raw = _make_raw_frame()
    from model_training.train import prepare_data

    prepared = prepare_data(raw, config)
    pipeline = build_pipeline(config)
    pipeline.fit(prepared[config.feature_columns], prepared[config.target])
    preds = pipeline.predict(prepared[config.feature_columns])
    assert preds.shape == (len(prepared),)
    assert np.isfinite(preds).all()
    assert isinstance(pipeline.named_steps["model"], RandomForestRegressor)


def test_train_writes_artifacts_with_ridge_model(tmp_path):
    data_path = tmp_path / "raw.csv"
    _make_raw_frame(300).to_csv(data_path, index=False)
    config = TrainingConfig(
        data_path=data_path,
        output_dir=tmp_path / "models",
        model_type="ridge",
        model_params={"alpha": 2.0},
    )

    result = train(config)

    assert (config.output_dir / "model.joblib").exists()
    assert result["metadata"]["config"]["model_type"] == "ridge"
    assert result["metrics"]["test"]["n_samples"] > 0


def test_different_model_types_produce_different_predictions(tmp_path):
    """Sanity check that model_type actually changes the fitted model, not just its label."""
    data_path = tmp_path / "raw.csv"
    _make_raw_frame(300).to_csv(data_path, index=False)

    linear_config = TrainingConfig(
        data_path=data_path, output_dir=tmp_path / "linear", model_type="linear_regression"
    )
    forest_config = TrainingConfig(
        data_path=data_path,
        output_dir=tmp_path / "forest",
        model_type="random_forest",
        model_params={"n_estimators": 10, "max_depth": 3},
    )

    linear_result = train(linear_config)
    forest_result = train(forest_config)

    # Different model families on the same data/split should not produce identical metrics.
    assert linear_result["metrics"]["test"] != forest_result["metrics"]["test"]


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_linear_regression():
    config = _parse_args([])
    assert config.model_type == "linear_regression"
    assert config.model_params == {}


def test_parse_args_reads_model_type_and_params():
    config = _parse_args(
        ["--model-type", "random_forest", "--model-params", '{"n_estimators": 50, "max_depth": 4}']
    )
    assert config.model_type == "random_forest"
    assert config.model_params == {"n_estimators": 50, "max_depth": 4}


def test_parse_args_rejects_unknown_model_type():
    with pytest.raises(SystemExit):
        _parse_args(["--model-type", "not_a_real_model"])


def test_parse_args_rejects_invalid_json_model_params():
    with pytest.raises(SystemExit):
        _parse_args(["--model-params", "{not valid json}"])
