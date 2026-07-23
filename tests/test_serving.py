"""Unit tests for the serving core (serving.predictor).

These exercise the framework-free logic against a tiny pipeline built with the real
training code, so they need neither Gradio, the Hub, nor the 500 MB dataset. The Gradio
UI in serving.app is intentionally not imported here (gradio is a Space-only dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_training.config import TrainingConfig
from model_training.train import build_pipeline, prepare_data
from serving.predictor import (
    introspect_features,
    load_pipeline,
    predict_one,
)


def _make_raw_frame(n: int = 200) -> pd.DataFrame:
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


@pytest.fixture
def fitted_pipeline():
    config = TrainingConfig()
    prepared = prepare_data(_make_raw_frame(), config)
    pipeline = build_pipeline(config)
    pipeline.fit(prepared[config.feature_columns], prepared[config.target])
    return pipeline


def _sample_payload() -> dict:
    return {
        "mileage_km_raw": 80_000,
        "power_hp": 110,
        "power_kw": 81,
        "nr_seats": 5,
        "nr_doors": 5,
        "gears": 6,
        "cylinders": 4,
        "cylinders_volume_cc": 1600,
        "co2_emission_grper_km": 120,
        "fuel_cons_comb_l100_km": 5.5,
        "vehicle_age_years": 6,
        "make": "BMW",
        "transmission": "Manual",
        "drive_train": "4WD",
        "fuel_category": "Diesel",
        "body_type": "Coupe",
        "vehicle_type": "Car",
        "seller_type": "Dealer",
    }


def test_introspect_recovers_feature_contract(fitted_pipeline):
    spec = introspect_features(fitted_pipeline)

    # Numeric side includes the engineered age; categorical side carries vocabularies.
    assert "vehicle_age_years" in spec.numeric
    assert "make" in spec.categorical
    assert set(spec.categories["make"]).issubset({"BMW", "Audi", "Suzuki", "Porsche"})
    assert set(spec.columns) == set(TrainingConfig().feature_columns)


def test_predict_one_returns_finite_float(fitted_pipeline):
    spec = introspect_features(fitted_pipeline)
    price = predict_one(fitted_pipeline, spec, _sample_payload())
    assert isinstance(price, float)
    assert np.isfinite(price)


def test_predict_one_tolerates_missing_and_unknown(fitted_pipeline):
    spec = introspect_features(fitted_pipeline)
    payload = {"mileage_km_raw": "120000", "make": "NeverSeenBrand"}  # partial + unknown + string
    price = predict_one(fitted_pipeline, spec, payload)
    assert np.isfinite(price)


def test_load_pipeline_roundtrip(fitted_pipeline, tmp_path):
    import joblib

    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)

    loaded = load_pipeline(path)
    spec = introspect_features(loaded)
    assert np.isfinite(predict_one(loaded, spec, _sample_payload()))


def test_load_pipeline_rejects_non_pipeline(tmp_path):
    import joblib

    path = tmp_path / "junk.joblib"
    joblib.dump({"not": "a pipeline"}, path)
    with pytest.raises(TypeError):
        load_pipeline(path)
