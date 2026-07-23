"""Unit tests for the training pipeline that run on tiny synthetic data.

These deliberately avoid touching the real 500 MB dataset so they stay fast and can run
in CI without `dvc pull`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_training.config import TrainingConfig
from model_training.train import build_pipeline, prepare_data, train


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


def test_prepare_data_engineers_age_and_filters():
    config = TrainingConfig(reference_year=2025)
    raw = _make_raw_frame()
    prepared = prepare_data(raw, config)

    assert "vehicle_age_years" in prepared.columns
    assert (prepared["vehicle_age_years"] == 5).all()  # 2025 - 2020
    # Only feature columns + target survive.
    assert set(prepared.columns) == set(config.feature_columns + [config.target])


def test_prepare_data_drops_out_of_bounds_prices():
    config = TrainingConfig(min_price=1_000, max_price=100_000)
    raw = _make_raw_frame()
    raw.loc[0, "price"] = 10  # too cheap
    raw.loc[1, "price"] = 10_000_000  # too expensive
    prepared = prepare_data(raw, config)
    assert prepared["price"].between(1_000, 100_000).all()


def test_pipeline_fits_and_predicts():
    config = TrainingConfig()
    raw = _make_raw_frame()
    prepared = prepare_data(raw, config)
    pipeline = build_pipeline(config)
    pipeline.fit(prepared[config.feature_columns], prepared[config.target])
    preds = pipeline.predict(prepared[config.feature_columns])
    assert preds.shape == (len(prepared),)
    assert np.isfinite(preds).all()


def test_train_writes_artifacts(tmp_path):
    data_path = tmp_path / "raw.csv"
    _make_raw_frame(300).to_csv(data_path, index=False)
    config = TrainingConfig(data_path=data_path, output_dir=tmp_path / "models")

    result = train(config)

    assert (config.output_dir / "model.joblib").exists()
    assert (config.output_dir / "metrics.json").exists()
    assert (config.output_dir / "metadata.json").exists()
    assert set(result["metrics"]) == {"train", "test"}
    assert result["metrics"]["test"]["n_samples"] > 0


def test_read_raw_missing_file_raises():
    from model_training.train import _read_raw

    config = TrainingConfig(data_path=tmp_missing())
    with pytest.raises(FileNotFoundError):
        _read_raw(config)


def test_main_passes_when_gate_unset(tmp_path):
    from model_training.train import main

    data_path = tmp_path / "raw.csv"
    _make_raw_frame(300).to_csv(data_path, index=False)

    exit_code = main(["--data-path", str(data_path), "--output-dir", str(tmp_path / "models")])

    assert exit_code == 0


def test_main_fails_when_rmse_exceeds_gate(tmp_path):
    from model_training.train import main

    data_path = tmp_path / "raw.csv"
    _make_raw_frame(300).to_csv(data_path, index=False)

    # Synthetic data is random noise w.r.t. price, so RMSE will be large --
    # an impossibly tight gate should reliably fail it.
    exit_code = main(
        [
            "--data-path",
            str(data_path),
            "--output-dir",
            str(tmp_path / "models"),
            "--rmse-gate",
            "0.01",
        ]
    )

    assert exit_code == 1


def tmp_missing():
    return TrainingConfig().output_dir / "does-not-exist.csv"
