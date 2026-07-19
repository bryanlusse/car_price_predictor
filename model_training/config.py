"""Central configuration for the car-price training pipeline.

Keeping feature lists, paths and hyper-parameters in one importable place means the
training script, the tests and any future serving/CD code all agree on the exact same
contract. Everything here is a plain dataclass with defaults, so it can be constructed
from CLI args, environment variables or a config file without changing the rest of the
code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# MLFlow setup
MLFLOW_EXPERIMENT_NAME = "car_price_predictor"
MLFLOW_ARTIFACT_URI = os.environ.get(
    "MLFLOW_ARTIFACT_URI", "s3://car-price-predictor/mlflow-artifacts"
)

SUPPORTED_MODEL_TYPES: list[str] = [
    "linear_regression",
    "ridge",
    "lasso",
    "random_forest",
    "gradient_boosting",
]

# Repo root = two levels up from this file (model_training/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Column that we predict.
TARGET = "price"

# Raw columns consumed from the CSV. Reading only these keeps memory + parse time low
# (the raw file is ~500 MB, most of which is the free-text `description` column).
NUMERIC_FEATURES: list[str] = [
    "mileage_km_raw",
    "power_hp",
    "power_kw",
    "nr_seats",
    "nr_doors",
    "gears",
    "cylinders",
    "cylinders_volume_cc",
    "co2_emission_grper_km",
    "fuel_cons_comb_l100_km",
]

CATEGORICAL_FEATURES: list[str] = [
    "make",
    "transmission",
    "drive_train",
    "fuel_category",
    "body_type",
    "vehicle_type",
    "seller_type",
]

# Derived numeric feature engineered in train.py from `registration_date`.
DERIVED_NUMERIC_FEATURES: list[str] = ["vehicle_age_years"]

# Extra raw columns we need to load for filtering / feature engineering but do not feed
# to the model directly.
AUXILIARY_COLUMNS: list[str] = ["price_currency", "registration_date"]


@dataclass
class TrainingConfig:
    """All knobs for a single training run.

    Defaults reproduce the standard run; a CD job overrides fields via CLI flags.
    """

    data_path: Path = REPO_ROOT / "model_training" / "data" / "autoscout24_dataset_20251108.csv"
    output_dir: Path = REPO_ROOT / "models"

    target: str = TARGET
    numeric_features: list[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    categorical_features: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))

    model_type: str = "linear_regression"
    model_params: dict[str, Any] = field(default_factory=dict)

    # Only keep listings priced in this currency (the dataset mixes a few).
    currency: str = "EUR"
    # Guard-rail bounds to drop obviously broken / non-representative prices.
    min_price: float = 500.0
    max_price: float = 500_000.0
    # Reference year used to turn a registration date into a vehicle age.
    reference_year: int = 2025

    test_size: float = 0.2
    random_state: int = 42

    # Rare-category collapsing for the one-hot encoder (keeps make/model dimensionality sane).
    min_category_frequency: float = 0.01

    @property
    def feature_columns(self) -> list[str]:
        """Model input columns after feature engineering."""
        return self.numeric_features + DERIVED_NUMERIC_FEATURES + self.categorical_features

    @property
    def numeric_model_features(self) -> list[str]:
        return self.numeric_features + DERIVED_NUMERIC_FEATURES
