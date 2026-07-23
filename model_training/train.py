"""Train a car-price prediction model (linear regression, ridge, lasso, random forest,
or gradient boosting -- see --model-type).

Designed to be run both interactively and as a non-interactive CD step:

    train-model --data-path model_training/data/autoscout24_dataset_20251108.csv \
                --output-dir models

The script is deterministic (fixed ``random_state``), reads only the columns it needs,
writes versionable artifacts (model + metrics + run metadata) and exits non-zero on
failure so a pipeline can gate on it.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_training.config import (
    AUXILIARY_COLUMNS,
    MLFLOW_ARTIFACT_URI,
    MLFLOW_EXPERIMENT_NAME,
    TrainingConfig,
)

logger = logging.getLogger("car_price_predictor.train")

MODEL_REGISTRY: dict[str, type] = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def _read_raw(config: TrainingConfig) -> pd.DataFrame:
    """Load only the columns the pipeline needs from the (large) raw CSV."""
    usecols = (
        [config.target] + config.numeric_features + config.categorical_features + AUXILIARY_COLUMNS
    )
    logger.info("Reading %s (%d columns)", config.data_path, len(usecols))
    if not config.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.data_path}. Run `dvc pull` to fetch it from the remote."
        )
    df = pd.read_csv(config.data_path, usecols=usecols, low_memory=False)
    logger.info("Loaded %d raw rows", len(df))
    return df


def _get_or_create_experiment(name: str, artifact_location: str) -> str:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(name, artifact_location=artifact_location)


def prepare_data(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Filter to usable rows and engineer derived features.

    Returns a frame with exactly ``config.feature_columns`` + target, all numeric
    columns coerced, ready to be split and fed to the pipeline.
    """
    df = df.copy()

    # Keep a single currency so `price` is comparable, and drop rows without a target.
    df = df[df["price_currency"] == config.currency]
    df = df.dropna(subset=[config.target])

    # Coerce numeric feature columns (some arrive as strings / with stray tokens).
    for col in config.numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Engineer vehicle age from the registration date.
    reg_year = pd.to_datetime(df["registration_date"], errors="coerce").dt.year
    df["vehicle_age_years"] = config.reference_year - reg_year

    # Sanity bounds on the target to drop broken listings and extreme outliers.
    df = df[(df[config.target] >= config.min_price) & (df[config.target] <= config.max_price)]

    keep = config.feature_columns + [config.target]
    df = df[keep]
    logger.info("Prepared %d rows after filtering", len(df))
    if df.empty:
        raise ValueError("No rows left after filtering; check currency/price bounds.")
    return df


def _build_model(config: TrainingConfig):
    try:
        model_cls = MODEL_REGISTRY[config.model_type]
    except KeyError as err:
        raise ValueError("Unknown model_type") from err
    params = dict(config.model_params)
    if "random_state" in inspect.signature(model_cls).parameters:
        params.setdefault("random_state", config.random_state)
    return model_cls(**params)


def build_pipeline(config: TrainingConfig) -> Pipeline:
    """Assemble the preprocessing + linear-regression pipeline.

    A single ``Pipeline`` object means preprocessing is fit on train data only and
    travels with the model when it is serialized, so serving cannot skew from training.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=config.min_category_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.numeric_model_features),
            ("cat", categorical_transformer, config.categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", _build_model(config=config)),
        ]
    )


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute regression metrics on a held-out split."""
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": rmse,
        "r2": float(r2_score(y, preds)),
        "n_samples": int(len(y)),
    }


def train(config: TrainingConfig) -> dict:
    """Run the full training flow and persist artifacts. Returns the run summary."""
    raw = _read_raw(config)
    data = prepare_data(raw, config)

    X = data[config.feature_columns]
    y = data[config.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    experiment_id = _get_or_create_experiment(MLFLOW_EXPERIMENT_NAME, MLFLOW_ARTIFACT_URI)
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params({k: str(v) for k, v in asdict(config).items()})

        pipeline = build_pipeline(config)
        pipeline.fit(X_train, y_train)

        train_metrics = evaluate(pipeline, X_train, y_train)
        test_metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.sklearn.log_model(
            pipeline, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / "model.joblib"
    metrics_path = config.output_dir / "metrics.json"
    metadata_path = config.output_dir / "metadata.json"

    joblib.dump(pipeline, model_path)

    metrics = {"train": train_metrics, "test": test_metrics}
    metrics_path.write_text(json.dumps(metrics, indent=2))

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "data_path": str(config.data_path),
        "n_rows_used": int(len(data)),
        "feature_columns": config.feature_columns,
        "target": config.target,
        "config": {k: str(v) for k, v in asdict(config).items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved model -> %s", model_path)
    return {"model_path": str(model_path), "metrics": metrics, "metadata": metadata}


def _parse_args(argv: list[str] | None = None) -> tuple[TrainingConfig, float | None]:
    parser = argparse.ArgumentParser(description="Train a car-price model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML champion config (model_type, model_params, frozen data "
        "knobs). This is the single source of truth CD trains from. Any flag below "
        "overrides the corresponding field for ad-hoc runs.",
    )
    # Overridable fields default to None so we can tell 'not passed' from an explicit
    # value: a passed flag wins over the --config file, which wins over dataclass defaults.
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to the raw AutoScout24 CSV. Must be present locally "
        "(run `dvc pull` first if it isn't).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write model.joblib, metrics.json, and metadata.json into. "
        "Created if it doesn't exist.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of rows held out for the test split, e.g. 0.2 = 20%% test / 80%% train.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Seed controlling the train/test split and, where supported, the model's "
        "own randomness (e.g. RandomForestRegressor). Fix this to keep runs reproducible.",
    )
    parser.add_argument(
        "--currency",
        default=None,
        help="Only listings priced in this currency are kept (the raw dataset mixes a few).",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=None,
        help="Drop listings priced below this (guards against broken/placeholder prices).",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Drop listings priced above this (guards against extreme outliers).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=sorted(MODEL_REGISTRY),
        help="Which estimator to train. See MODEL_REGISTRY for the full mapping.",
    )
    parser.add_argument(
        "--model-params",
        type=json.loads,
        default=None,
        help="Hyperparameters for the chosen --model-type, as a JSON object, e.g. "
        '\'{"alpha": 1.0}\' for ridge/lasso or \'{"n_estimators": 300, "max_depth": 8}\' '
        "for the tree ensembles. Unset params fall back to sklearn's own defaults.",
    )
    parser.add_argument(
        "--rmse-gate",
        type=float,
        default=None,
        help="If set, exit non-zero when test RMSE exceeds this value -- lets a CD "
        "pipeline block a regression from being published. Unset = no gate.",
    )
    args = parser.parse_args(argv)

    base = TrainingConfig.from_file(args.config) if args.config else TrainingConfig()
    overridable = (
        "data_path",
        "output_dir",
        "test_size",
        "random_state",
        "currency",
        "min_price",
        "max_price",
        "model_type",
        "model_params",
    )
    overrides = {
        name: getattr(args, name) for name in overridable if getattr(args, name) is not None
    }
    return replace(base, **overrides), args.rmse_gate


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config, rmse_gate = _parse_args(argv)
    try:
        result = train(config)
    except Exception:
        logger.exception("Training failed")
        return 1
    print(json.dumps(result["metrics"], indent=2))

    test_rmse = result["metrics"]["test"]["rmse"]
    if rmse_gate is not None and test_rmse > rmse_gate:
        logger.error("Quality gate failed: test RMSE %.2f > gate %.2f", test_rmse, rmse_gate)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
