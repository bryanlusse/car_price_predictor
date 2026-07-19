"""Train a linear-regression model to predict used-car prices.

Designed to be run both interactively and as a non-interactive CD step:

    train-model --data-path model_training/data/autoscout24_dataset_20251108.csv \
                --output-dir models

The script is deterministic (fixed ``random_state``), reads only the columns it needs,
writes versionable artifacts (model + metrics + run metadata) and exits non-zero on
failure so a pipeline can gate on it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_training.config import AUXILIARY_COLUMNS, TrainingConfig

logger = logging.getLogger("car_price_predictor.train")


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
            ("model", LinearRegression()),
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

    pipeline = build_pipeline(config)
    logger.info("Fitting model on %d rows", len(X_train))
    pipeline.fit(X_train, y_train)

    train_metrics = evaluate(pipeline, X_train, y_train)
    test_metrics = evaluate(pipeline, X_test, y_test)
    logger.info(
        "Test metrics -> MAE: %.0f  RMSE: %.0f  R2: %.3f",
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["r2"],
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


def _parse_args(argv: list[str] | None = None) -> TrainingConfig:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train the car-price linear-regression model.")
    parser.add_argument("--data-path", type=Path, default=defaults.data_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    parser.add_argument("--currency", default=defaults.currency)
    parser.add_argument("--min-price", type=float, default=defaults.min_price)
    parser.add_argument("--max-price", type=float, default=defaults.max_price)
    args = parser.parse_args(argv)
    return TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        currency=args.currency,
        min_price=args.min_price,
        max_price=args.max_price,
    )


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _parse_args(argv)
    try:
        result = train(config)
    except Exception:
        logger.exception("Training failed")
        return 1
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
