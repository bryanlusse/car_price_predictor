"""Model loading and single-row prediction -- the framework-free core of the app.

Kept deliberately free of Gradio and of any import-time network access so it can be
unit-tested against a locally built pipeline. `app.py` wires this into a Gradio UI and
`inference/client.py` never touches it (it hits the deployed HTTP endpoint instead).

The served artifact is the exact ``sklearn.Pipeline`` that ``model_training`` fits, so
preprocessing (imputation, scaling, one-hot encoding) travels with the model and serving
cannot skew from training. That also means we can *introspect* the fitted pipeline for the
input contract -- feature names and the categorical vocabularies the encoder saw -- rather
than duplicating the feature list from ``model_training.config`` (which the standalone
Space cannot import).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger("car_price_predictor.serving")


class FeatureSpec:
    """The input contract, recovered from the fitted pipeline's ColumnTransformer."""

    def __init__(
        self,
        numeric: list[str],
        categorical: list[str],
        categories: dict[str, list[str]],
    ) -> None:
        self.numeric = numeric
        self.categorical = categorical
        # Known category values per categorical column (for populating dropdowns). The
        # encoder is set to ``handle_unknown="infrequent_if_exist"``, so values outside
        # these lists are tolerated at predict time rather than raising.
        self.categories = categories

    @property
    def columns(self) -> list[str]:
        """All model input columns. Order is irrelevant -- the ColumnTransformer selects
        by name -- but a stable order makes the UI and API deterministic."""
        return self.numeric + self.categorical


def load_pipeline_from_hub(
    repo_id: str, *, revision: str = "main", token: str | None = None
) -> Pipeline:
    """Download ``model.joblib`` from the HF Hub model repo and deserialize it.

    ``revision`` selects the environment: ``main`` (dev), or the ``test`` / ``prod`` tags
    written by the promotion step. Imported lazily so this module stays importable (and
    testable) without ``huggingface_hub`` installed.
    """
    from huggingface_hub import hf_hub_download

    logger.info("Downloading model.joblib from %s@%s", repo_id, revision)
    local_path = hf_hub_download(
        repo_id=repo_id, filename="model.joblib", revision=revision, token=token
    )
    return load_pipeline(local_path)


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a pipeline from a local ``.joblib`` file."""
    pipeline = joblib.load(Path(path))
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected an sklearn Pipeline, got {type(pipeline).__name__}")
    return pipeline


def introspect_features(pipeline: Pipeline) -> FeatureSpec:
    """Recover the input contract from the fitted ``preprocessor`` ColumnTransformer.

    Mirrors the structure built in ``model_training.train.build_pipeline``: a ``num``
    transformer over the numeric columns and a ``cat`` transformer whose final step is a
    fitted ``OneHotEncoder`` exposing ``categories_``.
    """
    try:
        pre = pipeline.named_steps["preprocessor"]
    except (AttributeError, KeyError) as err:
        raise ValueError("Pipeline has no 'preprocessor' step to introspect.") from err

    numeric: list[str] = []
    categorical: list[str] = []
    categories: dict[str, list[str]] = {}

    for name, transformer, cols in pre.transformers_:
        cols = list(cols)
        if name == "num":
            numeric = cols
        elif name == "cat":
            categorical = cols
            onehot = transformer
            if hasattr(transformer, "named_steps"):  # a sub-Pipeline (imputer + onehot)
                onehot = transformer.named_steps.get("onehot", transformer)
            encoder_categories = getattr(onehot, "categories_", None)
            if encoder_categories is not None:
                for col, cats in zip(cols, encoder_categories, strict=True):
                    # Drop the NaN sentinel the imputer/encoder may carry.
                    values = [c for c in cats.tolist() if isinstance(c, str)]
                    categories[col] = sorted(values)

    if not numeric and not categorical:
        raise ValueError("Could not recover any feature columns from the pipeline.")
    return FeatureSpec(numeric=numeric, categorical=categorical, categories=categories)


def predict_one(pipeline: Pipeline, spec: FeatureSpec, payload: dict[str, Any]) -> float:
    """Predict a single price from a feature -> value mapping.

    Tolerant by design: unknown keys are ignored, missing features become NaN/None (the
    pipeline's imputers fill them), numeric strings are coerced. This keeps the HTTP API
    forgiving for callers that send a partial or loosely-typed payload.
    """
    row = {col: payload.get(col) for col in spec.columns}
    frame = pd.DataFrame([row], columns=spec.columns)

    for col in spec.numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in spec.categorical:
        # Empty strings mean "not provided" -> let the imputer handle it.
        frame[col] = frame[col].replace("", np.nan)

    prediction = pipeline.predict(frame)
    return float(np.asarray(prediction).ravel()[0])
