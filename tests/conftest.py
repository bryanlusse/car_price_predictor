"""Shared pytest fixtures.

Notably: an autouse fixture that isolates MLflow for every test. `train()` logs to
MLflow (params, metrics, and the fitted model) as part of the normal training flow.
Without this fixture, that logging targets the real tracking/artifact store configured
for local dev or CI (an R2/S3 bucket), which requires real credentials and network
access -- neither of which should be a precondition for running the unit test suite.
"""

from __future__ import annotations

import mlflow
import pytest

import model_training.train as train_module


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path, monkeypatch):
    """Redirect MLflow to a throwaway local tracking + artifact store per test."""
    tracking_dir = tmp_path / "mlflow-tracking"
    artifact_dir = tmp_path / "mlflow-artifacts"
    tracking_dir.mkdir()
    artifact_dir.mkdir()

    # The plain `file:` tracking backend is deprecated in newer MLflow versions (it
    # raises, pointing at a database backend instead) -- sqlite is the lightweight
    # equivalent for tests. The artifact store has no such restriction, so it stays
    # a local `file:` URI.
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")

    # MLFLOW_ARTIFACT_URI is read once at import time in model_training.config, so
    # patching the env var alone wouldn't affect an already-imported module. Patch the
    # name where train.py actually uses it instead.
    monkeypatch.setattr(train_module, "MLFLOW_ARTIFACT_URI", f"file:{artifact_dir}")

    for var in (
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(var, raising=False)

    # mlflow.sklearn.log_model also infers/pins the current Python environment (shells
    # out to `uv export`/pip to build conda.yaml/requirements.txt) on every call. That's
    # slow and orthogonal to what these tests check -- we're verifying *our* code calls
    # MLflow correctly, not re-testing MLflow's own packaging internals -- so it's
    # stubbed out to a fast no-op for the unit test suite.
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *args, **kwargs: None)

    yield
