"""Publish a locally trained model to the HF Hub model repo.

Runs as a CD step immediately after `train-model`. Publishing always
targets `main`, which *is* the dev environment -- test/prod are promoted
separately and explicitly (see model_registry/promote_env.py), never
auto-updated by this step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from huggingface_hub import CommitInfo, HfApi, ModelCard, ModelCardData

logger = logging.getLogger("car_price_predictor.publish")

REQUIRED_FILES = ("model.joblib", "metrics.json", "metadata.json")


def _build_model_card(metrics: dict, metadata: dict, *, repo_id: str) -> ModelCard:
    card_data = ModelCardData(
        language="en",
        license="mit",
        tags=["tabular-regression", "car-price-prediction", "scikit-learn"],
    )
    content = f"""
# Car Price Predictor (dev)

Auto-published by CD directly from `train-model`. This revision (`main`)
is the **dev** environment -- updates on every successful training run
that clears the RMSE gate. See git tags `test` / `prod` for promoted,
human-approved revisions.

Model type: `{metadata.get("config", {}).get("model_type", "unknown")}`

## Test-split metrics

```json
{json.dumps(metrics.get("test", {}), indent=2)}
```

## How to load a specific environment

```python
from huggingface_hub import hf_hub_download
import joblib

# revision: "main" (dev), "test", or "prod"
path = hf_hub_download(repo_id="{repo_id}", filename="model.joblib", revision="main")
model = joblib.load(path)
```
""".strip()
    return ModelCard.from_template(card_data, template_str="{{ content }}", content=content)


def publish(model_dir: Path, *, repo_id: str) -> CommitInfo:
    missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {model_dir} -- did `train-model` run successfully?"
        )

    metrics = json.loads((model_dir / "metrics.json").read_text())
    metadata = json.loads((model_dir / "metadata.json").read_text())

    card = _build_model_card(metrics, metadata, repo_id=repo_id)
    card.save(model_dir / "README.md")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    commit_info = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        commit_message=f"Publish model (test rmse={metrics.get('test', {}).get('rmse', 'n/a')})",
    )
    logger.info("Published %s -> %s", model_dir, commit_info.commit_url)
    return commit_info


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a trained model to HF Hub.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing model.joblib, metrics.json, metadata.json "
        "(the --output-dir passed to train-model).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF Hub model repo, e.g. yourname/car-price-predictor.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    try:
        publish(args.model_dir, repo_id=args.repo_id)
    except Exception:
        logger.exception("Publish failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
