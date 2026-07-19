# car_price_predictor

Car Price Predictor — a linear-regression model that estimates used-car prices from
[AutoScout24 listings](https://www.kaggle.com/datasets/clkmuhammed/autoscout24-car-listings-dataset).

## Setup

```bash
uv sync                 # runtime deps only
uv sync --extra dev     # + dev tools (pytest, ruff, mypy, dvc, pre-commit)
```

## Data versioning (DVC)

The raw dataset (~500 MB) is **not** stored in git. It is tracked with
[DVC](https://dvc.org): git holds the small pointer file
`model_training/data/autoscout24_dataset_20251108.csv.dvc`, while the actual bytes live
in a Cloudflare R2 bucket (S3-compatible remote).

### Credentials

Credentials are **never committed**. DVC reads them from the environment (the standard
AWS variables), which works both locally and in CI:

```bash
export AWS_ACCESS_KEY_ID=<r2-access-key-id>
export AWS_SECRET_ACCESS_KEY=<r2-secret-access-key>
```

(Alternatively, `dvc remote modify --local storage access_key_id ...` writes them to the
gitignored `.dvc/config.local`.)

The non-secret remote settings (bucket + endpoint) live in `.dvc/config`. Verify the
`url` and `endpointurl` match your R2 account before pushing.

### Fetch / update data

```bash
dvc pull                # download the dataset referenced by the committed .dvc file
# ...after replacing the CSV with a new version:
dvc add model_training/data/<file>.csv   # re-hash and update the pointer
git add model_training/data/*.dvc && git commit -m "data: update dataset"
dvc push                # upload the new version to R2
```

Because the pointer file is versioned in git, any commit pins an exact dataset version —
`git checkout <sha> && dvc pull` reproduces the data that produced a given model.

## Training

The training entry point is installed as a console script and is safe to run in a CD job
(deterministic, non-interactive, exits non-zero on failure):

```bash
train-model                                  # uses defaults from model_training/config.py
train-model --output-dir models --test-size 0.2 --random-state 42
```

It writes three versionable artifacts to `--output-dir` (default `models/`):

| File            | Contents                                                       |
| --------------- | ------------------------------------------------------------- |
| `model.joblib`  | The fitted preprocessing + `LinearRegression` pipeline        |
| `metrics.json`  | Train/test MAE, RMSE, R²                                       |
| `metadata.json` | Timestamp, sklearn version, data path, row count, run config  |

### Typical CD flow

```bash
uv sync --extra pipeline   # deps + dvc (no dev tooling)
dvc pull                   # fetch the exact pinned dataset
train-model --output-dir models
# gate the pipeline on metrics.json, then archive / register model.joblib
```

## Development

```bash
uv run pytest              # unit tests (run on synthetic data, no dvc pull needed)
uv run ruff check .        # lint
uv run ruff format .       # format
pre-commit run --all-files # all hooks
```
