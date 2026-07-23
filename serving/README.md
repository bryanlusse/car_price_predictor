---
title: Car Price Predictor
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Car Price Predictor -- serving app

Gradio app that serves the [`blusse7/car-price-predictor`](https://huggingface.co/blusse7/car-price-predictor)
model: enter a used car's listing attributes and get an estimated resale price (EUR).

Auto-deployed to a Hugging Face Space by the CD workflow whenever a training run clears
the RMSE gate and publishes a new model. Dependencies are resolved with **uv** from
`pyproject.toml` (no `requirements.txt`).

## Configuration

The app is configured entirely through environment variables (set them as Space secrets
/ variables), so the same code serves any environment:

| Variable            | Default                        | Purpose                                        |
| ------------------- | ------------------------------ | ---------------------------------------------- |
| `HF_MODEL_REPO_ID`  | `blusse7/car-price-predictor`  | HF Hub model repo to load.                     |
| `MODEL_REVISION`    | `main`                         | Revision to serve: `main` (dev) / `test` / `prod`. |
| `HF_TOKEN`          | *(unset)*                      | Only needed if the model repo is private.      |

## Endpoints

- **UI** -- a form at the Space root.
- **HTTP API** -- `api_name="/predict"`, taking a single JSON string of `feature -> value`
  and returning `{"price": <float>, "currency": "EUR"}`. Unknown keys are ignored and
  missing features are imputed, so a partial payload is accepted. This is what the
  `inference.client` smoke test calls.

## Local run

```bash
uv run --project serving python serving/app.py
# then open http://localhost:7860
```
