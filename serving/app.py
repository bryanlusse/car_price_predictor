"""Gradio serving app for the car-price predictor -- the entry point on the HF Space.

Only the ``serving/`` folder is uploaded to the Space (see the CD workflow), and the
upload deliberately drops ``__init__.py`` so the app runs *flat* there (``import
predictor``). Locally the same folder is a package (``serving.predictor``) so the unit
tests and tooling can import it. The dual import below bridges the two layouts.

Configuration is entirely via environment variables so the same image serves any
environment without a code change:

    HF_MODEL_REPO_ID   HF Hub model repo to serve, e.g. blusse7/car-price-predictor.
    MODEL_REVISION     Which revision to load: "main" (dev, default) / "test" / "prod".
    HF_TOKEN           Only needed if the model repo is private.
"""

from __future__ import annotations

import json
import logging
import os
from itertools import islice

import gradio as gr

try:  # local package layout (tests, `python -m serving.app`)
    from serving.predictor import (
        FeatureSpec,
        introspect_features,
        load_pipeline_from_hub,
        predict_one,
    )
except ImportError:  # flat layout on the HF Space (__init__.py is not uploaded)
    from predictor import (  # type: ignore[no-redef]
        FeatureSpec,
        introspect_features,
        load_pipeline_from_hub,
        predict_one,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("car_price_predictor.serving.app")

DEFAULT_MODEL_REPO_ID = "blusse7/car-price-predictor"

# Nicer labels for the raw feature names shown in the form.
FEATURE_LABELS = {
    "mileage_km_raw": "Mileage (km)",
    "power_hp": "Power (hp)",
    "power_kw": "Power (kW)",
    "nr_seats": "Number of seats",
    "nr_doors": "Number of doors",
    "gears": "Gears",
    "cylinders": "Cylinders",
    "cylinders_volume_cc": "Engine volume (cc)",
    "co2_emission_grper_km": "CO2 emission (g/km)",
    "fuel_cons_comb_l100_km": "Fuel consumption (l/100km)",
    "vehicle_age_years": "Vehicle age (years)",
    "make": "Make",
    "transmission": "Transmission",
    "drive_train": "Drive train",
    "fuel_category": "Fuel category",
    "body_type": "Body type",
    "vehicle_type": "Vehicle type",
    "seller_type": "Seller type",
}

# Short, friendly hints shown under each field. Falls back to nothing if a column
# isn't listed here, so new features never break the form.
FEATURE_HINTS = {
    "mileage_km_raw": "Total distance driven, in kilometers.",
    "power_hp": "Engine power in horsepower.",
    "power_kw": "Engine power in kilowatts.",
    "vehicle_age_years": "Years since first registration.",
    "co2_emission_grper_km": "Official CO2 rating, grams per km.",
    "fuel_cons_comb_l100_km": "Combined fuel use, liters per 100 km.",
    "nr_seats": "Total seating capacity.",
    "nr_doors": "Number of doors, including the trunk/hatch.",
    "gears": "Number of gears in the transmission.",
    "cylinders": "Number of engine cylinders.",
    "cylinders_volume_cc": "Total engine displacement, in cc.",
}
# Every numeric field should show *some* hint, even a generic one, so that fields in
# the same row all render at the same height (a field without `info=` is shorter than
# one with it, which otherwise makes rows look visually misaligned).
_DEFAULT_HINT = "See your listing details."

PLACEHOLDER_RESULT = """
<div style="text-align:center; padding: 2.25rem 0.5rem;">
  <div style="font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase;
              color: var(--body-text-color-subdued);">Estimated resale price</div>
  <div style="font-size:2.25rem; font-weight:700;
  letter-spacing:-0.02em; color: var(--body-text-color-subdued); margin-top:0.35rem;">
    —
  </div>
  <div style="font-size:0.85rem; color: var(--body-text-color-subdued); margin-top:0.5rem;">
    Fill in the details and press <b>Predict price</b>.
  </div>
</div>
"""


def _label(col: str) -> str:
    return FEATURE_LABELS.get(col, col.replace("_", " ").capitalize())


def _chunk(items: list, size: int):
    """Yield successive ``size``-length slices of ``items``."""
    it = iter(items)
    while chunk := list(islice(it, size)):
        yield chunk


def build_demo(pipeline, spec: FeatureSpec) -> gr.Blocks:
    """Assemble the Gradio UI plus a JSON HTTP endpoint, both backed by ``predict_one``."""

    def _format(price: float) -> str:
        return f"€ {price:,.0f}"

    def _result_html(price: float) -> str:
        return f"""
<div style="text-align:center; padding: 2.25rem 0.5rem;">
  <div style="font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase;
              color: var(--body-text-color-subdued);">Estimated resale price</div>
  <div style="font-size:2.75rem; font-weight:700; letter-spacing:-0.02em;
  font-variant-numeric: tabular-nums; color: var(--primary-500); margin-top:0.35rem;">
    {_format(price)}
  </div>
</div>
"""

    def _error_html(message: str) -> str:
        return f"""
<div style="text-align:center; padding: 2rem 0.5rem; color: var(--error-text-color, #b91c1c);">
  <div style="font-size:1rem; font-weight:600;">Couldn't compute a price</div>
  <div style="font-size:0.85rem; margin-top:0.35rem;">{message}</div>
</div>
"""

    # --- Human-facing form -------------------------------------------------------------
    numeric_inputs = [
        gr.Number(label=_label(col), value=None, info=FEATURE_HINTS.get(col, _DEFAULT_HINT))
        for col in spec.numeric
    ]
    categorical_inputs = [
        gr.Dropdown(
            label=_label(col),
            choices=spec.categories.get(col, []),
            value=None,
            allow_custom_value=True,  # tolerate values the encoder never saw
            info=FEATURE_HINTS.get(col),
        )
        for col in spec.categorical
    ]
    form_inputs = numeric_inputs + categorical_inputs

    def predict_from_form(*values: object) -> str:
        payload = dict(zip(spec.columns, values, strict=True))
        try:
            price = predict_one(pipeline, spec, payload)
        except Exception as exc:  # surface a friendly message instead of a raw traceback
            logger.exception("Prediction failed")
            return _error_html(str(exc))
        return _result_html(price)

    # --- Programmatic JSON endpoint (used by the smoke test) ---------------------------
    def predict_json(payload_json: str) -> dict:
        """Accept a JSON object of feature -> value and return the predicted price.

        A single string arg keeps the API stable regardless of how many features the
        current model has, so ``inference/client.py`` never has to track input arity.
        """
        payload = json.loads(payload_json) if payload_json else {}
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object of feature -> value.")
        price = predict_one(pipeline, spec, payload)
        return {"price": round(price, 2), "currency": "EUR"}

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )

    # Small CSS polish: tighter heading weight/tracking and consistent number rendering
    # for the price, since the default theme font doesn't set these.
    custom_css = """
    .gradio-container h1 { font-weight: 700; letter-spacing: -0.02em; }
    .gradio-container h1, .gradio-container p { font-family: inherit; }
    label span { font-weight: 500 !important; }
    #disclaimer { opacity: 0.75; font-size: 0.8rem; }
    /* Keep every field in a row starting at the same top edge, and reserve a fixed
       line-height for the info/hint text so a longer hint on one field can't push
       it (or its neighbors) out of alignment with the rest of the row. */
    .gradio-container .form { align-items: flex-start !important; }
    .gradio-container span.info { min-height: 1.1em; display: block; line-height: 1.1em; }
    """

    with gr.Blocks(title="Car Price Predictor", theme=theme, css=custom_css) as demo:
        gr.Markdown(
            """
            # 🚗 Car Price Predictor
            Estimate the resale price of a used car from its listing attributes.
            Leave a field blank to let the model impute it.
            """
        )

        with gr.Row(equal_height=False):
            # --- Left: the form, split into digestible sections ------------------------
            with gr.Column(scale=3):
                if numeric_inputs:
                    with gr.Accordion("📏 Numbers & measurements", open=True):
                        for row_fields in _chunk(numeric_inputs, 3):
                            with gr.Row():
                                for field in row_fields:
                                    field.render()

                if categorical_inputs:
                    with gr.Accordion("🏷️ Categories", open=True):
                        for row_fields in _chunk(categorical_inputs, 3):
                            with gr.Row():
                                for field in row_fields:
                                    field.render()

                with gr.Row():
                    clear_button = gr.ClearButton(value="Clear", components=form_inputs)
                    predict_button = gr.Button("Predict price", variant="primary", scale=2)

            # --- Right: sticky-feeling result panel ------------------------------------
            with gr.Column(scale=2):
                with gr.Group():
                    output = gr.HTML(value=PLACEHOLDER_RESULT)
                gr.Markdown(
                    "*This is an estimate based on historical listings, "
                    "not a guaranteed sale price.*",
                    elem_id="disclaimer",
                )

        predict_button.click(predict_from_form, inputs=form_inputs, outputs=output)
        clear_button.add(output)
        clear_button.click(lambda: PLACEHOLDER_RESULT, inputs=None, outputs=output)

        # API-only endpoint; reachable at api_name="/predict".
        gr.api(predict_json, api_name="predict")

    return demo


def create_demo() -> gr.Blocks:
    """Load the configured model from the Hub and build the demo. Called at startup."""
    repo_id = os.environ.get("HF_MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID)
    revision = os.environ.get("MODEL_REVISION", "main")
    token = os.environ.get("HF_TOKEN")

    logger.info("Loading model from %s@%s", repo_id, revision)
    pipeline = load_pipeline_from_hub(repo_id, revision=revision, token=token)
    spec = introspect_features(pipeline)
    logger.info(
        "Serving %d numeric + %d categorical features",
        len(spec.numeric),
        len(spec.categorical),
    )
    return build_demo(pipeline, spec)


# Built at import time: the HF Space runner both executes this file and auto-detects a
# top-level `demo`, so eager construction (and the Hub download it triggers) is intended.
demo = create_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
