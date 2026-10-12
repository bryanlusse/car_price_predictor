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

import html
import json
import logging
import os
from itertools import islice

import gradio as gr
import spaces

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

# --- Form layout ---------------------------------------------------------------------
# The features that move the price most go first, in this order (3 per row). Columns
# the loaded model doesn't have are skipped, so this never breaks on a model change.
PRIMARY_COLUMNS = [
    "make",
    "body_type",
    "vehicle_type",
    "mileage_km_raw",
    "vehicle_age_years",
]
# Technical specs that most listings only partly fill in: shown dimmed, collapsed by
# default, and clearly marked optional. Blank values are imputed by the model anyway.
OPTIONAL_COLUMNS = [
    "power_hp",
    "power_kw",
    "gears",
    "cylinders_volume_cc",
    "co2_emission_grper_km",
    "fuel_cons_comb_l100_km",
]
# Anything that is in neither list (new features, seats, doors, ...) lands in the
# "More details" section, so the form always covers every model column.

# Simple side-on car silhouette. Uses currentColor so the wrapper decides the tint.
_CAR_SVG = """
<svg viewBox="0 0 240 92" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Car illustration" style="width:100%; height:auto; display:block;">
  <path d="M8 64 V54 Q8 47 16 45 L48 39 L72 17 Q76 13 84 13 H150 Q158 13 164 19
           L186 40 L220 46 Q232 48 232 58 V64 Q232 68 227 68 H13 Q8 68 8 64 Z"
        fill="currentColor"/>
  <path d="M80 21 H108 V39 H58 Z M116 21 H148 L170 39 H116 Z"
        fill="var(--background-fill-primary, #fff)" opacity="0.85"/>
  <g fill="var(--body-text-color, #1f2937)">
    <circle cx="62" cy="68" r="15"/>
    <circle cx="180" cy="68" r="15"/>
  </g>
  <g fill="var(--background-fill-primary, #fff)">
    <circle cx="62" cy="68" r="6"/>
    <circle cx="180" cy="68" r="6"/>
  </g>
</svg>
"""

PLACEHOLDER_RESULT = f"""
<div style="text-align:center; padding: 2.5rem 1rem 2.25rem;">
  <div style="width:70%; margin:0 auto; color: var(--body-text-color-subdued);
              opacity:0.35;">{_CAR_SVG}</div>
  <div style="font-size:0.95rem; color: var(--body-text-color-subdued); margin-top:1.25rem;">
    Estimated resale price
  </div>
  <div style="font-size:2.75rem; font-weight:700; letter-spacing:-0.02em;
              color: var(--body-text-color-subdued); margin-top:0.15rem;">
    —
  </div>
  <div style="font-size:0.9rem; color: var(--body-text-color-subdued); margin-top:0.5rem;">
    Fill in the key details and press <b>Predict price</b>.
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


def _render_rows(fields: list, per_row: int = 3) -> None:
    """Render pre-built components into rows of ``per_row`` (call inside a Blocks context)."""
    for row_fields in _chunk(fields, per_row):
        with gr.Row():
            for field in row_fields:
                field.render()


@spaces.GPU(duration=1)
def _zerogpu_warmup() -> None:
    """No-op to satisfy ZeroGPU's startup requirement.

    This app runs a CPU-only scikit-learn pipeline and never needs GPU access,
    but ZeroGPU Spaces refuse to start unless at least one @spaces.GPU function
    is defined. This function is never called.
    """
    return None


def build_demo(pipeline, spec: FeatureSpec) -> gr.Blocks:
    """Assemble the Gradio UI plus a JSON HTTP endpoint, both backed by ``predict_one``."""

    def _format(price: float) -> str:
        return f"€ {price:,.0f}"

    def _result_html(price: float, filled: int, total: int) -> str:
        return f"""
<div class="result-card" style="text-align:center; padding: 2.5rem 1rem 2.25rem;">
  <div class="result-car" style="width:70%; margin:0 auto; color: var(--primary-500);">
    {_CAR_SVG}
  </div>
  <div style="font-size:0.95rem; color: var(--body-text-color-subdued); margin-top:1.25rem;">
    Estimated resale price
  </div>
  <div style="font-size:3.5rem; font-weight:700; letter-spacing:-0.03em; line-height:1.1;
              font-variant-numeric: tabular-nums; color: var(--primary-500);
              margin-top:0.15rem;">
    {_format(price)}
  </div>
  <div style="font-size:0.9rem; color: var(--body-text-color-subdued); margin-top:0.75rem;">
    Based on {filled} of {total} details. The more you fill in, the sharper the estimate.
  </div>
</div>
"""

    def _error_html(message: str) -> str:
        return f"""
<div style="text-align:center; padding: 2rem 0.5rem; color: var(--error-text-color, #b91c1c);">
  <div style="font-size:1rem; font-weight:600;">Couldn't compute a price</div>
  <div style="font-size:0.85rem; margin-top:0.35rem;">{html.escape(message)}</div>
</div>
"""

    # --- Human-facing form -------------------------------------------------------------
    # Build every component once, keyed by column, so the layout can group and reorder
    # them freely while the click handler still receives values in ``spec.columns`` order.
    components: dict[str, gr.Number | gr.Dropdown] = {}
    for col in spec.numeric:
        is_optional = col in OPTIONAL_COLUMNS
        components[col] = gr.Number(
            label=f"{_label(col)} (optional)" if is_optional else _label(col),
            value=None,
            info=FEATURE_HINTS.get(col, _DEFAULT_HINT),
            elem_classes=["optional-field"] if is_optional else None,
        )
    for col in spec.categorical:
        is_optional = col in OPTIONAL_COLUMNS
        components[col] = gr.Dropdown(
            label=f"{_label(col)} (optional)" if is_optional else _label(col),
            choices=spec.categories.get(col, []),
            value=None,
            allow_custom_value=True,  # tolerate values the encoder never saw
            info=FEATURE_HINTS.get(col),
            elem_classes=["optional-field"] if is_optional else None,
        )

    form_inputs = [components[col] for col in spec.columns]
    primary_fields = [components[c] for c in PRIMARY_COLUMNS if c in components]
    optional_fields = [components[c] for c in OPTIONAL_COLUMNS if c in components]
    shown = set(PRIMARY_COLUMNS) | set(OPTIONAL_COLUMNS)
    other_fields = [components[c] for c in spec.columns if c not in shown]

    def predict_from_form(*values: object) -> str:
        payload = dict(zip(spec.columns, values, strict=True))
        try:
            price = predict_one(pipeline, spec, payload)
        except Exception as exc:  # surface a friendly message instead of a raw traceback
            logger.exception("Prediction failed")
            return _error_html(str(exc))
        filled = sum(1 for v in values if v not in (None, "", 0))
        return _result_html(price, filled, len(values))

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

    # Small CSS polish: tighter heading weight/tracking, dimmed optional fields, a sticky
    # result card, and one short entrance animation when a price arrives.
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

    /* Optional technical specs: greyed out until you hover or click into them. */
    .optional-field { opacity: 0.55; transition: opacity 0.15s ease; }
    .optional-field:hover, .optional-field:focus-within { opacity: 1; }
    .section-note { color: var(--body-text-color-subdued); font-size: 0.9rem; margin: 0; }

    /* Result card: stays in view next to the form on wide screens. */
    #result-col { position: sticky; top: 1rem; align-self: flex-start; }
    #result-card { border: 2px solid var(--primary-500); border-radius: 16px; }

    /* The one animated moment: the car rolls in when a new price arrives. */
    @keyframes car-roll-in {
      from { transform: translateX(-36px); opacity: 0; }
      to   { transform: translateX(0);     opacity: 1; }
    }
    .result-car { animation: car-roll-in 0.45s ease-out; }
    @media (prefers-reduced-motion: reduce) { .result-car { animation: none; } }
    @media (max-width: 768px) { #result-col { position: static; } }
    """

    with gr.Blocks(title="Car Price Predictor", theme=theme, css=custom_css) as demo:
        gr.Markdown(
            """
            # 🚗 Car Price Predictor
            Estimate the resale price of a used car from its listing attributes.
            Start with the key details; everything else is optional and the model
            fills in what you leave blank.
            """
        )

        with gr.Row(equal_height=False):
            # --- Left: the form, most important fields first ---------------------------
            with gr.Column(scale=3):
                if primary_fields:
                    gr.Markdown("### Key details")
                    _render_rows(primary_fields)

                if other_fields:
                    with gr.Accordion("More details", open=True):
                        _render_rows(other_fields)

                if optional_fields:
                    with gr.Accordion("Technical specs (optional)", open=False):
                        gr.Markdown(
                            "You can skip all of these. Fill in what you know to sharpen "
                            "the estimate.",
                            elem_classes="section-note",
                        )
                        _render_rows(optional_fields)

                with gr.Row():
                    clear_button = gr.ClearButton(value="Clear", components=form_inputs)
                    predict_button = gr.Button("Predict price", variant="primary", scale=2)

            # --- Right: sticky result panel --------------------------------------------
            with gr.Column(scale=2, elem_id="result-col"):
                with gr.Group(elem_id="result-card"):
                    output = gr.HTML(value=PLACEHOLDER_RESULT)
                gr.Markdown(
                    "*This is an estimate based on historical listings, "
                    "not a guaranteed sale price.*",
                    elem_id="disclaimer",
                )

        # scroll_to_output brings the result into view on narrow screens, where the
        # result card sits below the form instead of beside it.
        predict_button.click(
            predict_from_form,
            inputs=form_inputs,
            outputs=output,
            scroll_to_output=True,
        )
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
