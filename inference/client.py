"""Smoke-test / thin client for the deployed serving Space.

Runs as the final CD step (`python -m inference.client --space-repo-id ... --wait-healthy`)
to prove the freshly deployed Space actually answers a prediction request -- catching a
broken deploy before anyone relies on it. Also usable standalone to score a single car.

It talks to the Space's HTTP API only (never imports the model), so it is a genuine
end-to-end check of the deployed artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logger = logging.getLogger("car_price_predictor.inference.client")

# A representative used car. The endpoint imputes anything missing and tolerates unknown
# categories, so this is only meant to elicit a plausible, non-null price.
SAMPLE_CAR: dict[str, object] = {
    "mileage_km_raw": 80_000,
    "power_hp": 110,
    "power_kw": 81,
    "nr_seats": 5,
    "nr_doors": 5,
    "gears": 6,
    "cylinders": 4,
    "cylinders_volume_cc": 1600,
    "co2_emission_grper_km": 120,
    "fuel_cons_comb_l100_km": 5.5,
    "vehicle_age_years": 6,
    "make": "Volkswagen",
    "transmission": "Manual",
    "drive_train": "Front",
    "fuel_category": "Petrol",
    "body_type": "Sedans",
    "vehicle_type": "Used",
    "seller_type": "Dealer",
}

# Space runtime stages (from huggingface_hub) that mean "give up now".
_TERMINAL_ERROR_STAGES = {"RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "PAUSED", "STOPPED"}


def wait_until_healthy(repo_id: str, *, token: str | None, timeout: float, poll: float) -> None:
    """Block until the Space reports the RUNNING stage, or raise on timeout / error stage."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    deadline = time.monotonic() + timeout
    last_stage = None
    while True:
        runtime = api.get_space_runtime(repo_id=repo_id)
        stage = runtime.stage
        if stage != last_stage:
            logger.info("Space %s stage: %s", repo_id, stage)
            last_stage = stage
        if stage == "RUNNING":
            return
        if stage in _TERMINAL_ERROR_STAGES:
            raise RuntimeError(f"Space {repo_id} entered terminal stage {stage!r}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Space {repo_id} not RUNNING after {timeout:.0f}s (last stage {stage!r})"
            )
        time.sleep(poll)


def predict(
    repo_id: str, car: dict[str, object], *, token: str | None, connect_retries: int = 5
) -> dict:
    """Call the Space's /predict endpoint with one car and return its JSON response."""
    from gradio_client import Client

    client = None
    last_err: Exception | None = None
    for attempt in range(1, connect_retries + 1):
        try:
            client = Client(repo_id, token=token, verbose=False)
            break
        except Exception as err:  # noqa: BLE001 -- the app may still be warming up
            last_err = err
            logger.info("Connect attempt %d/%d failed: %s", attempt, connect_retries, err)
            time.sleep(3)
    if client is None:
        raise ConnectionError(f"Could not connect to Space {repo_id}") from last_err

    result = client.predict(json.dumps(car), api_name="/predict")
    # gradio_client may hand back a dict or its JSON string form depending on version.
    if isinstance(result, str):
        result = json.loads(result)
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed serving Space.")
    parser.add_argument(
        "--space-repo-id",
        required=True,
        help="HF Space to hit, e.g. blusse7/car-price-predictor-dev.",
    )
    parser.add_argument(
        "--wait-healthy",
        action="store_true",
        help="Poll the Space runtime until it reports RUNNING before calling it.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Max seconds to wait for the Space to become healthy (with --wait-healthy).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between runtime-stage polls (with --wait-healthy).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    token = os.environ.get("HF_TOKEN")

    try:
        if args.wait_healthy:
            wait_until_healthy(
                args.space_repo_id,
                token=token,
                timeout=args.timeout,
                poll=args.poll_interval,
            )
        result = predict(args.space_repo_id, SAMPLE_CAR, token=token)
    except Exception:
        logger.exception("Smoke test failed")
        return 1

    price = result.get("price") if isinstance(result, dict) else None
    if not isinstance(price, (int, float)) or price <= 0:
        logger.error("Unexpected prediction response: %r", result)
        return 1

    logger.info("Smoke test OK -- predicted price: %s %s", price, result.get("currency", ""))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
