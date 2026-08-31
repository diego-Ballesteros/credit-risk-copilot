"""Turn a stream of applicants into real HTTP requests against the model API and measure it.

Run it with::

    uv run uvicorn credit_copilot.api.model_app:app --port 8000     # in another terminal
    uv run python scripts/run_online_simulation.py
    uv run python scripts/run_online_simulation.py --api-url http://127.0.0.1:8000 --n 3000

**Why the requests go over HTTP and not through the pipeline in process.** Latency measured
by calling `predict_proba` is the latency of scikit-learn, which nobody experiences. What an
analyst waits for includes JSON parsing, twenty-three Pydantic validations, the data-contract
check, the correlation-identifier middleware, the response serialisation and the socket. Those
are the parts a change to this project can plausibly break, and they are exactly the parts an
in-process measurement cannot see.

**The honesty gate, and why this script refuses to say "degradation".** A PR-AUC computed on
rows the model was fitted on is not a performance estimate; it is a measurement of how well
the forest memorised. `scripts/register_production_model.py` fits the production artefact on
**all 30,000 rows** and says so in a run tag, which this script reads. When the served stream
is drawn from those rows - which is the only thing this dataset allows today - the comparison
against the cross-validated PR-AUC is reported as **optimism**, not as degradation, and the
sign of the gap is the evidence: an in-sample number comes out *higher*. Calling that
"degradation of -0.3" would be reporting contamination as a result.

To measure real degradation, pass `--holdout-ids`, a JSON list of the source `ID` values the
served model was **not** fitted on. Building that file requires a registry version fitted on a
proper subset, which is a decision recorded outside this script.

**Why the stream carries a declared fraction of invalid requests.** With a stream of valid
applicants the error rate is zero and the breakdown by type is empty, which measures nothing.
A declared fraction of malformed requests - a missing field, a null, an out-of-range value, an
unrecognised category - is what makes "error rate by type" a measurement, and it exercises
under load the refusal path that section 7.5 of `docs/MODEL_CARD.md` depends on.

Exit code 0 when the run completed, 1 when the API could not be reached or MLflow could not
be configured, 2 when the stream itself failed.
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import mlflow
import numpy as np
import pandas as pd

from credit_copilot.config import settings
from credit_copilot.console import enable_unicode_console
from credit_copilot.data import schema
from credit_copilot.data.loader import RawDataUnavailableError, load_raw_dataframe
from credit_copilot.models.metrics import compute_metrics
from credit_copilot.models.registry import PREDICTOR_COLUMNS
from credit_copilot.models.tracking import (
    MLflowConfigurationError,
    configure_mlflow,
    ensure_experiment,
)
from credit_copilot.monitoring.drift import DriftReport, compare_frames

EXPERIMENT_NAME: Final[str] = "credit-risk-online"
"""Experiment the online runs land in, kept apart from the phase-2 modelling experiment.

Separate because these runs measure a *deployment* and those measured a *model*, and mixing
them would put two kinds of number under one heading where a reader would compare them.
"""

DEFAULT_API_URL: Final[str] = "http://127.0.0.1:8000"
DEFAULT_STREAM_SIZE: Final[int] = 3_000
DEFAULT_CONCURRENCY: Final[int] = 8
DEFAULT_INVALID_FRACTION: Final[float] = 0.05

READY_TIMEOUT_SECONDS: Final[float] = 300.0
"""How long to wait for `/ready` before giving up.

Generous on purpose: the service now starts immediately and loads in the background
(ADR-0010, decision 3), so a client that gave up in five seconds would be measuring its own
impatience rather than the service.
"""

REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0
"""Per-request ceiling. A request that exceeds it is recorded as a timeout, not retried:
a retry would hide the latency it was meant to measure."""

PERCENTILES: Final[tuple[int, ...]] = (50, 95, 99)

RULE: Final[str] = "=" * 96


@dataclass(frozen=True)
class Outcome:
    """What one request did.

    Attributes:
        row: Position of the applicant in the stream.
        expected_invalid: Whether the request was deliberately malformed.
        status: HTTP status code, or 0 when the request never completed.
        latency_ms: Wall-clock time from sending to the response being parsed.
        probability: The probability returned, when there was one.
        error_type: The `error.type` of the envelope, a transport failure name, or `None`.
    """

    row: int
    expected_invalid: bool
    status: int
    latency_ms: float
    probability: float | None
    error_type: str | None


def parse_args() -> argparse.Namespace:
    """Read where to send the stream and how big it should be.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the model API.")
    parser.add_argument("--n", type=int, default=DEFAULT_STREAM_SIZE, help="Requests to send.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Requests in flight at once. Reported with every latency figure, because a "
        "percentile without its concurrency is not comparable to anything.",
    )
    parser.add_argument(
        "--invalid-fraction",
        type=float,
        default=DEFAULT_INVALID_FRACTION,
        help="Share of deliberately malformed requests, so the error breakdown measures "
        "something. Set 0 to send only valid applicants.",
    )
    parser.add_argument(
        "--holdout-ids",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON list of source ID values the served model was NOT fitted on. Without it "
        "the stream is in-sample and the PR-AUC comparison is reported as optimism.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Defaults to config.py.")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Skip logging. For a local dry run only."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


def load_population() -> pd.DataFrame:
    """Read the source rows with their identifiers and the target.

    `load_raw_dataframe` rather than `load_dataset` because the stream has to be traceable
    back to a client identifier, and the loader drops `ID` by decision (ADR-0004). The
    identifier is used to select the stream and never sent to the API: it is not a feature.

    Returns:
        Every source row, renamed to canonical columns.

    Raises:
        RawDataUnavailableError: The raw file has not been downloaded.
    """
    return load_raw_dataframe().rename(columns=dict(schema.RAW_TO_CANONICAL))


def build_stream(
    population: pd.DataFrame, size: int, seed: int, holdout_ids: Sequence[int] | None
) -> tuple[pd.DataFrame, bool]:
    """Draw the stream, from the holdout when there is one and from everything when there is not.

    Args:
        population: Every source row.
        size: Requested number of requests.
        seed: Seed for the draw, from `config.py`.
        holdout_ids: Identifiers the served model was not fitted on, when known.

    Returns:
        The stream and whether it is a clean holdout.

    Raises:
        ValueError: A holdout file was supplied and none of its identifiers are in the data.
    """
    if holdout_ids:
        eligible = population[population[schema.ID_COLUMN].isin(list(holdout_ids))]
        if eligible.empty:
            raise ValueError(
                f"None of the {len(holdout_ids)} holdout identifiers appear in the dataset. "
                "Refusing to fall back to the full population: that would silently turn a "
                "clean measurement into an in-sample one."
            )
        return eligible.sample(n=min(size, len(eligible)), random_state=seed), True
    return population.sample(n=min(size, len(population)), random_state=seed), False


def build_invalid_payload(applicant: Mapping[str, int], kind: str) -> dict[str, Any]:
    """Break one applicant in a declared way.

    The four kinds are the four refusals `api/schemas.py` promises, so the error breakdown
    reports what the contract claims rather than one generic failure repeated.

    Args:
        applicant: A valid applicant.
        kind: `missing_field`, `explicit_null`, `out_of_range` or `unknown_category`.

    Returns:
        A payload the API must refuse.
    """
    broken = dict(applicant)
    if kind == "missing_field":
        broken.pop("PAY_AMT3", None)
    elif kind == "explicit_null":
        broken["PAY_AMT3"] = None  # type: ignore[assignment]
    elif kind == "out_of_range":
        broken["AGE"] = 7
    else:
        broken["PAY_STATUS_1"] = 15
    return {"applicant": broken}


INVALID_KINDS: Final[tuple[str, ...]] = (
    "missing_field",
    "explicit_null",
    "out_of_range",
    "unknown_category",
)


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def wait_until_ready(client: httpx.Client, api_url: str) -> Mapping[str, Any]:
    """Block until the service reports it can score, or give up loudly.

    Polling `/ready` rather than `/health` is the whole point of the two endpoints: `/health`
    answers 200 while the artefact is still loading, so a client that used it as a gate would
    start the stream against a service that answers 503 to every request and would report a
    100% error rate as a property of the system.

    Args:
        client: An open HTTP client.
        api_url: Base URL.

    Returns:
        The readiness payload.

    Raises:
        RuntimeError: The service did not become ready in time.
    """
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last = "no response yet"
    while time.monotonic() < deadline:
        try:
            response = client.get(f"{api_url}/ready", timeout=10.0)
        except httpx.HTTPError as error:
            last = f"{type(error).__name__}: {error}"
        else:
            if response.status_code == 200:
                payload: Mapping[str, Any] = response.json()
                return payload
            last = f"HTTP {response.status_code}: {response.text[:200]}"
        time.sleep(1.0)
    raise RuntimeError(
        f"{api_url}/ready did not report ready within {READY_TIMEOUT_SECONDS:.0f}s. "
        f"Last answer: {last}"
    )


def send_one(
    client: httpx.Client, api_url: str, row: int, payload: Mapping[str, Any], expected_invalid: bool
) -> Outcome:
    """Send one request and time it end to end.

    Args:
        client: An open HTTP client, reused so the measurement is not dominated by TCP setup.
        api_url: Base URL.
        row: Position in the stream.
        payload: The request body.
        expected_invalid: Whether this request was deliberately malformed.

    Returns:
        What happened, whether it succeeded or not. Failures are recorded, never raised: a
        stream that stops at the first error measures the first error.
    """
    started = time.perf_counter()
    try:
        response = client.post(f"{api_url}/predict", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    except httpx.HTTPError as error:
        return Outcome(
            row=row,
            expected_invalid=expected_invalid,
            status=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            probability=None,
            error_type=f"transport:{type(error).__name__}",
        )

    if response.status_code == 200:
        body = response.json()
        return Outcome(
            row=row,
            expected_invalid=expected_invalid,
            status=200,
            latency_ms=elapsed_ms,
            probability=float(body["probability_of_default"]),
            error_type=None,
        )
    try:
        error_type = str(response.json()["error"]["type"])
    except Exception:  # noqa: BLE001 - a body that is not our envelope is still a failure
        error_type = f"http:{response.status_code}"
    return Outcome(
        row=row,
        expected_invalid=expected_invalid,
        status=response.status_code,
        latency_ms=elapsed_ms,
        probability=None,
        error_type=error_type,
    )


def run_stream(
    api_url: str, payloads: Sequence[tuple[Mapping[str, Any], bool]], concurrency: int
) -> tuple[list[Outcome], float]:
    """Send every request with a fixed number in flight, and time the whole flow.

    Args:
        api_url: Base URL.
        payloads: The bodies to send, each with whether it is deliberately invalid.
        concurrency: Requests in flight at once.

    Returns:
        Every outcome in stream order, and the wall-clock seconds the flow took. Throughput
        is derived from that wall clock and not from the sum of latencies, which would
        describe a serial flow that never happened.
    """
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    with httpx.Client(limits=limits) as client:
        wait_until_ready(client, api_url)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(
                pool.map(
                    lambda item: send_one(client, api_url, item[0], item[1][0], item[1][1]),
                    enumerate(payloads),
                )
            )
        elapsed = time.perf_counter() - started
    return outcomes, elapsed


# ---------------------------------------------------------------------------
# Reading the flow
# ---------------------------------------------------------------------------


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """Summarise a latency sample.

    Args:
        values: Latencies in milliseconds.

    Returns:
        The requested percentiles, the mean and the maximum. Empty input returns an empty
        mapping rather than zeros, because a zero latency reads as a fast service.
    """
    if not values:
        return {}
    ordered = sorted(values)
    summary = {f"p{p}_ms": float(np.percentile(ordered, p, method="nearest")) for p in PERCENTILES}
    summary["mean_ms"] = float(statistics.fmean(ordered))
    summary["max_ms"] = float(ordered[-1])
    summary["min_ms"] = float(ordered[0])
    return summary


def error_breakdown(outcomes: Sequence[Outcome]) -> dict[str, int]:
    """Count failures by the type the envelope declared.

    Args:
        outcomes: Every outcome of the flow.

    Returns:
        Error type to count, most frequent first.
    """
    counts: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.error_type is not None:
            counts[outcome.error_type] = counts.get(outcome.error_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def measure_drift(population: pd.DataFrame, stream: pd.DataFrame) -> DriftReport:
    """Compare the served stream against the distribution the model was fitted on.

    Args:
        population: The rows the artefact was fitted on.
        stream: The rows that were sent.

    Returns:
        One entry per raw predictor, worst PSI first. Categorical columns are compared over
        the levels `schema.py` declares, not over quantiles of an arbitrary code.
    """
    levels = {
        column: sorted(set(schema.CATEGORICAL_LEVELS[column]) | set(accepted))
        for column, accepted in (
            (name, schema.OBSERVED_CODES_ACCEPTED.get(name, {}))
            for name in schema.CATEGORICAL_LEVELS
        )
        if column in PREDICTOR_COLUMNS
    }
    return compare_frames(
        population[list(PREDICTOR_COLUMNS)],
        stream[list(PREDICTOR_COLUMNS)],
        columns=list(PREDICTOR_COLUMNS),
        categorical_levels=levels,
    )


def fetch_model_info(api_url: str) -> Mapping[str, Any]:
    """Read the served artefact's identity and its cross-validated metrics.

    Read from the API rather than from a constant here, so the offline reference this run is
    compared against is the one the running service reports, not one this script remembers.

    Args:
        api_url: Base URL.

    Returns:
        The `/model-info` payload.
    """
    with httpx.Client() as client:
        response = client.get(f"{api_url}/model-info", timeout=30.0)
        response.raise_for_status()
        payload: Mapping[str, Any] = response.json()
        return payload


def offline_pr_auc(model_info: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Pull the cross-validated PR-AUC and its fold deviation out of `/model-info`.

    Args:
        model_info: The `/model-info` payload.

    Returns:
        The mean and the standard deviation, or `(None, None)` when the service could not
        read them from the registry.
    """
    for row in model_info.get("validation", {}).get("metrics", []):
        if row.get("name") == "pr_auc":
            return float(row["value"]), row.get("std")
    return None, None


def fitted_on_tag(model_info: Mapping[str, Any]) -> str:
    """Ask MLflow what the served version was fitted on.

    This is the evidence behind the honesty gate, and it is read rather than assumed: the
    registry run carries a `fitted_on` tag that `scripts/register_production_model.py` sets.

    Args:
        model_info: The `/model-info` payload, for the model name and version.

    Returns:
        The tag's value, or a note saying it could not be read.
    """
    try:
        # Configure first. Without this the client points at the default local store, the
        # lookup fails with "Model Version not found", and the honesty gate would report
        # "the tag could not be read" for a registry that is perfectly reachable - turning
        # the evidence behind the gate into a shrug.
        configure_mlflow()
        client = mlflow.MlflowClient()
        version = client.get_model_version(
            model_info["model"]["name"], model_info["model"]["version"]
        )
        run = client.get_run(str(version.run_id))
        return str(run.data.tags.get("fitted_on", "the run carries no fitted_on tag"))
    except Exception as error:  # noqa: BLE001 - the registry raises many unrelated types
        return f"could not be read: {type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    *,
    api_url: str,
    model_info: Mapping[str, Any],
    fitted_on: str,
    is_clean_holdout: bool,
    concurrency: int,
    outcomes: Sequence[Outcome],
    elapsed: float,
    online: Mapping[str, float] | None,
    cv_pr_auc: float | None,
    cv_pr_auc_std: float | None,
    drift: DriftReport,
) -> None:
    """Print everything the run measured, with each number next to what it is read against."""
    valid = [o for o in outcomes if not o.expected_invalid]
    invalid = [o for o in outcomes if o.expected_invalid]
    scored = [o for o in valid if o.probability is not None]

    print("\n" + RULE)
    print("ONLINE SIMULATION")
    print(RULE)
    print(f"API                : {api_url}")
    print(f"Model              : {model_info['model']['uri']}")
    print(f"Concurrency        : {concurrency} requests in flight")
    print(
        f"Requests           : {len(outcomes):,}  ({len(valid):,} valid, {len(invalid):,} "
        "deliberately malformed)"
    )

    print("\n" + "-" * 96)
    print(
        "LATENCY  (end to end over HTTP: JSON parse, 23 validations, contract check, "
        "middleware, socket)"
    )
    print("-" * 96)
    for label, sample in (("valid requests", valid), ("refused requests", invalid)):
        summary = percentiles([o.latency_ms for o in sample])
        if not summary:
            print(f"  {label:<20} (none sent)")
            continue
        print(
            f"  {label:<20} p50 {summary['p50_ms']:7.1f} ms   p95 {summary['p95_ms']:7.1f} ms   "
            f"p99 {summary['p99_ms']:7.1f} ms   max {summary['max_ms']:8.1f} ms"
        )
    print(f"\n  Wall clock         : {elapsed:.2f} s for {len(outcomes):,} requests")
    print(
        f"  Throughput         : {len(outcomes) / elapsed:,.1f} req/s sustained "
        f"at concurrency {concurrency}"
    )

    print("\n" + "-" * 96)
    print("ERRORS, BY TYPE")
    print("-" * 96)
    breakdown = error_breakdown(outcomes)
    total_errors = sum(breakdown.values())
    print(
        f"  Error rate         : {total_errors / len(outcomes):.4%} "
        f"({total_errors:,} of {len(outcomes):,})"
    )
    unexpected = sum(1 for o in valid if o.error_type is not None)
    print(f"  Unexpected failures: {unexpected:,}  (valid applicants that did not score)")
    for name, count in breakdown.items():
        print(f"    {name:<28} {count:>7,}")
    if not breakdown:
        print("    none")

    print("\n" + "-" * 96)
    print("DISCRIMINATION ON THE SERVED STREAM")
    print("-" * 96)
    if online is None:
        print("  Not computed: no valid request produced a probability.")
    else:
        print(f"  Rows scored        : {len(scored):,}")
        for name in ("pr_auc", "roc_auc", "brier", "precision_at_top_10pct"):
            print(f"  {name:<19}: {online[name]:.6f}")
        if cv_pr_auc is not None:
            gap = online["pr_auc"] - cv_pr_auc
            deviation = f" ± {cv_pr_auc_std:.6f}" if cv_pr_auc_std else ""
            print(f"\n  Cross-validated PR-AUC (the offline reference): {cv_pr_auc:.6f}{deviation}")
            print(f"  Stream PR-AUC minus cross-validated PR-AUC   : {gap:+.6f}")
            if is_clean_holdout:
                print("  Read as DEGRADATION: the stream is a declared holdout.")
            else:
                print("  Read as OPTIMISM, NOT degradation. The served artefact was fitted on")
                print(f"  these rows - registry tag `fitted_on` says: {fitted_on!r}.")
                print("  A positive gap here measures memorisation, not performance.")

    print("\n" + "-" * 96)
    print("FEATURE DRIFT: training distribution against served stream")
    print("-" * 96)
    print(drift.report())
    if not is_clean_holdout:
        print(
            "\n  NOTE: the stream is a random sample OF the training population, so a PSI near\n"
            "  zero is expected by construction. This run is a negative control on the\n"
            "  instrument - it shows the detector does not fire on identical distributions -\n"
            "  and it is NOT evidence that the production traffic will not drift."
        )
    print(RULE)


def log_to_mlflow(
    *,
    api_url: str,
    model_info: Mapping[str, Any],
    fitted_on: str,
    is_clean_holdout: bool,
    concurrency: int,
    invalid_fraction: float,
    seed: int,
    outcomes: Sequence[Outcome],
    elapsed: float,
    online: Mapping[str, float] | None,
    cv_pr_auc: float | None,
    drift: DriftReport,
) -> str:
    """Record the run, so an online measurement is not a number that scrolled past.

    Args:
        api_url: Where the stream was sent.
        model_info: The served artefact's identity and offline metrics.
        fitted_on: The registry's own statement of what the model was fitted on.
        is_clean_holdout: Whether the stream is a declared holdout.
        concurrency: Requests in flight.
        invalid_fraction: Declared share of malformed requests.
        seed: The draw's seed.
        outcomes: Every outcome.
        elapsed: Wall clock of the flow.
        online: Metrics on the served stream, when any row scored.
        cv_pr_auc: The offline reference.
        drift: The drift comparison.

    Returns:
        The run identifier.

    Raises:
        MLflowConfigurationError: The tracking server is not configured.
    """
    context = ensure_experiment(EXPERIMENT_NAME)
    valid = [o for o in outcomes if not o.expected_invalid]
    invalid = [o for o in outcomes if o.expected_invalid]
    breakdown = error_breakdown(outcomes)
    total_errors = sum(breakdown.values())

    with mlflow.start_run(experiment_id=context.experiment_id, run_name="online-simulation") as run:
        mlflow.set_tags(
            {
                "run_type": "online-simulation",
                "phase": "04-production",
                "transport": "http",
                "api_url": api_url,
                "model_uri": model_info["model"]["uri"],
                "fitted_on": fitted_on,
                # The single most important tag on this run. A future reader who sees a
                # PR-AUC of 0.9 here must find out why in the same place they find the 0.9.
                "stream_is_clean_holdout": str(is_clean_holdout).lower(),
                "pr_auc_gap_reading": "degradation" if is_clean_holdout else "optimism",
            }
        )
        mlflow.log_params(
            {
                "n_requests": len(outcomes),
                "concurrency": concurrency,
                "invalid_fraction": invalid_fraction,
                "random_state": seed,
                "request_timeout_s": REQUEST_TIMEOUT_SECONDS,
            }
        )

        metrics: dict[str, float] = {
            "wall_clock_s": elapsed,
            "throughput_rps": len(outcomes) / elapsed,
            "error_rate": total_errors / len(outcomes),
            "unexpected_failures": float(sum(1 for o in valid if o.error_type is not None)),
            "drift_max_psi": drift.max_psi,
            "drift_features_above_threshold": float(len(drift.drifted)),
        }
        for label, sample in (("valid", valid), ("refused", invalid)):
            for name, value in percentiles([o.latency_ms for o in sample]).items():
                metrics[f"latency_{label}_{name}"] = value
        for name, count in breakdown.items():
            metrics[f"errors_{name.replace(':', '_')}"] = float(count)
        if online is not None:
            metrics.update({f"online_{name}": value for name, value in online.items()})
            if cv_pr_auc is not None:
                metrics["offline_cv_pr_auc"] = cv_pr_auc
                metrics["pr_auc_gap"] = online["pr_auc"] - cv_pr_auc
        mlflow.log_metrics(metrics)

        mlflow.log_text(drift.report(), "drift_report.txt")
        mlflow.log_text(
            json.dumps(
                [
                    {
                        "feature": item.feature,
                        "psi": item.psi,
                        "band": item.band.value,
                        "ks_statistic": item.ks_statistic,
                        "ks_p_value": item.ks_p_value,
                        "is_categorical": item.is_categorical,
                    }
                    for item in drift.features
                ],
                indent=2,
            ),
            "drift_by_feature.json",
        )
        latencies = pd.DataFrame(
            [
                {
                    "row": o.row,
                    "expected_invalid": o.expected_invalid,
                    "status": o.status,
                    "latency_ms": o.latency_ms,
                    "error_type": o.error_type or "",
                }
                for o in outcomes
            ]
        )
        mlflow.log_text(latencies.to_csv(index=False), "request_log.csv")
        return str(run.info.run_id)


def main() -> int:
    """Send the stream, measure it, and record what it measured.

    Returns:
        0 on success, 1 if the API or MLflow could not be reached, 2 if the stream failed.
    """
    enable_unicode_console()
    args = parse_args()
    seed = settings.random_state if args.seed is None else args.seed

    try:
        population = load_population()
    except RawDataUnavailableError as error:
        print(f"The raw dataset is not available:\n{error}", file=sys.stderr)
        return 2

    holdout_ids: list[int] | None = None
    if args.holdout_ids is not None:
        holdout_ids = [int(value) for value in json.loads(args.holdout_ids.read_text("utf-8"))]

    stream, is_clean_holdout = build_stream(population, args.n, seed, holdout_ids)

    try:
        model_info = fetch_model_info(args.api_url)
    except httpx.HTTPError as error:
        print(f"Could not reach {args.api_url}/model-info: {error}", file=sys.stderr)
        return 1
    fitted_on = fitted_on_tag(model_info)

    if not is_clean_holdout:
        print(RULE)
        print("THE SERVED STREAM IS NOT A CLEAN HOLDOUT - READ THIS BEFORE THE NUMBERS")
        print(RULE)
        print(f"  The registry says the served version was fitted on: {fitted_on!r}")
        print("  Every row in this stream was therefore seen during fitting. Latency,")
        print("  throughput and error rate are unaffected and are real measurements.")
        print("  The PR-AUC of the stream is NOT a performance estimate, and the gap against")
        print("  the cross-validated figure is reported as OPTIMISM, not as degradation.")
        print("  Pass --holdout-ids to measure degradation against rows the model never saw.")

    rng = random.Random(seed)
    payloads: list[tuple[Mapping[str, Any], bool]] = []
    for _, row in stream.iterrows():
        applicant = {column: int(row[column]) for column in PREDICTOR_COLUMNS}
        if rng.random() < args.invalid_fraction:
            kind = INVALID_KINDS[rng.randrange(len(INVALID_KINDS))]
            payloads.append((build_invalid_payload(applicant, kind), True))
        else:
            payloads.append(({"applicant": applicant}, False))

    try:
        outcomes, elapsed = run_stream(args.api_url, payloads, args.concurrency)
    except RuntimeError as error:
        print(f"The API never became ready:\n{error}", file=sys.stderr)
        return 1

    scored = [(index, o) for index, o in enumerate(outcomes) if o.probability is not None]
    online: Mapping[str, float] | None = None
    if scored:
        targets = stream[schema.TARGET_COLUMN].to_numpy()
        labels = [int(targets[index]) for index, _ in scored]
        probabilities = [o.probability for _, o in scored]
        online = compute_metrics(labels, probabilities)

    cv_pr_auc, cv_pr_auc_std = offline_pr_auc(model_info)
    drift = measure_drift(population, stream)

    print_report(
        api_url=args.api_url,
        model_info=model_info,
        fitted_on=fitted_on,
        is_clean_holdout=is_clean_holdout,
        concurrency=args.concurrency,
        outcomes=outcomes,
        elapsed=elapsed,
        online=online,
        cv_pr_auc=cv_pr_auc,
        cv_pr_auc_std=cv_pr_auc_std,
        drift=drift,
    )

    if args.no_mlflow:
        print(
            "\nNot logged: --no-mlflow was given. A measurement that is not recorded is a "
            "measurement nobody can cite."
        )
        return 0
    try:
        run_id = log_to_mlflow(
            api_url=args.api_url,
            model_info=model_info,
            fitted_on=fitted_on,
            is_clean_holdout=is_clean_holdout,
            concurrency=args.concurrency,
            invalid_fraction=args.invalid_fraction,
            seed=seed,
            outcomes=outcomes,
            elapsed=elapsed,
            online=online,
            cv_pr_auc=cv_pr_auc,
            drift=drift,
        )
    except MLflowConfigurationError as error:
        print(f"\nMLflow is not configured:\n{error}", file=sys.stderr)
        return 1
    print(f"\nLogged to MLflow: experiment {EXPERIMENT_NAME}, run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
