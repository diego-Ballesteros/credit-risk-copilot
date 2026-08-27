"""MLflow configuration against the remote tracking server. The only place credentials move.

**Why the three environment variables and not `dagshub.init()`.** The helper would work and
it would cost a dependency whose only job is to set three variables this project already
reads from `config.py`. It would also bind the tracking setup to one specific host: the day
the server moves, a project configured this way changes one line in `.env`, while a project
configured through a vendor helper changes code. MLflow's own authentication contract is
those three variables, and talking to it directly is what keeps the setup portable.

**Why a missing credential is a hard failure.** MLflow's default behaviour when it has no
tracking URI is to write runs to a local `mlruns/` directory and say nothing. That failure
mode is the worst kind available here: the script prints metrics, exits 0, and the runs
that are supposed to be the evidence of the project are sitting in an untracked folder on
one machine. `configure_mlflow` refuses to continue instead, and names which variable is
missing.

**Why no function here returns the raw URI.** Every path out of this module hands back a
redacted copy, so a caller cannot print a secret by accident even if a URI ever arrives
carrying `user:password@` in it. The credentials themselves are moved from `config.py` into
`os.environ`, which is where MLflow's request layer reads them, and are never returned,
logged or attached to a run.
"""

import contextlib
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import mlflow
from mlflow.entities import Experiment

from credit_copilot.config import settings

DEFAULT_EXPERIMENT_NAME: Final[str] = "credit-risk-baselines"
"""Experiment that holds the phase-2 floor: the baselines and the leakage check."""

LEAKAGE_CHECK_TAG: Final[str] = "run_type"
"""Tag key that separates a diagnostic run from a real experiment."""

LEAKAGE_CHECK_TAG_VALUE: Final[str] = "leakage-check-shuffled-target"
"""Tag value marking a run whose target was permuted. Never a model result."""

BASELINE_TAG_VALUE: Final[str] = "baseline"
"""Tag value marking a run that measures a floor rather than a candidate model."""

_REQUIRED_SETTINGS: Final[tuple[str, ...]] = (
    "mlflow_tracking_uri",
    "mlflow_tracking_username",
    "mlflow_tracking_password",
)
"""The three `Settings` fields MLflow needs. All three or nothing."""


class MLflowConfigurationError(RuntimeError):
    """MLflow cannot be pointed at the remote server with the configuration available."""


@dataclass(frozen=True)
class ExperimentContext:
    """Everything a caller needs to talk about an experiment, and nothing secret.

    Attributes:
        name: Experiment name as registered on the server.
        experiment_id: Server-assigned identifier.
        tracking_uri: Tracking URI with any embedded credentials removed. Safe to print.
        url: Address of the experiment in MLflow's web interface. Safe to print.
    """

    name: str
    experiment_id: str
    tracking_uri: str
    url: str


def redact_uri(uri: str) -> str:
    """Remove any `user:password@` block from a URI so it can be printed.

    A tracking URI is normally harmless, but the form `https://user:token@host/path` is
    valid and is a common way to configure a private server. Redacting unconditionally
    means no caller has to know which form it was handed.

    Args:
        uri: Any URI.

    Returns:
        The same URI with the userinfo replaced by `***` when present, unchanged otherwise.
    """
    parts = urlsplit(uri)
    if not parts.hostname or "@" not in parts.netloc:
        return uri
    netloc = f"***@{parts.hostname}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _missing_settings() -> Sequence[str]:
    """Names of the required credential settings that are absent or blank.

    Returns:
        The environment-variable names, uppercased, in declaration order.
    """
    return [
        name.upper() for name in _REQUIRED_SETTINGS if not (getattr(settings, name) or "").strip()
    ]


def _enable_unicode_console() -> None:
    """Make the standard streams able to carry the characters MLflow writes to them.

    At the end of every run MLflow writes two lines to stdout that begin with an emoji, and
    it does so unconditionally - there is no setting that turns them off. A Windows console
    defaults to cp1252, which cannot encode them, so the write raises `UnicodeEncodeError`
    from inside `end_run`.

    That crash is worth defending against specifically, because of *when* it happens: the
    metrics have already been sent and the run already exists on the server, so the script
    dies with a non-zero exit code over a measurement that in fact succeeded. It is the
    exact inversion of what an exit code is for, and it would be read as a failed run.

    Reconfiguring to UTF-8 is idempotent and a no-op on a console that already speaks it.
    If the stream refuses to be reconfigured at all, the fallback only makes encoding errors
    non-fatal, because a decorative line MLflow prints must never be able to fail a run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="backslashreplace")


def configure_mlflow() -> str:
    """Point MLflow at the remote tracking server, or refuse to continue.

    Moves the three credentials from `config.py` into the environment variables MLflow's
    request layer reads, and sets the tracking URI on the client. Nothing here is cached:
    calling it twice is harmless and is how each script guarantees its own state rather
    than assuming an earlier import arranged it.

    It also prepares the console for the output MLflow is about to produce - see
    `_enable_unicode_console`. That belongs here rather than in each script because the
    problem it avoids is created by pointing MLflow at a server, and a script that forgot
    the call would fail *after* recording its run.

    Returns:
        The tracking URI with any embedded credentials redacted. This is the only URI the
        module hands back, so anything a caller prints is safe by construction.

    Raises:
        MLflowConfigurationError: If any of the three settings is missing or blank. The
            message names the variables, because the alternative - MLflow silently writing
            to a local `mlruns/` directory - produces runs that look real and are not.
    """
    missing = _missing_settings()
    if missing:
        raise MLflowConfigurationError(
            "MLflow cannot reach the tracking server: "
            f"{', '.join(missing)} missing or blank in the .env file. "
            "Refusing to continue, because MLflow would otherwise fall back to a local "
            "mlruns/ directory without saying so and the run would not be recorded "
            "anywhere the project can find it. Copy .env.example to .env and fill it in."
        )

    _enable_unicode_console()

    uri = (settings.mlflow_tracking_uri or "").strip()
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = (settings.mlflow_tracking_username or "").strip()
    os.environ["MLFLOW_TRACKING_PASSWORD"] = (settings.mlflow_tracking_password or "").strip()
    mlflow.set_tracking_uri(uri)
    return redact_uri(uri)


def experiment_web_url(tracking_uri: str, experiment_id: str) -> str:
    """Derive the browser address of an experiment from the tracking URI.

    The tracking URI *is* the root of the MLflow web interface, and that interface is a
    single-page app that routes on the fragment: an experiment lives at
    `<tracking URI>/#/experiments/<id>`. This holds for a self-hosted MLflow on
    `http://host:5000` and for a DagsHub repository on `<repo>.mlflow` alike, because it is
    MLflow's own convention rather than any one vendor's.

    Verified against the live server rather than assumed: fetching the tracking URI without
    credentials returns the MLflow app, so the link resolves for a reader who is not signed
    in. An earlier version of this function stripped the `.mlflow` suffix and pointed at
    DagsHub's own experiments tab instead; that page exists, but it does not read MLflow's
    fragment, so the experiment id in the link was silently ignored.

    Args:
        tracking_uri: Tracking URI, redacted or not.
        experiment_id: Server-assigned experiment identifier.

    Returns:
        A browser URL for the experiment, with any embedded credentials redacted.
    """
    return f"{redact_uri(tracking_uri).rstrip('/')}/#/experiments/{experiment_id}"


def ensure_experiment(name: str = DEFAULT_EXPERIMENT_NAME) -> ExperimentContext:
    """Configure MLflow, make sure the experiment exists, and return its context.

    Creating the experiment when it is absent rather than failing is deliberate: the first
    run of a fresh clone should work, and an experiment is a container with no content to
    get wrong. An experiment the server has soft-deleted is a different situation and is
    reported rather than resurrected, because restoring it silently would merge new runs
    into a history somebody chose to remove.

    Args:
        name: Experiment name.

    Returns:
        The experiment's name, identifier, redacted tracking URI and web address.

    Raises:
        MLflowConfigurationError: If the credentials are missing, if the server cannot be
            reached, or if the named experiment exists in a deleted state.
    """
    tracking_uri = configure_mlflow()

    try:
        experiment: Experiment | None = mlflow.get_experiment_by_name(name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(name)
        elif experiment.lifecycle_stage == "deleted":
            raise MLflowConfigurationError(
                f"The experiment {name!r} exists on the server in a deleted state. "
                "Restore it from the MLflow interface or choose another name; writing "
                "into it from here would silently revive a history somebody removed."
            )
        else:
            experiment_id = experiment.experiment_id
    except MLflowConfigurationError:
        raise
    except Exception as error:  # noqa: BLE001 - the transport raises many unrelated types
        raise MLflowConfigurationError(
            f"Could not reach the MLflow tracking server at {tracking_uri}: {error}"
        ) from error

    mlflow.set_experiment(experiment_id=experiment_id)
    return ExperimentContext(
        name=name,
        experiment_id=str(experiment_id),
        tracking_uri=tracking_uri,
        url=experiment_web_url(tracking_uri, str(experiment_id)),
    )
