"""Pointing MLflow at the remote server, and refusing to continue when it cannot be.

The failure this module exists to prevent is not an exception: it is MLflow falling back to
a local `mlruns/` directory without saying so, producing runs that look real, are green, and
are nowhere the project can find them. Entry 010 of `docs/ERRORS_AND_LEARNINGS.md` records
what that silent default costs when it happens at the registry instead.
"""

from types import SimpleNamespace

import pytest

from credit_copilot.models import tracking
from credit_copilot.models.tracking import (
    MLflowConfigurationError,
    configure_mlflow,
    ensure_experiment,
    experiment_web_url,
    redact_uri,
)

# ---------------------------------------------------------------------------
# redact_uri
# ---------------------------------------------------------------------------


def test_a_uri_without_credentials_is_returned_unchanged() -> None:
    """Redaction must not damage the ordinary case."""
    assert redact_uri("https://host/path") == "https://host/path"


def test_embedded_credentials_are_replaced() -> None:
    """`https://user:token@host` is a valid and common way to configure a private server."""
    redacted = redact_uri("https://someone:s3cr3t@dagshub.com/repo.mlflow")

    assert "s3cr3t" not in redacted
    assert "someone" not in redacted
    assert "dagshub.com/repo.mlflow" in redacted


def test_a_port_survives_redaction() -> None:
    """Dropping the port would produce a printable URI that no longer reaches the server."""
    assert redact_uri("http://user:pw@localhost:5000/x") == "http://***@localhost:5000/x"


def test_a_string_that_is_not_a_uri_is_left_alone() -> None:
    """Redaction is called on whatever a caller holds; it must not raise on nonsense."""
    assert redact_uri("not-a-uri") == "not-a-uri"


# ---------------------------------------------------------------------------
# configure_mlflow
# ---------------------------------------------------------------------------


def _set_credentials(
    monkeypatch: pytest.MonkeyPatch, uri: str, user: str, password: str
) -> list[str]:
    """Point the settings at the given credentials and capture the URI MLflow is handed.

    Returns:
        A list that receives the URI passed to `mlflow.set_tracking_uri`.
    """
    monkeypatch.setattr(tracking.settings, "mlflow_tracking_uri", uri)
    monkeypatch.setattr(tracking.settings, "mlflow_tracking_username", user)
    monkeypatch.setattr(tracking.settings, "mlflow_tracking_password", password)
    # Restored by monkeypatch even though `configure_mlflow` writes them directly.
    for name in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ):
        monkeypatch.setenv(name, "sentinel")

    seen: list[str] = []
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", seen.append)
    return seen


@pytest.mark.parametrize(
    ("uri", "user", "password", "expected"),
    [
        ("", "u", "p", "MLFLOW_TRACKING_URI"),
        ("https://host", "", "p", "MLFLOW_TRACKING_USERNAME"),
        ("https://host", "u", "", "MLFLOW_TRACKING_PASSWORD"),
        ("https://host", "u", "   ", "MLFLOW_TRACKING_PASSWORD"),
    ],
)
def test_a_missing_credential_is_named_rather_than_defaulted(
    monkeypatch: pytest.MonkeyPatch, uri: str, user: str, password: str, expected: str
) -> None:
    """Each of the three, and a blank-but-present value, must refuse by name.

    The whitespace case matters: an `.env` line written as `MLFLOW_TRACKING_PASSWORD= ` is
    present as far as the environment is concerned and empty as far as the server is.
    """
    _set_credentials(monkeypatch, uri, user, password)

    with pytest.raises(MLflowConfigurationError) as error:
        configure_mlflow()

    message = str(error.value)
    assert expected in message
    assert "mlruns/" in message


def test_all_three_missing_are_reported_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """One variable per attempt would mean three edit-and-rerun cycles for one mistake."""
    _set_credentials(monkeypatch, "", "", "")

    with pytest.raises(MLflowConfigurationError) as error:
        configure_mlflow()

    message = str(error.value)
    assert "MLFLOW_TRACKING_URI" in message
    assert "MLFLOW_TRACKING_USERNAME" in message
    assert "MLFLOW_TRACKING_PASSWORD" in message


def test_configuring_sets_the_environment_and_returns_a_redacted_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The URI handed back is the only one a caller prints, so it must be safe by itself."""
    seen = _set_credentials(monkeypatch, "https://user:s3cr3t@host/repo.mlflow", "u", "p")

    returned = configure_mlflow()

    assert "s3cr3t" not in returned
    assert tracking.os.environ["MLFLOW_TRACKING_USERNAME"] == "u"
    assert tracking.os.environ["MLFLOW_TRACKING_PASSWORD"] == "p"
    # MLflow itself is handed the unredacted URI: redaction is for printing, not for talking.
    assert seen == ["https://user:s3cr3t@host/repo.mlflow"]


def test_surrounding_whitespace_is_stripped_before_use(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trailing space in an `.env` line must not become part of the hostname."""
    seen = _set_credentials(monkeypatch, "  https://host/repo.mlflow  ", " u ", " p ")

    configure_mlflow()

    assert seen == ["https://host/repo.mlflow"]


# ---------------------------------------------------------------------------
# experiment_web_url
# ---------------------------------------------------------------------------


def test_the_experiment_url_routes_on_the_fragment() -> None:
    """MLflow's interface is a single-page app: the id lives after the `#`, not in the path.

    An earlier version pointed at a page that does not read the fragment, so the identifier
    was silently ignored and every link opened the same view.
    """
    url = experiment_web_url("https://dagshub.com/x/y.mlflow", "3")

    assert url == "https://dagshub.com/x/y.mlflow/#/experiments/3"


def test_a_trailing_slash_does_not_double_up() -> None:
    """Both spellings of the tracking URI are common and must produce the same link."""
    assert experiment_web_url("http://host:5000/", "0") == experiment_web_url(
        "http://host:5000", "0"
    )


def test_the_experiment_url_never_carries_credentials() -> None:
    """This string ends up in a report; it cannot leak the token that produced the run."""
    assert "tok" not in experiment_web_url("https://u:tok@host/r.mlflow", "1")


# ---------------------------------------------------------------------------
# ensure_experiment
# ---------------------------------------------------------------------------


def test_a_missing_experiment_is_created_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first run of a fresh clone has to work: an experiment is an empty container."""
    _set_credentials(monkeypatch, "https://host/r.mlflow", "u", "p")
    monkeypatch.setattr(tracking.mlflow, "get_experiment_by_name", lambda _name: None)
    monkeypatch.setattr(tracking.mlflow, "create_experiment", lambda _name: "7")
    monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda **_kwargs: None)

    context = ensure_experiment("credit-risk-baselines")

    assert context.experiment_id == "7"
    assert context.url.endswith("/#/experiments/7")


def test_an_existing_experiment_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Creating a second experiment with the same name would split one history in two."""
    _set_credentials(monkeypatch, "https://host/r.mlflow", "u", "p")
    existing = SimpleNamespace(lifecycle_stage="active", experiment_id="0")
    monkeypatch.setattr(tracking.mlflow, "get_experiment_by_name", lambda _name: existing)
    monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda **_kwargs: None)

    assert ensure_experiment().experiment_id == "0"


def test_a_soft_deleted_experiment_is_reported_and_never_revived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing into it would merge new runs into a history somebody chose to remove."""
    _set_credentials(monkeypatch, "https://host/r.mlflow", "u", "p")
    deleted = SimpleNamespace(lifecycle_stage="deleted", experiment_id="4")
    monkeypatch.setattr(tracking.mlflow, "get_experiment_by_name", lambda _name: deleted)

    with pytest.raises(MLflowConfigurationError) as error:
        ensure_experiment("credit-risk-agent")

    assert "deleted state" in str(error.value)


def test_an_unreachable_server_names_the_uri_it_could_not_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport error says nothing about *which* server; the message has to add it."""
    _set_credentials(monkeypatch, "https://user:s3cr3t@host/r.mlflow", "u", "p")

    def explode(_name: str) -> object:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(tracking.mlflow, "get_experiment_by_name", explode)

    with pytest.raises(MLflowConfigurationError) as error:
        ensure_experiment()

    message = str(error.value)
    assert "host/r.mlflow" in message
    assert "s3cr3t" not in message
