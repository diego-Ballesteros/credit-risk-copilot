"""Central project configuration: canonical paths and environment variables.

This module is the single source of truth for paths and credentials. The project
root is derived from the location of this file rather than from the working
directory, so running a script from any folder always resolves the same `.env`
file and the same data directories.
"""

from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict

RANDOM_STATE: Final[int] = 42
"""Single random seed for the whole project.

Every component that introduces randomness (splits, sampling, training) reads it
from here. An equivalent value is never hardcoded elsewhere: exact reproduction of
the reported metrics by a third party depends on there being a single control point.
"""

_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = _MODULE_DIR.parents[1]
"""Repository root: `src/credit_copilot/config.py` -> `src/` -> root."""


class Settings(BaseSettings):
    """Project configuration: derived paths and secrets read from the `.env` file.

    The credential fields are deliberately optional. Importing the package must not
    fail in an environment without secrets (for example CI, which runs the tests
    without an `.env` file); a missing credential is validated at its point of use,
    not at import time.

    Attributes:
        anthropic_api_key: Anthropic API key used by the agentic copilot.
        mlflow_tracking_uri: URI of the MLflow tracking server.
        mlflow_tracking_username: MLflow server username, when required.
        mlflow_tracking_password: MLflow server password, when required.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: str | None = None

    @property
    def random_state(self) -> int:
        """Global random seed for the project.

        Returns:
            The value of `RANDOM_STATE`, exposed as a read-only property so it
            cannot be overridden from the environment.
        """
        return RANDOM_STATE

    @property
    def project_root(self) -> Path:
        """Repository root.

        Returns:
            Absolute path to the project root.
        """
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        """Root data directory.

        Returns:
            Absolute path to `data/`.
        """
        return PROJECT_ROOT / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Directory for raw data, exactly as downloaded from the source.

        Returns:
            Absolute path to `data/raw/`.
        """
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Directory for processed data, ready for training.

        Returns:
            Absolute path to `data/processed/`.
        """
        return self.data_dir / "processed"

    @property
    def corpus_dir(self) -> Path:
        """Directory for the document corpus that feeds the RAG pipeline.

        Returns:
            Absolute path to `data/corpus/`.
        """
        return self.data_dir / "corpus"

    @property
    def docs_dir(self) -> Path:
        """Project documentation directory.

        Returns:
            Absolute path to `docs/`.
        """
        return PROJECT_ROOT / "docs"


settings = Settings()
"""Single configuration instance. Import it from here; do not re-instantiate."""
