"""Smoke tests: verify that the project scaffolding is operational."""

from pathlib import Path

import credit_copilot
from credit_copilot import RANDOM_STATE, settings


def test_package_imports() -> None:
    """The package imports from the editable install and exposes its version."""
    assert credit_copilot.__version__ == "0.1.0"
    assert credit_copilot.settings is settings


def test_random_state_is_42() -> None:
    """The global seed is pinned to 42 and settings exposes it."""
    assert RANDOM_STATE == 42
    assert settings.random_state == 42


def test_declared_data_paths_exist() -> None:
    """The data paths declared in settings exist on disk."""
    declared: list[Path] = [
        settings.project_root,
        settings.data_dir,
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.corpus_dir,
        settings.docs_dir,
    ]
    missing = [path for path in declared if not path.is_dir()]
    assert not missing, f"Declared paths missing on disk: {missing}"
