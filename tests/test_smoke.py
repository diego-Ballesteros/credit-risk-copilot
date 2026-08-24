"""Tests de humo: verifican que el andamiaje del proyecto esta operativo."""

from pathlib import Path

import credit_copilot
from credit_copilot import RANDOM_STATE, settings


def test_package_imports() -> None:
    """El paquete se importa desde la instalacion editable y expone su version."""
    assert credit_copilot.__version__ == "0.1.0"
    assert credit_copilot.settings is settings


def test_random_state_is_42() -> None:
    """La semilla global esta fijada en 42 y settings la expone."""
    assert RANDOM_STATE == 42
    assert settings.random_state == 42


def test_declared_data_paths_exist() -> None:
    """Las rutas de datos declaradas en settings existen en disco."""
    declared: list[Path] = [
        settings.project_root,
        settings.data_dir,
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.corpus_dir,
        settings.docs_dir,
    ]
    missing = [path for path in declared if not path.is_dir()]
    assert not missing, f"Rutas declaradas que no existen en disco: {missing}"
