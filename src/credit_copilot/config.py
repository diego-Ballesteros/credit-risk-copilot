"""Configuracion central del proyecto: rutas canonicas y variables de entorno.

Este modulo es la unica fuente de verdad para rutas y credenciales. La raiz del
proyecto se deriva de la ubicacion de este archivo y no del directorio de trabajo,
de modo que ejecutar un script desde cualquier carpeta resuelve siempre el mismo
`.env` y los mismos directorios de datos.
"""

from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict

RANDOM_STATE: Final[int] = 42
"""Semilla unica del proyecto.

Todo componente que introduzca aleatoriedad (splits, muestreos, entrenamiento)
la toma de aqui. Nunca se hardcodea un valor equivalente en otro modulo: la
reproducibilidad exacta de las metricas por parte de un tercero depende de que
exista un solo punto de control.
"""

_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = _MODULE_DIR.parents[1]
"""Raiz del repositorio: `src/credit_copilot/config.py` -> `src/` -> raiz."""


class Settings(BaseSettings):
    """Configuracion del proyecto: rutas derivadas y secretos leidos del `.env`.

    Los campos de credenciales son opcionales de forma deliberada. Importar el
    paquete no debe fallar en un entorno sin secretos (por ejemplo la CI, que
    ejecuta los tests sin `.env`); la ausencia de una credencial se valida en el
    punto de uso, no en el import.

    Attributes:
        anthropic_api_key: Clave de la API de Anthropic para el copiloto agentico.
        mlflow_tracking_uri: URI del servidor de tracking de MLflow.
        mlflow_tracking_username: Usuario del servidor de MLflow, si aplica.
        mlflow_tracking_password: Password del servidor de MLflow, si aplica.
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
        """Semilla global del proyecto.

        Returns:
            El valor de `RANDOM_STATE`, expuesto como propiedad de solo lectura
            para que no pueda sobrescribirse desde el entorno.
        """
        return RANDOM_STATE

    @property
    def project_root(self) -> Path:
        """Raiz del repositorio.

        Returns:
            Ruta absoluta a la raiz del proyecto.
        """
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        """Directorio raiz de datos.

        Returns:
            Ruta absoluta a `data/`.
        """
        return PROJECT_ROOT / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Directorio de datos crudos, tal como se descargan de la fuente.

        Returns:
            Ruta absoluta a `data/raw/`.
        """
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Directorio de datos procesados, listos para entrenamiento.

        Returns:
            Ruta absoluta a `data/processed/`.
        """
        return self.data_dir / "processed"

    @property
    def corpus_dir(self) -> Path:
        """Directorio del corpus documental que alimenta el RAG.

        Returns:
            Ruta absoluta a `data/corpus/`.
        """
        return self.data_dir / "corpus"

    @property
    def docs_dir(self) -> Path:
        """Directorio de documentacion del proyecto.

        Returns:
            Ruta absoluta a `docs/`.
        """
        return PROJECT_ROOT / "docs"


settings = Settings()
"""Instancia unica de configuracion. Se importa desde aqui; no se reinstancia."""
