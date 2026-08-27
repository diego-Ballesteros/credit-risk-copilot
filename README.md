# credit-risk-copilot

Agente de AI que asiste en las decisiones de préstamos a usuarios basado en tools

**Estado: en desarrollo.** Proyecto final del Curso II de la especialización de
Machine Learning Engineering. Este README es una versión mínima; el README completo
del entregable se escribe en la fase final.

## Requisitos previos

- Python 3.11 (`>=3.11,<3.12`)
- [UV](https://docs.astral.sh/uv/) como único gestor de paquetes y entornos

UV descarga el intérprete 3.11 automáticamente si no está instalado, así que basta
con tener UV en el PATH.

## Setup local

```bash
git clone <url-del-repositorio>
cd credit-risk-copilot

# Crea el entorno, instala dependencias e instala el paquete en modo editable
uv sync --dev

# Hooks de calidad de código
uv run pre-commit install

# Credenciales: copiar la plantilla y rellenar los valores localmente
cp .env.example .env
```

El paquete queda instalado en modo editable desde `src/`, de modo que scripts y
notebooks lo importan con `import credit_copilot` sin manipular `sys.path`.

## Comandos

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formato
uv run mypy src/             # tipos
uv run pytest -v             # tests con cobertura
```

Con [Task](https://taskfile.dev/) instalado, `task check` encadena los tres bloques
en el mismo orden que la CI.

## Documentación

| Documento | Contenido |
| --- | --- |
| `CLAUDE.md` | Instrucciones de trabajo para el asistente |
| `docs/METHODOLOGY.md` | Metodología del proyecto |
| `docs/ROADMAP.md` | Plan de fases |
| `docs/GIT_STRATEGY.md` | Modelo de ramas, commits y PRs |
| `docs/DATA_DICTIONARY.md` | Contrato de datos: columnas, tipos, rangos y discrepancias con la fuente |
| `docs/EVALUATION.md` | Registro de mediciones y baselines |
| `docs/MODEL_CARD.md` | Qué hace el modelo productivo, con qué datos, su umbral operativo, sus limitaciones y para qué **no** debe usarse |
| `docs/ERRORS_AND_LEARNINGS.md` | Registro de errores y aprendizajes |
| `docs/analysis/` | Mediciones de registro sobre los datos y el modelo, cada una reproducible por un script del repositorio |
| `docs/adr/` | Architecture Decision Records |
