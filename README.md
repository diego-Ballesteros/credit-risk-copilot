# credit-risk-copilot

Agente de AI que asiste en las decisiones de prestamos a usuarios basado en tools

**Estado: en desarrollo.** Proyecto final del Curso II de la especializacion de
Machine Learning Engineering. Este README es una version minima; el README completo
del entregable se escribe en la fase final.

## Requisitos previos

- Python 3.11 (`>=3.11,<3.12`)
- [UV](https://docs.astral.sh/uv/) como unico gestor de paquetes y entornos

UV descarga el interprete 3.11 automaticamente si no esta instalado, asi que basta
con tener UV en el PATH.

## Setup local

```bash
git clone <url-del-repositorio>
cd credit-risk-copilot

# Crea el entorno, instala dependencias e instala el paquete en modo editable
uv sync --dev

# Hooks de calidad de codigo
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

## Documentacion

| Documento | Contenido |
| --- | --- |
| `CLAUDE.md` | Instrucciones de trabajo para el asistente |
| `docs/METHODOLOGY.md` | Metodologia del proyecto |
| `docs/ROADMAP.md` | Plan de fases |
| `docs/GIT_STRATEGY.md` | Modelo de ramas, commits y PRs |
| `docs/EVALUATION.md` | Registro de mediciones y baselines |
| `docs/ERRORS_AND_LEARNINGS.md` | Registro de errores y aprendizajes |
| `docs/adr/` | Architecture Decision Records |
