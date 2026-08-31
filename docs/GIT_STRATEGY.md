# Estrategia de Git

Este documento fija el modelo de ramas, la convención de commits y la política de
pull requests del proyecto. Es normativo: si una práctica no está descrita aquí, no
se usa.

---

## 1. Modelo de ramas

Usamos **GitHub Flow con una rama de integración persistente**. Es GitHub Flow
porque todo el trabajo ocurre en ramas cortas que vuelven al tronco vía pull
request, y lleva una rama persistente extra porque `main` está protegida y
necesitamos un punto donde varias features convivan e integren antes de ser
candidatas a release.

Tres tipos de rama, ni una más:

| Rama | Vida | Rol |
| --- | --- | --- |
| `main` | Permanente | Rama de release. Protegida. Solo recibe merges vía PR desde `develop`. Cada commit en `main` es un estado entregable y etiquetado. |
| `develop` | Permanente | Rama de integración. Recibe las features terminadas. Es la base desde la que se abre cualquier rama nueva. |
| `feature/NN-nombre-corto` | Efímera | Una unidad de trabajo. Nace de `develop` y muere al mergearse en `develop`. |

> **Nota sobre el nombre.** La rama de integración se llama **`develop`**, siguiendo la
> convención de git-flow, que es el nombre estándar en la industria. El enunciado del
> entregable la menciona como rama **«Development»**: son la misma rama. Se optó por
> conservar el nombre convencional y documentar aquí la equivalencia, en vez de renombrar
> la rama para que coincidiera literalmente con el enunciado. Esta nota existe para que un
> lector externo — un evaluador, o cualquiera que llegue al repo sin contexto — no tenga
> que inferir la correspondencia por su cuenta.

### Reglas duras

- Nunca se commitea directo a `main`. Está protegida a nivel de GitHub.
- Nunca se commitea directo a `develop`. Todo entra por PR.
- Una rama `feature/` nace siempre de `develop` actualizado, no de `main` ni de
  otra feature.
- Una rama `feature/` se borra tras el merge. El historial ya guarda la traza.
- Si `develop` avanza mientras trabajas, actualiza tu rama con `merge`, no con
  `rebase`: reescribir commits ya empujados destruye la evidencia del proceso.

### Convención de nombres de rama

    feature/NN-nombre-corto

- `NN`: número de fase con dos dígitos (`00`, `01`, ...), en el orden del roadmap.
- `nombre-corto`: dos o tres palabras en kebab-case, minúsculas, sin acentos.

Los nombres canónicos son los de las fases del roadmap. No son ejemplos ilustrativos:
son las ramas que el proyecto va a usar, y están fijadas en `docs/ROADMAP.md`.

    feature/00-fundacion
    feature/01-data-and-eda
    feature/02-modeling
    feature/03-genai
    feature/04-production
    feature/05-closing

> **Excepción histórica.** `feature/00-fundacion` conserva su nombre en español.
> Renombrarla exigiría reescribir el historial, y no lo vale. **De la fase 1 en adelante
> todos los nombres de rama van en inglés**, según la convención de idioma de la
> sección 6.

El prefijo numérico hace que `git branch` y el listado de PRs salgan en el orden
cronológico del proyecto, que es exactamente el orden en que se evalúa.

---

## 2. Diagrama del flujo

                                                  tag v0.1.0     tag v0.2.0
                                                       |              |
    main         o------------------------------------ o ------------ o
                  \                                   /              /
                   \                    PR: develop -> main         /
                    \                             /                /
    develop          o------o------o------o------o---------------o
                      \    / \    / \    / \    /                /
                       \  /   \  /   \  /   \  /                /
    feature/            oo     oo     oo     oo             o--o
                  00-fundacion  01-eda  02-baseline      03-modelo
                        |          |         |               |
                     PR + CI    PR + CI   PR + CI         PR + CI

Lectura del diagrama:

1. Cada `feature/NN-*` sale de `develop` y vuelve a `develop` por PR.
2. `develop` acumula features integradas y verdes en CI.
3. Cuando `develop` alcanza un estado entregable, se abre una PR
   `develop -> main`.
4. El merge en `main` se etiqueta con una versión semántica.

---

## 3. Conventional Commits

Los mensajes de commit se escriben **en inglés** (ver la sección 6). Formato:

    <tipo>(<ámbito opcional>): <descripción en imperativo, minúscula, sin punto final>

    <cuerpo opcional: el porqué del cambio, no el qué>

    <footer opcional: refs, BREAKING CHANGE>

La descripción va en imperativo y no supera los ~72 caracteres. El cuerpo explica
**por qué** se hizo el cambio; el **qué** ya lo dice el diff.

### Prefijos que usamos

| Prefijo | Cuándo se usa | Ejemplo |
| --- | --- | --- |
| `feat` | Nueva funcionalidad del producto | `feat(model): add LightGBM classifier with early stopping` |
| `fix` | Corrección de un comportamiento defectuoso | `fix(config): resolve project root from __file__ instead of cwd` |
| `docs` | Solo documentación | `docs(git): document the no-squash merge policy` |
| `style` | Formato sin cambio de comportamiento | `style: apply ruff format to src/` |
| `refactor` | Reestructuración sin cambio de comportamiento ni de features | `refactor(data): extract parquet loading into its own function` |
| `perf` | Mejora de rendimiento | `perf(features): vectorize debt ratio computation` |
| `test` | Agregar o corregir tests | `test(config): cover the data paths declared in settings` |
| `build` | Dependencias, empaquetado, `pyproject.toml` | `build(deps): add shap and lightgbm to main dependencies` |
| `ci` | Workflows y automatización de CI | `ci: run mypy over src/ before the tests` |
| `chore` | Mantenimiento que no encaja arriba | `chore: add mlruns/ to gitignore` |
| `revert` | Revertir un commit previo | `revert: feat(model): add LightGBM classifier` |

### Ámbitos habituales

`config`, `data`, `features`, `model`, `eval`, `agent`, `rag`, `api`, `docs`, `ci`.

### Cambios incompatibles

Se marcan con `!` tras el tipo y con un footer `BREAKING CHANGE:` que describe la
migración:

    feat(config)!: rename RANDOM_STATE to SEED

    BREAKING CHANGE: modules importing RANDOM_STATE must import SEED instead.

---

## 4. Política de pull requests

### Contenido de la descripción

Toda PR incluye, en este orden:

1. **Qué hace** — resumen en dos o tres frases.
2. **Por qué** — el problema o la fase del roadmap que resuelve.
3. **Cómo verificarlo** — comandos concretos que un revisor puede ejecutar y que
   deben pasar (`uv run pytest -v`, `task check`, un script, un endpoint).
4. **Decisiones tomadas** — alternativas descartadas y el motivo. Si la decisión es
   estructural, se enlaza el ADR correspondiente en `docs/adr/`.
5. **Qué queda fuera** — alcance deliberadamente excluido, para que el revisor no lo
   lea como un olvido.

### Requisitos para mergear

- La CI en verde. Es un check requerido en `main`.
- Sin conflictos con la rama destino.
- La descripción completa según el punto anterior.

### Merge commit, no squash

**Mergeamos con merge commit. No usamos squash ni rebase-merge.**

El motivo no es estético. La evaluación de este proyecto mira el historial de git
como evidencia del proceso: en qué orden se atacaron los problemas, qué se probó
antes de que funcionara, cuándo apareció un error y cuándo se corrigió. Un squash
colapsa esa secuencia en un único commit y borra precisamente lo que se está
evaluando. El merge commit conserva los commits individuales de la feature y además
deja registrado el punto de integración, que es información adicional, no ruido.

Corolario práctico: los commits dentro de una feature deben ser legibles de forma
individual, porque van a sobrevivir al merge y van a leerse.

---

## 5. Releases

Los releases se cortan sobre `main` con **versionado semántico**
(`MAJOR.MINOR.PATCH`):

- `MAJOR`: cambio incompatible en la interfaz pública del paquete.
- `MINOR`: funcionalidad nueva retrocompatible (típicamente, una fase del roadmap
  completada).
- `PATCH`: correcciones retrocompatibles.

Procedimiento:

    # Con la PR develop -> main ya mergeada y la CI en verde
    git checkout main
    git pull origin main

    git tag -a v0.2.0 -m "Fase 02: baseline de clasificacion con metricas registradas"
    git push origin v0.2.0

Los tags son anotados (`-a`), nunca ligeros: un tag anotado guarda autor, fecha y
mensaje, y por tanto es parte del registro del proceso.

La versión del tag y la de `pyproject.toml` se mantienen sincronizadas: el bump de
versión en `pyproject.toml` entra en la PR hacia `main`, antes del tag.

---

## 6. Convención de idioma

Regla única: **el idioma de un documento es el de su lector.** El código lo lee
quien programa, en un ecosistema cuyo vocabulario es inglés; la documentación del
entregable la leen el autor del proyecto y un evaluador hispanohablante.

| Ámbito | Idioma | Alcance |
| --- | --- | --- |
| Identificadores | Inglés | Funciones, clases, variables, constantes, módulos |
| Docstrings | Inglés | Toda función, clase y módulo |
| Comentarios en código | Inglés | Incluidos los de archivos de configuración |
| Nombres de archivos y carpetas | Inglés | `docs/METHODOLOGY.md`, `docs/DATA_DICTIONARY.md`, `src/credit_copilot/` |
| Nombres de ramas | Inglés | `main`, `develop`, `feature/NN-nombre-corto` |
| Mensajes de commit | Inglés | Tipo, ámbito, descripción, cuerpo y footers |
| `CLAUDE.md` | Español | Reglas de proceso para el Ejecutor |
| `docs/METHODOLOGY.md` | Español | Metodología de trabajo |
| `docs/ROADMAP.md` | Español | Plan de fases |
| `README.md` | Español | El entregable |
| `docs/GIT_STRATEGY.md` | Español | Este documento |
| `docs/EVALUATION.md` | Español | Registro de mediciones |
| `docs/ERRORS_AND_LEARNINGS.md` | Español | Registro de errores |
| `docs/adr/*.md` | Español | Todos los ADRs |

**Excepción sobre los datos:** los nombres de columnas de los datasets se conservan
**tal como los entrega la fuente**, sin traducir ni normalizar. Traducir `PAY_0` a
`pago_0` rompe la trazabilidad con la documentación original y con cualquier ejemplo
publicado del dataset. Lo que va en español es la **descripción** de la columna en
el diccionario de datos, no su nombre.
