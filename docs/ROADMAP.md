# Proyecto Curso II — Credit Risk Copilot

> Sistema de evaluación de riesgo crediticio con modelo de ML productivo y copiloto agéntico (LangGraph + RAG + tools).
>
> **Especialización MLE 2026 — Curso II**
> Documento de diseño y roadmap de ejecución.
> *Versión 1.0 — Fecha de creación: 23 de agosto de 2026*

---

## Tabla de contenidos

1. [El problema](#1-el-problema)
2. [Hipótesis y objetivos](#2-hipótesis-y-objetivos)
3. [Qué vamos a construir](#3-qué-vamos-a-construir)
4. [Dataset](#4-dataset)
5. [Arquitectura de la solución](#5-arquitectura-de-la-solución)
6. [Estrategia de evaluación (offline y online)](#6-estrategia-de-evaluación-offline-y-online)
7. [Stack técnico](#7-stack-técnico)
8. [Estructura del repositorio](#8-estructura-del-repositorio)
9. [Estrategia de Git](#9-estrategia-de-git)
10. [Roadmap de 15 días](#10-roadmap-de-15-días)
11. [Trazabilidad contra la rúbrica](#11-trazabilidad-contra-la-rúbrica)
12. [Riesgos y mitigaciones](#12-riesgos-y-mitigaciones)
13. [Principios de trabajo](#13-principios-de-trabajo)

---

## 1. El problema

### 1.1 Contexto de negocio

Una entidad financiera que otorga crédito rotativo enfrenta una decisión repetitiva y costosa: **estimar la probabilidad de que un cliente incumpla su obligación en el próximo período**. De esa estimación cuelgan tres decisiones de negocio:

| Decisión | Impacto |
|---|---|
| Aprobar o rechazar | Ingresos vs. pérdida esperada |
| Tasa y cupo asignado | Rentabilidad ajustada por riesgo |
| Provisión contable | Requerimiento de capital regulatorio |

El error tiene dos caras asimétricas:

- **Falso negativo** (aprobar a quien incumple) → pérdida directa del principal
- **Falso positivo** (rechazar a quien habría pagado) → ingreso no percibido + costo de oportunidad

En la práctica el costo del falso negativo suele ser varias veces mayor. Esto no es un detalle: **define la métrica del proyecto** y explica por qué `accuracy` es inútil aquí.

### 1.2 El problema real que resolvemos

Un modelo de riesgo que solo emite un número no resuelve el trabajo del analista. El analista de crédito necesita tres cosas más:

1. **Entender por qué** — un score de 0.73 sin explicación no es accionable ni defendible ante un comité
2. **Contrastar contra política** — ¿qué dice la normativa y la política interna para este perfil de riesgo?
3. **Explorar escenarios** — "¿y si el solicitante redujera su utilización de cupo al 30%?"

Hoy esas tres cosas viven en sistemas separados: el score en un motor, la política en un PDF de 200 páginas, y el escenario en la cabeza del analista. **Ese es el problema que ataca este proyecto.**

### 1.3 Definición formal del problema de ML

- **Tipo:** Aprendizaje supervisado
- **Subtipo:** Clasificación binaria
- **Unidad de análisis:** un cliente-período
- **Target:** `default` ∈ {0, 1} — ¿el cliente incumple el pago del próximo período?
- **Desbalance esperado:** ~22% de clase positiva
- **Salida requerida:** probabilidad calibrada, no solo la etiqueta

> **Nota metodológica clave.** Necesitamos una **probabilidad calibrada**, no solo una clasificación. Si el modelo dice 0.20, debe cumplirse que aproximadamente 20 de cada 100 clientes con ese score incumplan. Sin calibración no se puede calcular pérdida esperada ni fijar tasa. Esto obliga a medir **Brier score** y curvas de calibración, no solo AUC.

---

## 2. Hipótesis y objetivos

### 2.1 Hipótesis principal

> El **comportamiento de pago reciente** (trayectoria de mora, evolución de la utilización del cupo y ratio de pago sobre saldo) tiene mayor poder predictivo sobre el incumplimiento que los **atributos demográficos estáticos** del cliente.

Es falsable: se contrasta comparando un modelo entrenado solo con demografía contra uno con features de comportamiento, y se verifica con importancia de features y SHAP.

### 2.2 Hipótesis secundaria (componente GenAI)

> Un agente que combine la predicción del modelo, su explicación local (SHAP) y la recuperación de normativa aplicable produce recomendaciones **verificables y trazables**, superiores a un LLM sin acceso a esas herramientas.

Es falsable: se contrasta el agente completo contra un LLM baseline sin tools, midiendo *groundedness* y tasa de alucinación sobre un set de preguntas de evaluación.

### 2.3 Objetivos del proyecto

**Objetivos de producto**
- [ ] Modelo de clasificación con probabilidad calibrada, registrado como modelo productivo en MLflow
- [ ] Agente conversacional que responda consultas de riesgo con evidencia trazable
- [ ] API REST que sirva ambos componentes
- [ ] Sistema desplegable con un solo comando

**Objetivos de aprendizaje** *(igual de importantes)*
- [ ] Entender por qué cada métrica se elige y qué esconde
- [ ] Construir un pipeline de sklearn sin data leakage y saber explicar dónde estaría el leakage
- [ ] Manejar MLflow como registro de experimentos y de modelos, no como un logger de accuracy
- [ ] Construir un grafo de LangGraph entendiendo estado, nodos, aristas condicionales y ciclos
- [ ] Entender qué hace realmente un RAG: chunking, embeddings, recuperación, y por qué falla
- [ ] Evaluar un sistema GenAI con métricas, no con impresiones

---

## 3. Qué vamos a construir

### 3.1 Los dos componentes

```
┌──────────────────────────────────────────────────────────┐
│                    CREDIT RISK COPILOT                    │
├──────────────────────────────────────────────────────────┤
│                                                           │
│   COMPONENTE 1: MOTOR DE RIESGO (ML clásico)             │
│   ├─ Pipeline de preprocesamiento                        │
│   ├─ Modelo de clasificación calibrado                   │
│   ├─ Explicabilidad SHAP                                 │
│   └─ Registro y versionado en MLflow                     │
│                                                           │
│   COMPONENTE 2: COPILOTO (GenAI agéntico)                │
│   ├─ Grafo LangGraph con estado                          │
│   ├─ RAG sobre corpus normativo                          │
│   ├─ Tools que invocan el Componente 1                   │
│   └─ Simulador contrafactual                             │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**El principio de diseño no negociable:** el modelo de ML **es una herramienta del agente**, no un artefacto paralelo. El agente decide cuándo llamarlo, cuándo explicar, cuándo consultar normativa. Esta es la diferencia entre un proyecto integrado y dos proyectos pegados con cinta.

### 3.2 Las cuatro tools del agente

| Tool | Qué hace | De dónde sale |
|---|---|---|
| `score_solicitante` | Devuelve probabilidad de default | Modelo productivo desde MLflow Registry |
| `explicar_decision` | Top-N features que empujan el score, con dirección y magnitud | SHAP sobre el modelo |
| `simular_escenario` | Recalcula el score modificando atributos | Contrafactual sobre el pipeline |
| `consultar_politica` | Recupera fragmentos de normativa aplicables | RAG sobre el corpus |

### 3.3 El simulador contrafactual

Este es el componente que da la sensación de "simulador" sin caer en afirmaciones causales.

**Lo que SÍ afirma:** *"Si este cliente presentara una utilización de cupo del 30% en lugar del 85%, el modelo estimaría una probabilidad de incumplimiento de 0.12 en lugar de 0.31."*

**Lo que NO afirma:** *"Si el cliente reduce su utilización, dejará de incumplir."*

La diferencia es todo. La primera es una afirmación sobre el modelo — verificable y honesta. La segunda es una afirmación causal que los datos observacionales no soportan. **Esta distinción va documentada explícitamente en el Model Card**, y es una de las cosas que más madurez demuestra en el proyecto.

### 3.4 Flujo de una consulta real

```
Analista: "Cliente 8842. ¿Lo apruebo? ¿Y si le bajo el cupo?"
   │
   ▼
[LangGraph: nodo de planificación]
   │
   ├──► score_solicitante(8842)      → 0.31
   ├──► explicar_decision(8842)      → PAY_0=+0.14, utilizacion=+0.09, ...
   ├──► simular_escenario(8842, {limite_credito: -30%})  → 0.24
   └──► consultar_politica("riesgo medio-alto, cupo rotativo")
                                     → 3 fragmentos de normativa con cita
   │
   ▼
[Nodo de síntesis]
   │
   ▼
Respuesta fundamentada, con las cifras del modelo, las razones y la
referencia normativa citada. Nada inventado.
```

---

## 4. Dataset

### 4.1 Fuente seleccionada

**Default of Credit Card Clients** — UCI Machine Learning Repository (ID 350)

| Atributo | Valor |
|---|---|
| Registros | 30.000 |
| Variables | 23 predictoras + 1 target |
| Tamaño | ~3 MB (muy por debajo del límite de 100 MB) |
| Clase positiva | ~22% |
| Valores faltantes | Ninguno |
| Licencia | Uso académico abierto |

### 4.2 Por qué este dataset

- **Fricción mínima** — descarga directa, sin ETL, sin deduplicación, sin joins
- **Tipos mixtos** — categóricas (educación, estado civil) y numéricas → habilita OHE, bucketing y escalado
- **Estructura de panel** — 6 meses de historial de pago, saldos y abonos por cliente. Esto permite construir features de comportamiento *derivadas*, que es donde está el aprendizaje real de feature engineering
- **Diccionario de datos manejable** — 24 columnas se documentan bien; un dataset de 120 columnas convierte ese requisito en trabajo mecánico
- **Desbalance realista** — 22% es suficiente para que accuracy sea engañoso, sin ser tan extremo que complique el modelado

### 4.3 Limitación conocida y cómo la manejamos

El dataset **no tiene fecha de originación**, por lo que no es posible una validación *out-of-time* real. Consecuencias:

- La validación correcta aquí es **Stratified K-Fold**, no split cronológico
- Esto se documenta explícitamente en el Model Card como limitación para producción
- En un despliegue real se exigiría validación OOT; lo decimos, no lo escondemos

> Reconocer una limitación y explicar qué se haría distinto en producción vale más que fingir que no existe.

### 4.4 Features derivadas a construir

Estas no vienen en el dataset — las creamos nosotros, y son el corazón de la hipótesis principal:

| Feature | Definición | Intuición |
|---|---|---|
| `utilizacion_cupo_m` | `BILL_AMT_m / LIMIT_BAL` | Cliente al tope del cupo = señal de estrés |
| `tendencia_utilizacion` | Pendiente de la utilización sobre los **5 meses del bloque homogéneo** (meses 2 a 6) | ¿Se está deteriorando o recuperando? |
| `ratio_pago_m` | `PAY_AMT_m / BILL_AMT_(m+1)` | ¿Paga el mínimo o el total? |
| `racha_mora` | Meses consecutivos con atraso | Persistencia > severidad puntual |
| `mora_maxima` | Peor atraso en la ventana | Severidad histórica |
| `volatilidad_saldo` | Desviación estándar de saldos | Comportamiento errático |
| `meses_sin_pago` | Conteo de `PAY_AMT == 0` | Señal fuerte de incumplimiento |

> **El índice del panel corre hacia atrás: `1` es el mes más reciente (septiembre 2005) y
> `6` el más viejo (abril 2005), así que el mes cronológicamente anterior a `m` es `m+1`.**
> Por eso el denominador de `ratio_pago_m` es `BILL_AMT_(m+1)` y no `BILL_AMT_(m-1)`. Es una
> trampa que se pisa dos veces.

> **`tendencia_utilizacion` se calcula sobre 5 meses y no sobre 6.** La
> [decisión 3 del ADR-0004](adr/0004-codigos-no-documentados-de-pay-status.md) midió que el
> mes 1 no comparte la escala de los meses 2 a 6 en la zona baja de los códigos, así que
> ninguna feature de trayectoria agrega las seis columnas como un panel homogéneo. El mes 1
> entra al modelo como variable aparte.

---

## 5. Arquitectura de la solución

### 5.1 Diagrama de flujo

```
   FUENTE                PREPROCESAMIENTO              MODELADO
┌──────────┐         ┌──────────────────┐        ┌──────────────────┐
│ UCI 350  │────────►│ Validación       │───────►│ Baseline         │
│ (CSV)    │         │ Limpieza         │        │ Modelos avanzados│
└──────────┘         │ Feature eng.     │        │ Tuning (Optuna)  │
                     │ → parquet        │        │ Calibración      │
                     └──────────────────┘        └────────┬─────────┘
                                                          │
                     ┌────────────────────────────────────┘
                     ▼
              ┌──────────────┐
              │   MLflow     │  experimentos · métricas · artefactos
              │   Registry   │  modelo productivo versionado
              └──────┬───────┘
                     │
      ┌──────────────┴───────────────┐
      ▼                              ▼
┌──────────────┐            ┌─────────────────────┐
│  API REST    │            │   AGENTE LangGraph  │
│  (FastAPI)   │◄───tools───│                     │
│              │            │  ┌───────────────┐  │
│ /predict     │            │  │ RAG retriever │  │
│ /explain     │            │  └───────┬───────┘  │
│ /simulate    │            └──────────┼──────────┘
│ /health      │                       │
└──────┬───────┘                       ▼
       │                     ┌──────────────────┐
       │                     │  Vector Store    │
       ▼                     │  (Chroma)        │
┌──────────────┐             │  ← corpus        │
│ Monitoreo    │             │    normativo     │
│ drift · logs │             └──────────────────┘
│ latencia     │
└──────────────┘
```

### 5.2 El grafo del agente

```
              ┌─────────┐
              │  START  │
              └────┬────┘
                   ▼
           ┌───────────────┐
           │  Planificador │  ¿qué necesito para responder?
           └───────┬───────┘
                   ▼
         ┌─────────────────────┐
         │  Router condicional │
         └──┬────┬────┬────┬───┘
            │    │    │    │
     ┌──────┘    │    │    └──────┐
     ▼           ▼    ▼           ▼
 ┌───────┐  ┌───────┐ ┌────────┐ ┌──────────┐
 │ score │  │ shap  │ │ simular│ │ política │
 └───┬───┘  └───┬───┘ └───┬────┘ └────┬─────┘
     └──────────┴─────┬───┴───────────┘
                      ▼
            ┌──────────────────┐
            │  ¿Suficiente?    │──── no ──┐
            └────────┬─────────┘          │
                     │ sí                 │
                     ▼                    │
            ┌──────────────────┐          │
            │    Síntesis      │◄─────────┘
            └────────┬─────────┘
                     ▼
                 ┌───────┐
                 │  END  │
                 └───────┘
```

Los elementos que aportan aprendizaje real: **estado tipado compartido**, **aristas condicionales**, y el **ciclo de re-planificación** cuando la información recuperada es insuficiente.

### 5.3 Corpus para el RAG

Documentos públicos y reales — nada inventado:

| Documento | Uso |
|---|---|
| Circular Básica Contable y Financiera (SFC), Cap. II — Gestión del Riesgo de Crédito | Criterios de calificación y provisión |
| Ley 1266 de 2008 — Habeas Data financiero | Restricciones de uso de datos |
| Principios de Basilea para gestión de riesgo de crédito | Marco conceptual |
| Política interna de crédito (documento sintético, claramente etiquetado) | Umbrales y reglas de decisión |

> El último documento lo redactamos nosotros porque no existe versión pública de una política interna. Se etiqueta explícitamente como sintético en el README y en el Model Card. Transparencia sobre las fuentes.

---

## 6. Estrategia de evaluación (offline y online)

La rúbrica pide **métricas offline y online**. Como no hay tráfico real, "online" se construye con honestidad explícita.

### 6.1 Métricas offline — Modelo

| Métrica | Por qué la usamos |
|---|---|
| **PR-AUC** | Métrica principal. Con 22% de positivos, ROC-AUC es optimista |
| ROC-AUC | Comparabilidad con la literatura del dominio |
| KS / Gini | Estándar de la industria financiera; los evaluadores lo esperan |
| **Brier score** | Calidad de la calibración — sin esto no hay pérdida esperada |
| Curva de calibración | Diagnóstico visual del punto anterior |
| Precision@k | "De los 100 que marqué como riesgosos, ¿cuántos incumplieron?" |
| Matriz de confusión con costos | Traduce el modelo a pesos, no a porcentajes |

**Validación:** Stratified K-Fold (k=5), `random_state=42`, con **nested CV** para el tuning de hiperparámetros. El preprocesamiento va dentro del pipeline para evitar leakage entre folds.

### 6.2 Métricas offline — Sistema GenAI

| Métrica | Cómo se mide |
|---|---|
| Retrieval hit@k | Set de 20 preguntas con fragmento correcto anotado a mano |
| Groundedness | ¿Cada afirmación está respaldada por un fragmento recuperado? |
| Tasa de alucinación | Afirmaciones sin respaldo / total de afirmaciones |
| Precisión de tool-calling | ¿Invocó las herramientas correctas para la consulta? |

### 6.3 Métricas online — Simulación de producción

Un *holdout* que nunca vio el entrenamiento se convierte en un flujo de peticiones contra la API desplegada:

| Métrica | Qué revela |
|---|---|
| Latencia p50 / p95 / p99 | Viabilidad operativa (objetivo: p95 < 200 ms) |
| Throughput | Peticiones por segundo sostenidas |
| PSI / KS por feature | Data drift entre train y el flujo servido |
| Degradación de PR-AUC en el flujo | Performance real vs. laboratorio |
| Tasa de error de la API | Robustez |
| Latencia end-to-end del agente | Experiencia de uso |
| Costo por consulta (tokens) | Viabilidad económica |

**Todo esto se loguea como runs de MLflow.** El monitoreo no es un dashboard aparte: es experimentación registrada.

---

## 7. Stack técnico

| Capa | Herramienta | Nota |
|---|---|---|
| Gestión de paquetes | **UV** | Exclusivamente; nunca pip |
| Datos | pandas, pyarrow | Parquet para datos procesados |
| Visualización | matplotlib | Solo para los notebooks. **Dependencia de desarrollo, no de runtime**: ni la API ni el pipeline dibujan nada |
| Notebooks | nbconvert | Ejecuta el notebook completo con un kernel limpio. Existe para que *"corre de arriba a abajo"* sea **un comando reproducible y no una afirmación manual**. También dependencia de desarrollo |
| ML | scikit-learn, LightGBM | Pipeline + ColumnTransformer |
| Tuning | Optuna | Con pruning |
| Explicabilidad | SHAP | TreeExplainer |
| Balanceo | imbalanced-learn | Comparar contra `class_weight` |
| Tracking | **MLflow + DagsHub** | Registry incluido |
| Orquestación GenAI | **LangGraph** | Estado tipado + aristas condicionales |
| Vector store | ChromaDB | Persistente en disco |
| Embeddings | sentence-transformers | Modelo multilingüe (corpus en español) |
| LLM | Anthropic API | `claude-haiku-4-5` para tools, modelo mayor para síntesis |
| API | FastAPI + Pydantic | Validación estricta |
| Testing | pytest, pytest-cov | Objetivo: >80% coverage |
| Calidad | ruff, mypy | Pre-commit hooks |
| Contenedores | Docker + Compose | Multi-stage, usuario no-root |
| Registry | GHCR | **Reto ML 1** |
| CI/CD | GitHub Actions | Lint + tests + build + push |
| Cloud | Azure Container Apps + Terraform | **Reto ML 2** |

---

## 8. Estructura del repositorio

```
credit-risk-copilot/
├── README.md                      ← el entregable principal
├── pyproject.toml
├── uv.lock
├── Taskfile.yml
├── .gitignore  .env.example  .pre-commit-config.yaml
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODEL_CARD.md              ← requisito explícito
│   ├── DATA_DICTIONARY.md         ← requisito explícito
│   ├── GIT_STRATEGY.md            ← requisito explícito
│   ├── EVALUATION.md
│   └── adr/
│       ├── 0001-seleccion-dataset.md
│       ├── 0002-metrica-principal.md
│       ├── 0003-estrategia-validacion.md
│       ├── 0004-arquitectura-agente.md
│       └── 0005-estrategia-chunking.md
│
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── corpus/                    ← documentos del RAG
│
├── notebooks/
│   ├── 01_preprocessing.ipynb     ← requisito explícito
│   └── 02_machine_learning.ipynb  ← requisito explícito
│
├── src/credit_copilot/            ← módulo reusable (requisito)
│   ├── config.py
│   ├── data/          loader · validator · preprocessor
│   ├── features/      builder · transformers
│   ├── models/        train · evaluate · calibrate · registry
│   ├── explain/       shap_service · counterfactual
│   ├── rag/           ingest · chunking · vectorstore · retriever
│   ├── agent/         graph · state · tools · prompts
│   ├── api/           app · schemas · dependencies
│   └── monitoring/    drift · metrics
│
├── scripts/                       ← scripts de ejecución (requisito)
│   ├── run_preprocessing.py
│   ├── run_training.py
│   ├── run_prediction.py
│   ├── build_rag_index.py
│   ├── run_online_simulation.py
│   └── run_agent_eval.py
│
├── tests/
├── docker/
│   ├── Dockerfile.api
│   └── docker-compose.yml
├── infra/                         ← Terraform (Reto ML 2)
└── .github/workflows/ci.yml
```

---

## 9. Estrategia de Git

**Modelo:** GitHub Flow adaptado con rama `develop` persistente, siguiendo la convención de git-flow (la rúbrica exige main + «Development»: es esta misma rama).

```
main ──────●───────────────────────────●─────► v1.0.0
            \                         /
develop      ●──●──●──●──●──●──●──●──●
              \  /    \  /    \  /
        feature/*  feature/*  feature/*
```

**Reglas:**
- `main` protegida; solo recibe merges desde `develop` vía PR
- Una rama por fase del roadmap, numerada desde `00` para que coincida con las fases de la
  sección 10: `feature/00-fundacion`, `feature/01-data-and-eda`, `feature/02-modeling`,
  `feature/03-genai`, `feature/04-production`, `feature/05-closing`
- Commits con Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`

> **Excepción histórica.** `feature/00-fundacion` conserva su nombre en español. Renombrarla
> exigiría reescribir el historial, y no lo vale. **De la fase 1 en adelante todos los nombres
> de rama van en inglés**, según la convención de idioma de `docs/GIT_STRATEGY.md`.

- Todo PR con descripción de cambios, checklist y evidencia (métrica o screenshot)
- Merge commits, no squash — la rúbrica pide evidencia de PRs cerradas exitosamente
- Tag `v1.0.0` al final, con release notes redactadas

**Cadencia objetivo:** una PR cerrada por fase → ~7 PRs. Muy por encima del mínimo exigido.

---

## 10. Roadmap de 15 días

> **Supuesto:** ~3-4 horas diarias, ~50 horas totales. Ajustar según disponibilidad real.
>
> **Convención:** cada fase termina con una rama mergeada a `develop` vía PR.

---

### FASE 0 — Fundación · Días 1-2

**Objetivo:** que el esqueleto esté completo antes de tocar una sola línea de modelado.

#### Día 1 — Andamiaje

- [ ] Crear repo en GitHub, ramas `main` y `develop`, protección de `main`
- [ ] Estructura de carpetas completa con `.gitkeep` en las provisionales
- [ ] `pyproject.toml` + `uv sync`
- [ ] `.gitignore` (datos, `.env`, `.venv`, `__pycache__`, `mlruns/`)
- [ ] `.env.example` con las variables necesarias
- [ ] `ruff` + `mypy` + pre-commit configurados
- [ ] CI mínimo en GitHub Actions: lint + import del paquete
- [ ] `docs/GIT_STRATEGY.md`
- [ ] ADR-0001 (dataset) y ADR-0002 (métrica principal)

**Concepto a estudiar hoy:** por qué PR-AUC y no accuracy en datos desbalanceados. Escribirlo en el ADR con tus palabras.

#### Día 2 — Datos crudos y contrato

- [ ] Script de descarga del dataset UCI → `data/raw/`
- [ ] Inspección inicial: shape, tipos, nulos, distribución del target
- [ ] `docs/DATA_DICTIONARY.md` — las 24 columnas, tipo, rango, significado
- [ ] Módulo `data/validator.py`: validación de esquema, rangos y tipos
- [ ] Tests del validador
- [ ] Cuenta de DagsHub creada y conectada al repo

> ✅ **Hito 0:** repo profesional funcionando, CI en verde, datos versionados y documentados.
> 🔀 **PR #1:** `feature/00-fundacion` → `develop`

---

### FASE 1 — Datos y EDA · Días 3-4

#### Día 3 — EDA

- [ ] `notebooks/01_preprocessing.ipynb`
  - [ ] Distribución del target y magnitud del desbalance
  - [ ] Univariado: histogramas, boxplots, detección de outliers
  - [ ] Bivariado contra el target: tasa de default por segmento
  - [ ] Matriz de correlación y detección de multicolinealidad (VIF)
  - [ ] Anomalías en categóricas (categorías no documentadas)
  - [ ] **Al menos 12 visualizaciones con interpretación escrita**
- [ ] Documentar 5+ hipótesis derivadas del EDA

**Conceptos a estudiar hoy:** qué revela un boxplot, cómo leer VIF, por qué la multicolinealidad afecta a unos modelos y no a otros.

#### Día 4 — Feature engineering y pipeline

- [ ] `features/builder.py` con las 7 features derivadas de la sección 4.4
- [ ] `data/preprocessor.py`: `ColumnTransformer` (OHE categóricas, escalado numéricas, bucketing donde aplique)
- [ ] Pipeline completo y serializable
- [ ] Guardar procesado en parquet
- [ ] `scripts/run_preprocessing.py` funcional end-to-end
- [ ] Tests: shape esperado, features derivadas correctas, y **que todo valor faltante
      tenga su columna indicadora correspondiente y ninguna feature impute en silencio**
      (las features producen faltantes **por diseño**, según la
      [decisión 2 del ADR-0005](adr/0005-diseno-de-features-de-comportamiento.md): un
      denominador que no supera el piso da `NaN`, nunca `0`)
- [ ] Cerrar el notebook 01 con conclusiones

**Concepto clave:** por qué el preprocesamiento va **dentro** del pipeline y no antes del split. Escribirlo en el notebook.

> ✅ **Hito 1:** dataset procesado, reproducible con un comando.
> 🔀 **PR #2:** `feature/01-data-and-eda` → `develop`

---

### FASE 2 — Modelado · Días 5-7

#### Día 5 — Baselines y MLflow

- [ ] MLflow apuntando a DagsHub; primer experimento
- [ ] Baseline trivial (clase mayoritaria) — el piso contra el que todo se compara
- [ ] Regresión logística con regularización L1 y L2
- [ ] Modelo solo-demografía vs. modelo solo-comportamiento → **contraste directo de la hipótesis principal**
- [ ] Loguear métricas, parámetros y artefactos en cada run

**Concepto a estudiar hoy:** qué hace realmente la regularización L1 vs L2 y por qué L1 produce selección de features.

#### Día 6 — Modelos avanzados y tuning

- [ ] Random Forest, Gradient Boosting, LightGBM
- [ ] Estrategias de desbalance: `class_weight` vs. SMOTE vs. sin tratamiento → **comparar, no asumir**
- [ ] Tuning con Optuna (nested CV para evitar sesgo optimista)
- [ ] Tabla comparativa de todos los modelos con intervalos de confianza

**Concepto a estudiar hoy:** por qué el boosting funciona; en qué se diferencia de bagging.

#### Día 7 — Calibración, explicabilidad y registro

- [ ] Calibración (`CalibratedClassifierCV`): comparar sigmoid vs. isotonic
- [ ] Curva de calibración antes/después + Brier score
- [ ] Selección del threshold operativo con matriz de costos explícita
- [ ] SHAP: summary plot, dependence plots, waterfall de casos individuales
- [ ] Análisis de errores: caracterizar dónde falla el modelo
- [ ] **Registrar el modelo ganador en MLflow Model Registry, etapa Production**
- [ ] `scripts/run_training.py` y `scripts/run_prediction.py`
- [ ] `docs/MODEL_CARD.md` — primera versión completa

> ✅ **Hito 2:** modelo productivo registrado, calibrado y explicable.
> 🔀 **PR #3:** `feature/02-modeling` → `develop`

---

### FASE 3 — GenAI · Días 8-11

#### Día 8 — Corpus y RAG

- [ ] Recopilar los documentos normativos en `data/corpus/`
- [ ] Redactar la política interna sintética (etiquetada como tal)
- [ ] `rag/chunking.py`: estrategia de chunking + ADR-0005 justificándola
- [ ] Embeddings con modelo multilingüe → ChromaDB persistente
- [ ] `scripts/build_rag_index.py`
- [ ] Set de evaluación: 20 preguntas con fragmento correcto anotado

**Conceptos a estudiar hoy:** qué es un embedding, por qué la similitud coseno, cómo el tamaño de chunk afecta la recuperación.

#### Día 9 — Evaluación del retrieval

- [ ] Medir hit@1, hit@3, hit@5 sobre el set de evaluación
- [ ] Probar 2-3 estrategias de chunking y comparar
- [ ] Loguear los resultados como experimentos de MLflow
- [ ] Documentar la estrategia ganadora en el ADR

> Este día es el que separa un RAG serio de uno decorativo. **No saltarlo.**

#### Día 10 — Tools y grafo

- [ ] Implementar las 4 tools con contratos Pydantic estrictos
- [ ] `explain/counterfactual.py` — el simulador de escenarios
- [ ] `agent/state.py` — estado tipado del grafo
- [ ] `agent/graph.py` — nodos, aristas condicionales, ciclo de re-planificación
- [ ] Tests unitarios de cada tool de forma aislada

**Conceptos a estudiar hoy:** cómo funciona el tool-calling de un LLM; qué es el estado en LangGraph y por qué importa.

#### Día 11 — Evaluación del agente

- [ ] Set de evaluación de ~15 consultas de analista
- [ ] Medir precisión de tool-calling, groundedness, tasa de alucinación
- [ ] **Baseline de contraste:** LLM sin tools sobre las mismas consultas → esto valida la hipótesis secundaria
- [ ] Loguear todo en MLflow
- [ ] Medir tokens y costo por consulta

> ✅ **Hito 3:** agente funcional y **medido**, no solo funcionando.
> 🔀 **PR #4:** `feature/03-genai` → `develop`

---

### FASE 4 — Producción · Días 12-13

#### Día 12 — API y contenedores

- [ ] FastAPI: `/predict`, `/explain`, `/simulate`, `/chat`, `/health`, `/model-info`
- [ ] Schemas Pydantic con validación estricta y manejo de errores
- [ ] Logging estructurado con IDs de correlación
- [ ] Dockerfile multi-stage, usuario no-root, imagen optimizada
- [ ] `docker-compose.yml`: API + MLflow + ChromaDB
- [ ] Tests de integración de los endpoints
- [ ] GitHub Actions: build y push a GHCR → **Reto ML 1 cubierto**

#### Día 13 — Monitoreo y métricas online

- [ ] `monitoring/drift.py`: PSI y KS por feature
- [ ] `scripts/run_online_simulation.py`: el holdout convertido en flujo de peticiones
- [ ] Medir latencia p50/p95/p99, throughput, tasa de error
- [ ] Medir degradación de PR-AUC en el flujo simulado
- [ ] Loguear todas las métricas online en MLflow
- [ ] *(Opcional, si sobra tiempo)* Terraform + Azure Container Apps → **Reto ML 2**

> ✅ **Hito 4:** sistema desplegable, medido en condiciones de producción simulada.
> 🔀 **PR #5:** `feature/04-production` → `develop`

---

### FASE 5 — Cierre · Días 14-15

#### Día 14 — Documentación

- [ ] **README.md completo** con las 6 secciones exigidas:
  - [ ] Problema de ML
  - [ ] Diagrama de flujo del proyecto
  - [ ] Descripción del dataset + diccionario de datos
  - [ ] Model Card
  - [ ] Resultados con métricas offline **y** online
  - [ ] Conclusiones
- [ ] Model Card final: limitaciones, sesgos, uso previsto, uso NO previsto
- [ ] `ARCHITECTURE.md` con decisiones justificadas
- [ ] Docstrings completos en todo el módulo
- [ ] Instrucciones de ejecución verificadas **desde cero en un entorno limpio**

#### Día 15 — Congelar

- [ ] Cobertura de tests ≥ 80%
- [ ] CI en verde
- [ ] Revisión final: ningún secreto en el repo, ningún dato pesado
- [ ] Verificar que el link público de MLflow/DagsHub funciona
- [ ] PR de cierre `feature/05-closing` → `develop`
- [ ] PR de release `develop` → `main`
- [ ] **Tag `v1.0.0` con release notes redactadas**
- [ ] **Congelar: ningún commit posterior**

> ✅ **Hito 5 — Proyecto entregado.**
> 🔀 **PR #6:** `feature/05-closing` → `develop` — documentación, Model Card y cobertura de tests
> 🔀 **PR #7:** `develop` → `main` — release, precede al tag `v1.0.0`

---

## 11. Trazabilidad contra la rúbrica

| Requisito | Peso | Dónde se cumple | Fase |
|---|---|---|---|
| Estructura de repo definida | 25% | Sección 8 | 0 |
| Notebook de preprocesamiento | — | `notebooks/01_preprocessing.ipynb` | 1 |
| Notebook de ML | — | `notebooks/02` | 2 |
| Carpeta de datos | — | `data/` con parquet | 1 |
| Módulo reusable | — | `src/credit_copilot/` | 0-4 |
| Scripts de ejecución | — | `scripts/` | 1-4 |
| Modelo de ML documentado | 50% | Fases 2 | 2 |
| Técnica GenAI alineada | 50% | Agente + RAG + tools | 3 |
| MLflow con métricas/params/artefactos | 50% | DagsHub, todas las fases | 2-4 |
| Modelo productivo | 50% | Model Registry, etapa Production | 2 |
| Commits, PRs, releases | 15% | Sección 9 | Todas |
| Ramas Main + Development | 15% | Sección 9 | 0 |
| Documentación de estrategia git | 15% | `docs/GIT_STRATEGY.md` | 0 |
| README con 6 secciones | 10% | Fase 5 | 5 |
| Model Card | 10% | `docs/MODEL_CARD.md` | 2, 5 |
| Docstrings y naming | 10% | Todo el módulo | Todas |
| **Reto ML 1** — contenedores + registry | +10% | Docker + GHCR | 4 |
| **Reto ML 2** — Azure + Actions + Terraform | +10% | `infra/` + CI/CD | 4 |

---

## 12. Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| El modelo rinde por debajo de lo esperado | Media | El proyecto no se juzga por el AUC sino por el rigor. Análisis de errores bien hecho vale más que una métrica bonita |
| LangGraph consume más tiempo del previsto | **Alta** | Fase 3 tiene 4 días. Si se desborda, entregar un grafo lineal funcional antes que uno complejo a medias |
| El RAG recupera fragmentos irrelevantes | Media | El día 9 existe exactamente para esto. Iterar chunking, no aceptar el primer resultado |
| Costos de API del LLM | Baja | Haiku para las tools, cachear respuestas en desarrollo, presupuesto acotado |
| Retos opcionales comen tiempo del core | **Alta** | Son opcionales. **Nunca** sacrificar el 100% obligatorio por el 20% extra |
| Desbordar los 15 días | Media | Cada fase tiene un hito mínimo viable. Si hay retraso, recortar alcance de fase, no saltarse fases |

**Regla de corte:** si al terminar el día 11 el agente no está funcional, se entrega la versión más simple que funcione y se documenta lo que faltó. Un sistema modesto pero completo y honesto supera a uno ambicioso e incompleto.

---

## 13. Principios de trabajo

1. **Arquitecto y ejecutor separados.** Las decisiones técnicas se toman en el chat; Claude Code escribe el código a partir de prompts precisos. Un prompt vago produce código vago.

2. **Entender antes de ejecutar.** No se copia una línea sin poder explicar por qué está ahí. Si no se puede explicar, se pregunta antes de avanzar.

3. **Cada decisión se documenta.** Los ADRs no son burocracia: son el registro de por qué el proyecto es como es. Se escriben en el momento, no al final.

4. **Medir en vez de suponer.** ¿SMOTE mejora? Se mide. ¿Chunks de 500 o de 1000? Se mide. ¿El agente supera al LLM solo? Se mide.

5. **Reproducibilidad no negociable.** `random_state=42` en todas partes. Un comando debe reproducir el resultado desde cero.

6. **Honestidad sobre limitaciones.** Lo que el modelo no puede hacer se escribe en el Model Card. Reconocer un límite es más profesional que ocultarlo.

7. **El commit final es el entregable.** Se planifica el cierre, no se improvisa.

---

*Documento vivo — actualizar al cerrar cada fase.*