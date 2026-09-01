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
11 bis. [Alcance no cubierto](#11-bis-alcance-no-cubierto)
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
- [x] Modelo de clasificación con probabilidad calibrada, registrado como modelo productivo en MLflow
- [x] Agente conversacional que responda consultas de riesgo con evidencia trazable
- [x] API REST que sirva ambos componentes
- [x] Sistema desplegable con un solo comando

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
| ML | scikit-learn | Pipeline + ColumnTransformer. El boosting es `HistGradientBoostingClassifier` — ver la nota de abajo |
| Tuning | Optuna | Con pruning |
| Explicabilidad | SHAP | TreeExplainer |
| Balanceo | imbalanced-learn | Comparar contra `class_weight` |
| Tracking | **MLflow + DagsHub** | Registry incluido |
| Orquestación GenAI | **LangGraph** | Estado tipado + aristas condicionales |

> **Sustitución de LightGBM, 2026-08-26.** LightGBM estaba en el stack y **no era utilizable
> en la máquina de desarrollo**. El paquete instala y su wheel trae `lib_lightgbm.dll`, pero
> cargarla requiere el runtime de Microsoft Visual C++, y este sistema solo tenía las
> variantes `*_clr0400` que empaqueta .NET. El fallo aparece al **importar**, no al
> instalar: `uv add lightgbm` reporta éxito y `import lightgbm` lanza `FileNotFoundError`.
>
> Se sustituye por **`HistGradientBoostingClassifier`** de scikit-learn: la misma familia de
> boosting por histogramas, ya instalada, sin dependencia nativa que resolver. La medición
> está en la entrada 004 de `docs/EVALUATION.md`, y esa entrada deja explícito que **no dice
> nada sobre LightGBM**.
>
> **Actualización, 2026-08-30 — la limitación ya no existe, y la sustitución se mantiene.**
> Durante la fase 3, `sentence-transformers` falló al importar `torch` por **este mismo
> mecanismo**: la DLL presente, el runtime ausente. Se instaló
> `winget install Microsoft.VCRedist.2015+.x64` y `import lightgbm` pasó a funcionar
> (verificado: `lightgbm 4.7.0`). La afirmación negativa de arriba caducó y queda corregida
> aquí en vez de borrada.
>
> **No se reejecuta la fase 2.** El boosting quedó por debajo del umbral de significancia
> práctica frente a la logística (+0,0163 contra 0,02) y por debajo del random forest, así
> que la sustitución no cambió qué modelo se eligió, y volver a medir movería un número
> dentro del ruido. La decisión del **ADR-0007** sigue siendo válida con la evidencia que la
> sustentó.
>
> La lección de fondo no es sobre LightGBM: **un fallo de carga de DLL nativa en esta máquina
> es una clase de error, no un incidente**, y se repitió idéntico con `torch` cuatro días
> después. Queda registrado en `docs/ERRORS_AND_LEARNINGS.md`.
| Vector store | ChromaDB | Persistente en disco |
| Embeddings | sentence-transformers | `intfloat/multilingual-e5-base`. **Corpus mayoritariamente en español, con un documento en inglés**: el BIS no publica versión oficial en español de BCBS 75, y traducirlo produciría un texto no citable, así que se indexa en su idioma original y el modelo multilingüe resuelve la consulta entre idiomas |
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
├── docs/                          ← el árbol REAL, no el planeado; ver la nota de abajo
│   ├── METHODOLOGY.md
│   ├── ROADMAP.md
│   ├── MODEL_CARD.md              ← requisito explícito
│   ├── DATA_DICTIONARY.md         ← requisito explícito
│   ├── GIT_STRATEGY.md            ← requisito explícito
│   ├── EVALUATION.md
│   ├── ERRORS_AND_LEARNINGS.md
│   ├── analysis/                  ← evidencia reproducible por un script
│   │   ├── undocumented-codes-evidence.md
│   │   ├── null-distribution-evidence.md
│   │   ├── fairness-evidence.md
│   │   ├── retrieval-evidence.md
│   │   ├── agent-evaluation-evidence.md
│   │   └── online-metrics-evidence.md
│   └── adr/
│       ├── 0001-seleccion-del-dataset.md
│       ├── 0002-metrica-principal-de-evaluacion.md
│       ├── 0003-nombre-de-la-rama-de-integracion.md
│       ├── 0004-codigos-no-documentados-de-pay-status.md
│       ├── 0005-diseno-de-features-de-comportamiento.md
│       ├── 0006-protocolo-de-verificacion-de-leakage.md
│       ├── 0007-decisiones-del-modelo-productivo.md
│       ├── 0008-estrategia-de-chunking-del-corpus.md
│       ├── 0009-diseno-del-agente.md
│       └── 0010-arquitectura-de-despliegue.md
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
│   ├── rag/           documents · chunking · embeddings · vectorstore
│   ├── agent/         graph · state · tools · prompts
│   ├── api/           schemas · dependencies · model_app · agent_app
│   └── monitoring/    drift · metrics
│
├── scripts/                       ← scripts de ejecución (requisito)
│   ├── run_preprocessing.py
│   ├── run_training.py
│   ├── run_prediction.py
│   ├── build_rag_index.py
│   ├── run_online_simulation.py
│   ├── run_agent_latency.py
│   └── evaluate_agent.py
│
├── tests/
├── docker/
│   ├── Dockerfile.model           ← imagen del modelo, sin torch (ADR-0010)
│   ├── Dockerfile.agent           ← imagen del copiloto
│   └── docker-compose.yml
└── .github/workflows/
    ├── ci.yml                     lint · formato · tipos · tests
    └── docker.yml                 build · verificación en la imagen · push a GHCR
```

> **Dos ausencias en este árbol son decisiones, no pendientes (2026-08-31).**
>
> **No hay `docs/ARCHITECTURE.md`.** El README cubre el contrato de comportamiento con sus dos
> diagramas de flujo, así que un documento aparte duplicaría ese contenido contra la regla de
> no solapamiento de la sección 4 de `docs/METHODOLOGY.md`, donde la fila correspondiente
> también se retiró.
>
> **No hay `infra/`.** El **Reto ML 2** no se intentó — ver la sección 11.
>
> **El resto del árbol sigue siendo el planeado y diverge del real en `scripts/`**, que hoy
> tiene veintitrés scripts en vez de los siete previstos: uno por medición, más
> `run_training.py` y `run_prediction.py`, que el enunciado exige por su nombre y que la fase 5
> añadió como envoltorios sobre lo que ya existía. Queda señalado y no corregido: este
> documento es el plan, y el árbol que describe el repositorio entregado es el de la sección 11
> del README.

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

- [x] Crear repo en GitHub, ramas `main` y `develop`, protección de `main`
- [x] Estructura de carpetas completa con `.gitkeep` en las provisionales
- [x] `pyproject.toml` + `uv sync`
- [x] `.gitignore` (datos, `.env`, `.venv`, `__pycache__`, `mlruns/`)
- [x] `.env.example` con las variables necesarias
- [x] `ruff` + `mypy` + pre-commit configurados
- [x] CI mínimo en GitHub Actions: lint + import del paquete
- [x] `docs/GIT_STRATEGY.md`
- [x] ADR-0001 (dataset) y ADR-0002 (métrica principal)

**Concepto a estudiar hoy:** por qué PR-AUC y no accuracy en datos desbalanceados. Escribirlo en el ADR con tus palabras.

#### Día 2 — Datos crudos y contrato

- [x] Script de descarga del dataset UCI → `data/raw/`
- [x] Inspección inicial: shape, tipos, nulos, distribución del target
- [x] `docs/DATA_DICTIONARY.md` — las 24 columnas, tipo, rango, significado
- [x] Módulo `data/validator.py`: validación de esquema, rangos y tipos
- [x] Tests del validador
- [x] Cuenta de DagsHub creada y conectada al repo

> ✅ **Hito 0:** repo profesional funcionando, CI en verde, datos versionados y documentados.
> 🔀 **PR #1:** `feature/00-fundacion` → `develop`

---

### FASE 1 — Datos y EDA · Días 3-4

#### Día 3 — EDA

- [x] `notebooks/01_preprocessing.ipynb`
  - [x] Distribución del target y magnitud del desbalance
  - [ ] Univariado: histogramas, boxplots, detección de outliers
  - [x] Bivariado contra el target: tasa de default por segmento
  - [x] Matriz de correlación y detección de multicolinealidad (VIF)
  - [x] Anomalías en categóricas (categorías no documentadas)
  - [x] **Al menos 12 visualizaciones con interpretación escrita**
- [x] Documentar 5+ hipótesis derivadas del EDA

**Conceptos a estudiar hoy:** qué revela un boxplot, cómo leer VIF, por qué la multicolinealidad afecta a unos modelos y no a otros.

#### Día 4 — Feature engineering y pipeline

- [x] `features/builder.py` con las 7 features derivadas de la sección 4.4
- [x] `data/preprocessor.py`: `ColumnTransformer` (OHE categóricas, escalado numéricas, bucketing donde aplique)
- [x] Pipeline completo y serializable
- [x] Guardar procesado en parquet
- [x] `scripts/run_preprocessing.py` funcional end-to-end
- [x] Tests: shape esperado, features derivadas correctas, y **que todo valor faltante
      tenga su columna indicadora correspondiente y ninguna feature impute en silencio**
      (las features producen faltantes **por diseño**, según la
      [decisión 2 del ADR-0005](adr/0005-diseno-de-features-de-comportamiento.md): un
      denominador que no supera el piso da `NaN`, nunca `0`)
- [x] Cerrar el notebook 01 con conclusiones

**Concepto clave:** por qué el preprocesamiento va **dentro** del pipeline y no antes del split. Escribirlo en el notebook.

> ✅ **Hito 1:** dataset procesado, reproducible con un comando.
> 🔀 **PR #2:** `feature/01-data-and-eda` → `develop`

---

### FASE 2 — Modelado · Días 5-7

#### Día 5 — Baselines y MLflow

- [x] MLflow apuntando a DagsHub; primer experimento
- [x] Baseline trivial (clase mayoritaria) — el piso contra el que todo se compara
- [ ] Regresión logística con regularización L1 y L2
- [x] Modelo solo-demografía vs. modelo solo-comportamiento → **contraste directo de la hipótesis principal**
- [x] Loguear métricas, parámetros y artefactos en cada run

**Concepto a estudiar hoy:** qué hace realmente la regularización L1 vs L2 y por qué L1 produce selección de features.

#### Día 6 — Modelos avanzados y tuning

- [x] Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`; LightGBM sustituido, ver la nota del stack)
- [x] Estrategias de desbalance: `class_weight` vs. SMOTE vs. sin tratamiento → **comparar, no asumir**
- [x] Tuning con Optuna (nested CV para evitar sesgo optimista)
- [ ] Tabla comparativa de todos los modelos con intervalos de confianza

**Concepto a estudiar hoy:** por qué el boosting funciona; en qué se diferencia de bagging.

#### Día 7 — Calibración, explicabilidad y registro

- [x] Calibración (`CalibratedClassifierCV`): comparar sigmoid vs. isotonic
- [x] Curva de calibración antes/después + Brier score
- [x] Selección del threshold operativo con matriz de costos explícita
- [x] SHAP: summary plot, dependence plots, waterfall de casos individuales
- [ ] Análisis de errores: caracterizar dónde falla el modelo
- [x] **Registrar el modelo ganador en MLflow Model Registry, etapa Production**
- [x] `scripts/run_training.py` y `scripts/run_prediction.py` — **fase 5.** Envoltorios sobre
      lo que ya existía: `run_training.py` delega en `register_production_model.py` y no
      reimplementa nada; `run_prediction.py` compone `load_registered_model`, `ApplicantRecord`
      en modo estricto y `decide`
- [x] `docs/MODEL_CARD.md` — primera versión completa

> ✅ **Hito 2:** modelo productivo registrado, calibrado y explicable.
> 🔀 **PR #3:** `feature/02-modeling` → `develop`

---

### FASE 3 — GenAI · Días 8-11

#### Día 8 — Corpus y RAG

- [x] Recopilar los documentos normativos en `data/corpus/`
- [x] Redactar la política interna sintética (etiquetada como tal)
- [x] `rag/chunking.py`: estrategia de chunking + ADR-0005 justificándola
- [x] Embeddings con modelo multilingüe → ChromaDB persistente
- [x] `scripts/build_rag_index.py`
- [x] Set de evaluación: 20 preguntas con fragmento correcto anotado

**Conceptos a estudiar hoy:** qué es un embedding, por qué la similitud coseno, cómo el tamaño de chunk afecta la recuperación.

#### Día 9 — Evaluación del retrieval

- [x] Medir hit@1, hit@3, hit@5 sobre el set de evaluación
- [x] Probar 2-3 estrategias de chunking y comparar
- [x] Loguear los resultados como experimentos de MLflow
- [x] Documentar la estrategia ganadora en el ADR

> Este día es el que separa un RAG serio de uno decorativo. **No saltarlo.**

#### Día 10 — Tools y grafo

- [x] Implementar las 4 tools con contratos Pydantic estrictos
- [x] `explain/counterfactual.py` — el simulador de escenarios
- [x] `agent/state.py` — estado tipado del grafo
- [x] `agent/graph.py` — nodos, aristas condicionales, ciclo de re-planificación
- [x] Tests unitarios de cada tool de forma aislada

**Conceptos a estudiar hoy:** cómo funciona el tool-calling de un LLM; qué es el estado en LangGraph y por qué importa.

#### Día 11 — Evaluación del agente

- [x] Set de evaluación de ~15 consultas de analista
- [x] Medir precisión de tool-calling, groundedness, tasa de alucinación
- [x] **Baseline de contraste:** LLM sin tools sobre las mismas consultas → esto valida la hipótesis secundaria
- [x] Loguear todo en MLflow
- [x] Medir tokens y costo por consulta

> ✅ **Hito 3:** agente funcional y **medido**, no solo funcionando.
> 🔀 **PR #4:** `feature/03-genai` → `develop`

---

### FASE 4 — Producción · Días 12-13

#### Día 12 — API y contenedores

- [x] FastAPI: `/predict`, `/explain`, `/simulate`, `/chat`, `/health`, `/model-info`
- [x] Schemas Pydantic con validación estricta y manejo de errores
- [x] Logging estructurado con IDs de correlación
- [x] Dockerfile multi-stage, usuario no-root, imagen optimizada
- [x] `docker-compose.yml`: **dos servicios de aplicación**, `model` y `agent`. Ni MLflow ni
      ChromaDB son servicios del compose: **MLflow es remoto** y **Chroma va embebido** en el
      proceso del copiloto como cliente persistente sobre un índice montado (ADR-0010)
- [x] Tests de integración de los endpoints
- [x] GitHub Actions: build y push a GHCR → **Reto ML 1 cubierto**

#### Día 13 — Monitoreo y métricas online

- [x] `monitoring/drift.py`: PSI y KS por feature
- [x] `scripts/run_online_simulation.py`: el holdout convertido en flujo de peticiones
- [x] Medir latencia p50/p95/p99, throughput, tasa de error
- [x] Medir degradación de PR-AUC en el flujo simulado
- [x] Loguear todas las métricas online en MLflow
- [ ] *(Opcional, si sobra tiempo)* Terraform + Azure Container Apps → **Reto ML 2**

> ✅ **Hito 4:** sistema desplegable, medido en condiciones de producción simulada.
> 🔀 **PR #5:** `feature/04-production` → `develop`

---

### FASE 5 — Cierre · Días 14-15

#### Día 14 — Documentación

- [x] **README.md completo** con las 6 secciones exigidas:
  - [x] Problema de ML
  - [x] Diagrama de flujo del proyecto
  - [x] Descripción del dataset + diccionario de datos
  - [x] Model Card
  - [x] Resultados con métricas offline **y** online
  - [x] Conclusiones
- [x] Model Card final: limitaciones, sesgos, uso previsto, uso NO previsto
- [x] Docstrings completos en todo el módulo
- [x] Instrucciones de ejecución verificadas **desde cero en un entorno limpio** — clon nuevo,
      sin `.venv`, sin `.env` y sin datos, siguiendo solo el README. Ver el reporte del turno 3
      de la fase 5

#### Día 15 — Congelar

- [x] Cobertura de tests ≥ 80% — **82%**, 380 tests
- [ ] CI en verde
- [ ] Revisión final: ningún secreto en el repo, ningún dato pesado
- [x] Verificar que el link público de MLflow/DagsHub funciona — **verificado sin
      credenciales**: la API de MLflow devuelve HTTP 200 y los cuatro experimentos. **Ojo: el
      enlace bueno es el que termina en `.mlflow`**; la página del repositorio en DagsHub es
      privada y redirige a login
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
| **Reto ML 1** — contenedores + registry | +10% | **✅ CUMPLIDO.** Dos imágenes construidas, verificadas y publicadas en GHCR por `.github/workflows/docker.yml`: `model` 1,13 GB y `agent` 2,47 GB (ADR-0010) | 4 |
| **Reto ML 2** — Azure + Actions + Terraform | +10% | **❌ NO INTENTADO, por decisión de alcance.** Ver abajo | — |

> **Por qué el Reto ML 2 no se intentó, y por qué se registra como no intentado y no como
> pendiente (2026-08-31).** Al llegar a la fase 4 quedaban dos días de los quince, y la
> sección 12 de este documento ya había fijado la regla de corte antes de saber si haría
> falta: *"son opcionales. **Nunca** sacrificar el 100% obligatorio por el 20% extra"*. El
> Reto ML 2 exige Terraform y una suscripción de Azure, y su parte de CI/CD —el pipeline que
> construye, verifica y publica— **sí se hizo**, dentro del Reto ML 1. Lo que no existe es la
> capa de infraestructura como código y el despliegue en la nube.
>
> **«No intentado» y «pendiente» no son lo mismo**, y la diferencia es lo que este registro
> existe para conservar: un pendiente afirma que alguien va a volver, y aquí nadie va a
> volver. La casilla del día 13 queda **sin marcar a propósito**, con esta nota al lado.

---

## 11 bis. Alcance no cubierto

**Cuatro tareas de este plan no se hicieron, y se registran en vez de borrarse.** Un plan que
oculta lo que no se hizo deja de servir como registro: el lector no puede distinguir entre lo
que se decidió no hacer y lo que nadie miró. Las cuatro son **mejoras reales** y ninguna
cambia una conclusión del proyecto, que es exactamente el criterio con el que se dejaron
fuera al quedar dos días.

| Tarea | Día | Qué falta, con precisión | Por qué no cambia ninguna conclusión |
| --- | ---: | --- | --- |
| **Boxplots en el EDA univariado** | 3 | El notebook 01 tiene 16 figuras, histogramas, VIF y detección de outliers; **cero boxplots**. La casilla los nombra literalmente | Los outliers ya están caracterizados por otra vía, y la decisión que dependía de ellos —`RobustScaler` en vez de `StandardScaler`— se tomó **midiendo** la asimetría y la curtosis, no mirando un dibujo |
| **Variante L1 de la logística** | 5 | `LOGISTIC_L1_RATIO = 0.0`, así que solo se midió L2. El módulo ya declara que la segunda configuración quedaba para otro turno | La logística es el **baseline**, no el candidato productivo. Lo que L1 aportaría es selección de features sobre un modelo que ya se descartó frente al forest por +0,0240 de PR-AUC |
| **Intervalos de confianza en la tabla comparativa** | 6 | `docs/EVALUATION.md` reporta **media ± desviación entre folds**, que no es un intervalo de confianza | Las decisiones del proyecto no se tomaron por solapamiento de intervalos sino contra un **umbral de significancia práctica de 0,02 fijado antes de ver resultados**, y las comparaciones que importaban se hicieron **pareadas fold a fold**, que es más informativo que un IC marginal |
| **Análisis de errores dedicado** | 7 | No existe una caracterización de *dónde* falla el modelo por segmento | Lo que un análisis de errores habría destapado —que el modelo trata distinto a distintos grupos— **está medido y con más rigor** en la entrada 010 y en la sección 6 bis del Model Card, que es equidad medida sobre probabilidades fuera de fold |

**Ninguna de las cuatro está prometida para después.** Si alguien las retoma, el sitio donde
está lo que sí se hizo es: el notebook 01 para el EDA, `models/estimators` para la logística,
`docs/EVALUATION.md` para el protocolo de comparación, y la entrada 010 para los errores por
grupo.

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