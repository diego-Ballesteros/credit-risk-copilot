# Model Card — Probabilidad de incumplimiento de tarjeta de crédito

- **Nombre en el registro:** `credit-risk-default-probability`
- **Versión:** 1
- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Estado:** candidato. **No desplegado**, y la sección *Para qué NO debe usarse* explica
  qué falta antes de que pueda estarlo.

---

## 1 · Qué hace

Recibe las **23 columnas crudas** de un cliente de tarjeta de crédito —cinco atributos
demográficos y dieciocho de comportamiento de pago sobre seis meses— y devuelve la
**probabilidad de que incumpla el pago el mes siguiente**.

Devuelve una **probabilidad, no una etiqueta**. La decisión de aceptar o rechazar se toma
comparando esa probabilidad contra un umbral, y ese umbral es una decisión de negocio que
se documenta en la sección 5 y **no forma parte del modelo**.

### El artefacto

Lo registrado es **un único objeto `Pipeline`** que va de la fila cruda a la probabilidad:

```
Pipeline
├── preprocess   → 23 columnas crudas → matriz de 110 columnas
│   ├── behaviour    22 features derivadas (ADR-0005)
│   ├── education    colapso de códigos no documentados (ADR-0004)
│   ├── clip         cap de ratios al percentil 99,5, aprendido en fit
│   └── columns      one-hot / imputación+escalado robusto / passthrough
└── model        → RandomForestClassifier + calibración sigmoide
```

Se registra el pipeline entero y no solo el clasificador, por la sección 6.3 de
`docs/METHODOLOGY.md`: el notebook, el script de entrenamiento y la API tienen que cargar
**el mismo artefacto**, porque dos implementaciones de la misma aritmética divergen y el
síntoma aparece en producción y no en los tests.

**Firma:** entrada de 23 columnas con los nombres canónicos que devuelve
`loader.load_dataset`; salida `(-1, 2)` en `float64`, donde la columna 1 es la probabilidad
de incumplimiento.

---

## 2 · Con qué datos se entrenó

| | |
| --- | --- |
| **Fuente** | UCI Machine Learning Repository, dataset 350, *Default of Credit Card Clients* |
| **Población** | Clientes de tarjeta de crédito en **Taiwán**, abril–septiembre de **2005** |
| **Filas** | 30.000 |
| **Prevalencia de incumplimiento** | **0,221200** (6.636 casos) |
| **Partición de entrenamiento del artefacto** | Las 30.000 filas |
| **Partición de las métricas** | `StratifiedKFold(5, shuffle=True, random_state=42)` |

### Por qué el artefacto se entrena sobre todo el dataset y las métricas no

Son dos operaciones distintas y confundirlas es el error que este proyecto más ha trabajado
por hacer imposible.

Una **métrica** es una afirmación sobre cómo se comporta el modelo con datos que no ha
visto, y solo es cierta si algo se retuvo de verdad. Por eso el preprocesador se ajusta
**dentro** de cada fold: un escalador que ya vio una fila de validación metió información
sobre ella en el número que esa fila produce después.

El **artefacto** no es una afirmación sobre nada: es el objeto que va a puntuar clientes
nuevos. Retenerle un quinto de los datos lo haría peor a cambio de una estimación que ese
ajuste no produce. Las métricas de la sección 4 vienen de validación cruzada, no de este
ajuste, y `scripts/register_production_model.py` deja esa distinción escrita.

---

## 3 · Configuración

| Componente | Valor | De dónde sale |
| --- | --- | --- |
| Modelo base | `RandomForestClassifier` | Entrada 004: único que superó el umbral de 0,02 frente a la logística |
| `n_estimators` | 300 | Entrada 006, estudio final |
| `max_depth` | 10 | Entrada 006, estudio final |
| `min_samples_leaf` | 18 | Entrada 006, estudio final |
| `max_features` | 0,3 | Entrada 006 — **el único hiperparámetro que los datos restringen**: lo eligieron los cinco folds externos y el estudio final |
| `class_weight` | **`None`** | Entrada 005: reponderar cuesta +0,0404 de Brier y no compra ordenamiento |
| Tratamiento de desbalance | **Ninguno** | Entrada 005: ni `class_weight` ni SMOTE mejoran PR-AUC, y ambos empeoran la calibración |
| Calibración | Sigmoide (Platt), `cv=3`, `ensemble=False` | Entrada 007 |
| Semilla | 42, desde `config.py` | Regla del proyecto |

### Una honestidad sobre el tuning y sobre la calibración

**El tuning no aportó nada distinguible del ruido:** +0,0028 de PR-AUC contra una
desviación entre folds de 0,0080 (entrada 006). Se usan los hiperparámetros tuneados por
`max_features=0,3`, que es la única dimensión sobre la que la evidencia habló; el forest sin
tunear sería una elección igualmente defendible.

**La calibración tampoco ayudó, y eso está medido:** el forest crudo obtuvo el **mejor**
Brier de los tres brazos (0,133228 contra 0,134009 sigmoide y 0,133551 isotónica) y el menor
error en cada decil. Un bosque de 300 árboles promediando frecuencias de hoja ya produce una
probabilidad, no una proporción de votos. Se conserva la calibración sigmoide porque su
costo está medido y es despreciable —**+0,0008 de Brier y exactamente 0,0000 de PR-AUC**— y
porque un mapa de dos parámetros es un seguro barato si la distribución de scores se mueve.
Quitarla es una alternativa defendible y la evidencia está en la entrada 007.

**La calibración isotónica se descartó por una razón medida:** costó **0,0133 de PR-AUC**.
Un mapa monótono no puede reordenar nada, pero la regresión isotónica es solo *no
decreciente*: colapsa rangos de score en un valor constante, eso crea empates, y los empates
son de lo que están hechas `precision@top-k%` y la precisión media.

---

## 4 · Métricas, con su baseline al lado

Validación cruzada de 5 folds, preprocesador ajustado dentro de cada fold, `random_state=42`.
**Ninguna de estas cifras se calculó sobre las filas con las que se ajustó el artefacto.**

| Métrica | Piso sin señal | Baseline trivial | Baseline logística L2 | **Modelo productivo** |
| --- | ---: | ---: | ---: | ---: |
| **PR-AUC** (decisión, ADR-0002) | 0,2212 | 0,2212 ± 0,0001 | 0,5402 ± 0,0103 | **0,5642 ± 0,0080** |
| ROC-AUC | 0,5000 | 0,5000 ± 0,0000 | 0,7762 ± 0,0072 | 0,7863 ± 0,0089 |
| KS | 0,0000 | 0,0000 ± 0,0000 | 0,4218 ± 0,0180 | 0,4392 ± 0,0191 |
| Gini | 0,0000 | 0,0000 ± 0,0000 | 0,5524 ± 0,0143 | 0,5726 ± 0,0177 |
| Brier *(menor es mejor)* | 0,1723 | 0,2212 ± 0,0001 | 0,1834 ± 0,0016 | **0,1334 ± 0,0021** |
| precision@top-10% | 0,2212 | 0,2212 ± 0,0001 | 0,6927 ± 0,0109 | 0,7063 ± 0,0176 |
| precision@top-5% | 0,2212 | 0,2212 ± 0,0001 | 0,7427 ± 0,0166 | 0,7687 ± 0,0090 |

**Lectura de negocio:** `precision@top-10% = 0,7063` significa que revisar el decil peor
puntuado encuentra **3,2 veces más incumplidores** que revisar un decil al azar.

### La alternativa descartada, con su distancia medida

La **regresión logística L2** se descartó como modelo productivo y la distancia es pequeña:

| | Logística | Forest productivo | Diferencia |
| --- | ---: | ---: | ---: |
| PR-AUC | 0,5402 | 0,5642 | **+0,0240** |
| Brier | 0,1834 | 0,1334 | **−0,0500 (27% mejor)** |
| precision@top-10% | 0,6927 | 0,7063 | +0,0136 |

El umbral de significancia práctica del proyecto es **0,02 en PR-AUC** —fijado antes de ver
ningún resultado, porque la desviación entre folds es de 0,010—. La ventaja de +0,0240 lo
supera, pero **por poco**, y buena parte de esas 240 diezmilésimas vienen de cambiar de
modelo y no del tuning ni del tratamiento del desbalance.

**Lo que el forest gana sin ambigüedad es calibración**, un 27% de Brier, y esa es la
diferencia que importa para el uso declarado: sin probabilidad no hay pérdida esperada.
**Lo que se pierde es interpretabilidad**: la logística tiene 110 coeficientes legibles y el
forest necesita SHAP para explicar una decisión.

---

## 5 · Umbral operativo y el supuesto que lo sustenta

**Umbral: 0,160.** Un cliente con probabilidad ≥ 0,160 se marca para rechazo.

### El supuesto

**Un falso negativo cuesta 5 veces un falso positivo.** Prestar a quien incumple cuesta el
principal; rechazar a quien habría pagado cuesta el margen no ganado. Solo importa el
cociente: el umbral que minimiza `5·FN + FP` es el mismo que minimiza `5000·FN + 1000·FP`.

El umbral **no** se eligió maximizando F1 ni ningún criterio interno del modelo. El ADR-0002
descarta F1 exactamente por esto: fija un umbral con un criterio que no sabe lo que cuesta
un error.

### Qué implica sobre 30.000 clientes

| | |
| --- | --- |
| Se rechazan | **11.635** clientes (38,8% del libro) |
| De ellos habrían incumplido | **4.769** (41,0% de los rechazos acertaron) |
| De ellos habrían pagado | **6.866** buenos clientes perdidos (29,4% de todos los pagadores) |
| Incumplidores atrapados | 4.769 de 6.636 (**71,9%**) |
| Incumplidores que pasan | **1.867** |

### Sensibilidad — y es la parte más importante de esta sección

| FN:FP | Umbral | Rechazados | Atrapados | Perdidos | Recall | Precisión |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3:1 | 0,220 | 7.768 | 3.942 | 3.826 | 0,5940 | 0,5075 |
| **5:1** | **0,160** | **11.635** | **4.769** | **6.866** | **0,7187** | **0,4099** |
| 10:1 | 0,105 | 22.329 | 6.200 | 16.129 | 0,9343 | 0,2777 |

**Mover el cociente de 3:1 a 10:1 mueve 14.561 clientes, el 48,5% del libro.** Esa amplitud
es la parte de la recomendación que viene del **supuesto** y no de los datos. Quien fije el
cociente está tomando una decisión mucho más grande que la de elegir el modelo.

---

## 6 · Qué usa el modelo, medido con SHAP

Diez features más importantes por SHAP medio absoluto, sobre 3.000 clientes, con los nombres
verificados de punta a punta —pipeline, modelo y SHAP declaran la misma lista de 110—.

| # | Feature | SHAP medio abs. | Grupo |
| --- | --- | ---: | --- |
| 1 | `IS_DELINQUENT_MOST_RECENT_M1` | 0,054999 | Comportamiento |
| 2 | `PAY_STATUS_1_2` | 0,030297 | Comportamiento |
| 3 | `MAX_DELINQUENCY_M2_M6` | 0,027409 | Comportamiento |
| 4 | `DELINQUENCY_STREAK_M2_M6` | 0,019836 | Comportamiento |
| 5 | `BILL_VOLATILITY_M2_M6` | 0,013601 | Comportamiento |
| 6 | `PAY_STATUS_1_1` | 0,012327 | Comportamiento |
| 7 | `LIMIT_BAL` | 0,007409 | **Demografía** |
| 8 | `UTILIZATION_M2` | 0,007325 | Comportamiento |
| 9 | `PAY_STATUS_2_2` | 0,006902 | Comportamiento |
| 10 | `BILL_AMT1` | 0,006176 | Comportamiento |

**El comportamiento concentra el 95,5% de la atribución total; la demografía, el 4,5%.**
Esto es una confirmación independiente de la hipótesis principal, que la entrada 003 midió
por una vía distinta: allí se midió cuánto *vale* cada grupo para la métrica, aquí cuánto lo
*usa* el modelo. Que coincidan no estaba garantizado.

Artefactos en MLflow: beeswarm global, gráfico de barras, cinco *dependence plots* y tres
*waterfall* individuales (riesgo alto, riesgo bajo, y un caso justo en el umbral).

---

## 7 · Limitaciones conocidas

### 7.1 · No hay validación fuera de tiempo — y un despliegue real la exigiría

**Es la limitación más importante y estaba prevista desde el ADR-0001.** El dataset **no
contiene fecha de originación del crédito**, por lo que no es posible una validación
*out-of-time*. La consecuencia directa, registrada en ese ADR, es que la estrategia de
validación es `StratifiedKFold` y **no un corte cronológico**.

Qué significa en la práctica: todas las métricas de la sección 4 estiman el desempeño sobre
**la misma población y el mismo periodo**. **No dicen nada** sobre cómo se comportaría el
modelo seis meses después, ni sobre su degradación ante un cambio de ciclo económico. Un
despliegue real exige un conjunto de validación posterior en el tiempo, y este proyecto no
puede construirlo con estos datos.

### 7.2 · Población y periodo muy concretos

Taiwán, 2005, clientes de tarjeta de crédito. No se afirma nada sobre otro país, otro
producto de crédito ni otra década. El periodo además **precede a la crisis financiera de
2008**, así que el modelo no ha visto un ciclo adverso.

### 7.3 · Códigos no documentados en los datos de origen

`PAY_STATUS_*` contiene los códigos `-2` y `0`, y `EDUCATION` los códigos `0`, `5` y `6`,
**que la documentación oficial de UCI no declara**. El ADR-0004 decidió qué hacer con cada
uno sobre evidencia medida, pero **su significado real sigue siendo desconocido**. La
feature más importante del modelo, `IS_DELINQUENT_MOST_RECENT_M1`, se construye sobre esa
escala.

### 7.4 · Las predicciones no son reproducibles bit a bit

El forest corre con `n_jobs=-1`, así que `predict_proba` acumula los votos de 300 árboles
entre hilos. La suma en coma flotante no es asociativa, de modo que **el mismo objeto
llamado dos veces sobre la misma fila no devuelve exactamente el mismo número**. Medido:
**5×10⁻¹⁶**, y exactamente cero con `n_jobs=1`. Es quince órdenes de magnitud por debajo del
umbral de 0,160 y no puede cambiar una decisión, pero invalida cualquier test que exija
igualdad de bits sobre predicciones.

### 7.5 · La firma declara enteros

La firma inferida declara las 23 columnas como enteros, porque así llegan de la fuente.
MLflow advierte que un entero en Python no puede representar un valor faltante: si una
petición real llega con un nulo, el enforcement de esquema fallará. El validador del
proyecto garantiza que el dataset no tiene nulos, pero **una API recibe lo que le mandan**.
Está sin resolver y es trabajo de la fase de API.

### 7.6 · SMOTE se evaluó sobre columnas one-hot

La entrada 005 midió SMOTE con la variante estándar sobre una matriz donde 74 de 110
columnas son indicadores. Interpolar indicadores produce valores como 0,37 en una columna
que significa "nivel educativo universitario", que no corresponde a ningún cliente que pueda
existir. `SMOTENC` es la variante para datos mixtos y **no se probó**. Parte del resultado
negativo de SMOTE puede deberse a esto.

### 7.7 · El desempeño es modesto en términos absolutos

PR-AUC 0,5642 sobre un piso de 0,2212. El modelo ordena bastante mejor que el azar y **no es
un oráculo**: con el umbral de 0,160, el 59% de los clientes rechazados habría pagado.

---

## 8 · Para qué NO debe usarse

- **No debe usarse para tomar decisiones de crédito reales sin validación fuera de tiempo.**
  Es la limitación 7.1 y es bloqueante para un despliegue.
- **No debe usarse como decisión automática sin revisión humana.** Con el umbral de 0,160,
  4 de cada 10 rechazos aciertan; los otros 6 son clientes que habrían pagado.
- **No debe usarse fuera de su población.** Otro país, otro producto o una década distinta
  están fuera de lo medido.
- **No debe usarse como si el umbral fuera una propiedad del modelo.** El 0,160 sale de un
  supuesto de costos 5:1; con 10:1 el modelo rechazaría a tres de cada cuatro clientes.
- **No debe usarse para explicar una decisión individual sin SHAP y sin las limitaciones de
  la sección 7.3.** La feature dominante se apoya en códigos cuyo significado real se
  desconoce.
- **No debe usarse para inferir causalidad.** SHAP atribuye la predicción del modelo, no el
  efecto de cambiar una variable en el mundo.
- **No debe usarse sobre variables protegidas sin un análisis de equidad que este proyecto
  todavía no ha hecho.** `SEX`, `EDUCATION`, `MARRIAGE` y `AGE` están entre las entradas, y
  **no se ha medido si el modelo produce tasas de rechazo dispares entre esos grupos**. Que
  la demografía pese solo el 4,5% de la atribución no es prueba de ausencia de sesgo: un peso
  pequeño sobre una población grande puede seguir siendo un trato desigual. Está fuera de lo
  medido hasta hoy.

---

## 9 · Reproducción

| Paso | Comando |
| --- | --- |
| Descargar los datos | `uv run python scripts/download_dataset.py` |
| Comparar modelos | `uv run python scripts/run_model_comparison.py` |
| Medir el desbalance | `uv run python scripts/run_imbalance_comparison.py` |
| Tuning (CV anidada) | `uv run python scripts/run_tuning.py` |
| Comparar calibración | `uv run python scripts/run_calibration.py` |
| Elegir el umbral | `uv run python scripts/run_threshold_selection.py` |
| Calcular SHAP | `uv run python scripts/run_shap_analysis.py` |
| Registrar el modelo | `uv run python scripts/register_production_model.py` |
| Verificar ausencia de fuga | `uv run python scripts/run_leakage_check.py` |

Todas las mediciones citadas están en `docs/EVALUATION.md` con su run de MLflow.

---

## 10 · Verificación de ausencia de fuga

El pipeline pasa la **prueba del target barajado** (entrada 002, ADR-0006): entrenado contra
un target permutado, el desempeño colapsa al azar —ROC-AUC 0,4882 contra una tolerancia de
±0,020 alrededor de 0,5, y PR-AUC 0,2150 contra ±0,015 alrededor de la prevalencia—. Las
tolerancias se derivaron midiendo la distribución nula sobre ocho permutaciones, no de una
intuición.
