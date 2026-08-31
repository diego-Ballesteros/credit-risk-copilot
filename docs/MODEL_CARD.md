# Model Card — Probabilidad de incumplimiento de tarjeta de crédito

- **Nombre en el registro:** `credit-risk-default-probability`
- **Versión:** 1
- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Estado:** candidato. **No desplegado**, y la sección *Para qué NO debe usarse* explica
  qué falta antes de que pueda estarlo.

---

## 0 · Las dos cosas que hay que leer antes que nada

### El supuesto de costos mueve más negocio que todo el modelado

El umbral operativo de **0,160** no sale del modelo: sale de suponer que **un falso negativo
cuesta 5 veces un falso positivo**, y ese cociente **no tiene respaldo empírico en este
dataset**. No hay datos de exposición, de recuperación ni de margen, así que no se pudo
medir: se declaró.

| FN:FP | Umbral | Clientes rechazados de 30.000 |
| --- | ---: | ---: |
| 3:1 | 0,220 | 7.768 |
| **5:1** | **0,160** | **11.635** |
| 10:1 | 0,105 | 22.329 |

**Mover el cociente entre 3:1 y 10:1 desplaza a 14.561 clientes: el 48,5% del libro.**

Para comparar: **todo lo ganado por modelado en esta fase** —pasar de regresión logística a
random forest, quitar el tratamiento de desbalance y tunear— **suma +0,0240 de PR-AUC**.
Quien fija el cociente de costos toma una decisión mucho mayor que la de elegir el modelo.
Conseguir datos que permitan medirlo tiene más valor esperado que cualquier mejora adicional
del modelo. Detalle en la sección 5 y en la entrada 008 de `docs/EVALUATION.md`.

### El modelo trata de forma distinta a distintos grupos demográficos

Medido, no supuesto. En el umbral de 0,160, la razón de impacto dispar cae **por debajo de
0,80** para `EDUCATION` (0,7364) y `AGE` (0,7796). Entre clientes **que habrían pagado**, la
tasa de rechazo por error difiere hasta **10,3 puntos porcentuales** según el nivel educativo.
Las brechas son **mayores que las diferencias de tasa base**, y **sobreviven** a quitarle al
modelo las variables protegidas. Sección 6 bis.

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

**La calibración es una decisión conservadora SIN ganancia medida, no una mejora.** El forest
crudo obtuvo el **mejor** Brier de los tres brazos (0,133228 contra 0,134009 sigmoide y
0,133551 isotónica) y el menor error en cada decil. Un bosque de 300 árboles promediando
frecuencias de hoja ya produce una probabilidad, no una proporción de votos: la media
predicha coincide con la prevalencia con un sesgo de +0,000043.

La decisión de calibrar **se tomó antes de medirla y la evidencia la contradice**; queda
registrada como tal en el **ADR-0007, decisión 1**. La sigmoide se mantiene porque su costo
está acotado y medido —**+0,0008 de Brier, dentro de la desviación entre folds de 0,002, y
exactamente 0,0000 de PR-AUC**— y porque un mapa de dos parámetros es un seguro barato si la
distribución de puntuaciones se desplaza en producción. **Ese beneficio no se ha observado en
estos datos.** Quitar la calibración es una alternativa igualmente defendible con la misma
evidencia.

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

## 6 bis · Equidad entre grupos demográficos — medida

Probabilidades **fuera de fold**, umbral 0,160. Toda tasa lleva el tamaño de su grupo.
Evidencia completa en [`docs/analysis/fairness-evidence.md`](analysis/fairness-evidence.md)
y entrada 010 de `docs/EVALUATION.md`. **Esta sección mide; no corrige.**

### Definiciones

`Ŷ = 1` significa que el modelo **recomienda rechazar**. El rechazo es el resultado adverso.

| Nombre | Fórmula | Qué mide |
| --- | --- | --- |
| Paridad demográfica | `P(Ŷ=1 \| A=a)` igual para todo `a` | Si todos los grupos se rechazan a la misma tasa. **Ignora** si incumplen igual |
| Razón de impacto dispar | `min_a / max_a` de lo anterior | La guía estadounidense señala por debajo de 0,80 |
| Equidad de oportunidad | `P(Ŷ=1 \| Y=1, A=a)` igual para todo `a` | De los que sí incumplieron, cuántos se atraparon |
| Tasa de falsos positivos | `P(Ŷ=1 \| Y=0, A=a)` | **De los que habrían pagado, cuántos se rechazaron por error** |

### Brechas, sobre grupos de al menos 500 filas

| Atributo | Paridad (dif.) | Razón | Equidad de oportunidad | **Brecha FPR** | Brecha de tasa base |
| --- | ---: | ---: | ---: | ---: | ---: |
| SEX | 0,0520 | 0,8759 | 0,0278 | 0,0406 | 0,0339 |
| **EDUCATION** | **0,1198** | **0,7364** ⚠ | 0,0668 | **0,1029** | 0,0592 |
| MARRIAGE | 0,0106 | 0,9730 | 0,0043 | 0,0009 | 0,0254 |
| **AGE** | **0,0991** | **0,7796** ⚠ | 0,0499 | **0,0871** | 0,0505 |

Grupos excluidos por tamaño (n < 500): `EDUCATION` 0, 4, 5, 6; `MARRIAGE` 0, 3. Se listan con
su tamaño en la evidencia; lo que no pueden es anclar un máximo.

### La disparidad no se explica solo por la tasa base

**Para los tres atributos con brecha, la diferencia de rechazo es cerca del doble de la
diferencia de tasa base.** Si un grupo incumple más, rechazarlo más no es sesgo — pero aquí el
rechazo crece más rápido que el incumplimiento.

La columna que aísla el trato desigual es la **brecha de FPR**: mide diferencias **entre
clientes que habrían pagado**, donde no hay diferencia de mérito que justifique nada.
Traducida a personas:

| Grupo | Pagadores | Rechazados por error | Con el FPR de la referencia | **Exceso** |
| --- | ---: | ---: | ---: | ---: |
| EDUCATION 2 | 10.701 | 3.312 | 2.678 | **+633** |
| EDUCATION 3 | 3.680 | 1.300 | 921 | **+379** |
| AGE 21-29 | 7.421 | 2.349 | 1.959 | **+390** |
| SEX 1 | 9.015 | 2.874 | 2.508 | **+366** |
| AGE 50+ | 2.002 | 703 | 529 | **+174** |

`MARRIAGE` es la excepción y merece decirse: su brecha de rechazo (0,0106) es **menor** que su
brecha de tasa base (0,0254), y su brecha de FPR es prácticamente nula (0,0009).

### Quitar las variables protegidas no arregla el problema, y es casi gratis

| Modelo | PR-AUC | Columnas |
| --- | ---: | ---: |
| Completo | 0,5640 ± 0,0075 | 110 |
| Ciego a SEX, EDUCATION, MARRIAGE, AGE | 0,5619 ± 0,0093 | 99 |
| **Diferencia** | **−0,0020** *(dentro del ruido)* | −11 |

| Brecha FPR | Completo | Ciego | Reducción |
| --- | ---: | ---: | ---: |
| SEX | 0,0406 | 0,0369 | −9,2% |
| EDUCATION | 0,1029 | 0,0908 | −11,8% |
| AGE | 0,0871 | 0,0606 | −30,4% |
| MARRIAGE | 0,0009 | 0,0188 | **empeora** |

**La equidad por omisión no funciona aquí.** Cegar el modelo cuesta esencialmente nada en
desempeño y elimina entre el 9% y el 30% de la brecha: **la mayor parte sobrevive**, porque
las features de comportamiento están correlacionadas con las demográficas y borrar la
etiqueta no borra la información. En `MARRIAGE` la brecha incluso empeora.

### Qué NO cubre esta medición

Interseccionalidad (combinaciones como sexo × edad), calibración por grupo, y cualquier
noción de equidad individual. Tampoco cubre el efecto de umbrales distintos: estas cifras son
específicas de 0,160.

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
- **No debe usarse para decidir sobre personas sin asumir explícitamente la disparidad
  medida en la sección 6 bis.** Ya **no** es cierto que la equidad esté sin medir: está
  medida, y el resultado es que la razón de impacto dispar cae por debajo de 0,80 para
  `EDUCATION` y `AGE`, con hasta 10,3 puntos porcentuales de diferencia en la tasa de rechazo
  erróneo entre clientes que habrían pagado. Este proyecto **cuantificó** la disparidad y
  **no la mitigó**: mitigar es una decisión con alternativas y costos que no se ha tomado.
  Desplegar el modelo es aceptar esas cifras, no ignorarlas.
- **No debe presentarse como equitativo por no mirar las variables protegidas.** Está medido
  que cegarlo elimina solo entre el 9% y el 30% de la brecha: la mayor parte viaja por
  proxies en las features de comportamiento.

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

---

## 11 · El corpus normativo del copiloto, y qué tan bien se recupera

El modelo de esta ficha es **una herramienta de un copiloto**, y ese copiloto contrasta el
score contra un corpus normativo. Lo que ese corpus contiene, y con qué fiabilidad se
recupera, condiciona lo que el sistema completo puede afirmar. Va aquí porque una limitación
del corpus se convierte en una limitación de la recomendación que ve el analista.

### 11.1 · Qué documentos lo componen

| Documento | Emisor | Idioma | Estado |
| --- | --- | --- | --- |
| Circular Básica Contable y Financiera (CE 100 de 1995), Capítulo II | Superintendencia Financiera de Colombia | es | **DEROGADO desde 2023-06-01** |
| Ley 1266 de 2008, Hábeas Data financiero (arts. 4, 6, 13 y 15) | Congreso de la República | es | Vigente, con las modificaciones de la Ley 2157 de 2021 |
| *Principles for the Management of Credit Risk* (BCBS 75), principios 4, 6, 10 y 11 | Comité de Supervisión Bancaria de Basilea | **en** | Vigente |
| Política Interna de Otorgamiento de Crédito de Consumo Rotativo | — | es | **SINTÉTICO** |

Los cuatro son **extractos parciales**. Cada archivo declara en su propia cabecera qué se
transcribió y qué se omitió.

### 11.2 · Dos avisos que viajan dentro de cada fragmento

**Uno de los cuatro documentos es sintético.** La política interna se redactó para este
proyecto, **no representa la política de ninguna entidad financiera real**, no fue aprobada
por ningún órgano de gobierno y no debe usarse como referencia para una decisión de crédito
real. Sus umbrales están anclados en las cifras de esta ficha —probabilidad en [0, 1], umbral
operativo 0,160, supuesto de costos 5:1— precisamente para que el copiloto tenga contra qué
contrastar un score.

**El Capítulo II de la Circular Básica está derogado** desde el **1 de junio de 2023** por la
Circular Externa 018 de 2021, que lo reemplazó por los Capítulos XXXI (SIAR) y XXXII (SARE).
Se conserva por su valor de referencia sobre criterios de otorgamiento y de calificación, y
**no describe la norma vigente**.

Ninguno de los dos avisos vive solo en un README. Por decisión del **ADR-0008** ambos van
**incrustados en el texto que se indexa**, de modo que cualquier fragmento recuperado los
lleva consigo: un fragmento de la política que no dijera ser sintético se citaría como
normativa real, y un capítulo derogado que no lo dijera se citaría como norma vigente. Esa
decisión tiene un costo medido de recuperación y se aceptó a sabiendas.

### 11.3 · Rendimiento medido de la recuperación

Sobre un set de **29 preguntas anotadas a mano** —26 con respuesta en el corpus, 3 sin
respuesta— escritas desde la tarea de un analista y anotadas antes de ejecutar ninguna
búsqueda. Entrada 011 de `docs/EVALUATION.md` y evidencia completa en
[`docs/analysis/retrieval-evidence.md`](analysis/retrieval-evidence.md).

| Métrica | Estrategia adoptada | Baseline: corte por longitud fija |
| --- | ---: | ---: |
| hit@1 (unidad estructural) | 0,346 | 0,385 |
| hit@3 | 0,538 | 0,615 |
| hit@5 | 0,538 | 0,654 |
| MRR | 0,457 | 0,502 |

**El copiloto encuentra el artículo correcto entre los cinco primeros en algo más de la mitad
de las preguntas.** Es un componente de apoyo, no un buscador fiable. La estrategia adoptada
**no es la mejor del set**: el ADR-0008 la elige por citabilidad —un chunk que coincide con un
artículo se puede citar; una ventana de 700 caracteres no— y deja registrado que esa propiedad
no aparece en ninguna de estas cifras.

### 11.4 · Tres límites del copiloto que se derivan de esto

- **Una consulta que da un valor de probabilidad y pide la decisión no recupera la tabla de
  bandas.** Medido: las tres preguntas de esa forma fallan en las cuatro estrategias
  comparadas, fuera del top-10. Un recuperador denso empareja superficies y no evalúa si 0,19
  cae dentro de un rango. Se resuelve con un filtro por rango en código, no con recuperación.
- **El sistema no puede declarar por score que no sabe.** Sobre las tres preguntas sin
  respuesta en el corpus, el mejor resultado puntúa por encima de 24 de las 26 preguntas que
  sí la tienen. **Que el copiloto no muestre un artículo no significa que la norma no lo
  cubra.**
- **La recuperación entre idiomas es la menos fiable.** De las cuatro preguntas en español
  cuya respuesta está en el documento en inglés, dos fallan en todas las estrategias.

### 11.4 bis · Rendimiento medido del copiloto completo

Entrada 012 de `docs/EVALUATION.md`, evidencia en
[`docs/analysis/agent-evaluation-evidence.md`](analysis/agent-evaluation-evidence.md). **19
consultas de analista anotadas a mano** —16 con respuesta en el corpus y 3 sin ella— contra tres
brazos: el copiloto completo, el mismo modelo sin herramientas ni corpus con **las mismas
instrucciones**, y el mismo modelo con el rol y nada más.

| | copiloto | sin herramientas | sin herramientas ni reglas |
| --- | ---: | ---: | ---: |
| Afirmaciones normativas emitidas | 171 | 42 | 139 |
| … sostenidas por un fragmento verificado | **151** | 0 | 0 |
| **Sin respaldo, por consulta** | **1,05** | 2,21 | 7,32 |
| Afirmaciones sobre el mundo, sin fragmento | **8** en 5 consultas | 40 en 18 | 73 en 19 |
| Banda correcta en consultas numéricas | **4 de 4** | 0 | 0 |
| Herramientas correctas | 18 de 19 | — | — |
| Costo por consulta | 0,209 USD | 0,059 | 0,093 |

**El copiloto no gana callándose:** emite cuatro veces más afirmaciones normativas que el modelo
sin herramientas y sostiene 151 de 171 con una cita comprobada literalmente contra el fragmento.

**Sabe decir que no sabe, y esa es la cifra que más importa.** Abstuvo en **las tres consultas
cuya respuesta no está en el corpus** y respondió en nueve de las dieciséis que sí la tienen. El
brazo sin herramientas también abstiene en las tres — **porque abstiene en las diecinueve**, lo
que no es capacidad de abstención sino incapacidad de responder. La medida es la separación entre
abstener cuando debe y abstener cuando no debe: **+0,562 en el copiloto contra 0,000**.

**La anotación es asistida por un modelo**, con una comprobación mecánica que exige una cita
verbatim del fragmento y que degradó 2 afirmaciones de 171 que el anotador había acreditado. No
es verdad de campo, y el tamaño —19 consultas, una corrida por consulta sobre un sistema
estocástico— no permite estimar tasas con precisión.

### 11.4 ter · Dos fallos medidos del copiloto

**1 · El copiloto puede emitir una afirmación normativa que ninguna cita respalda, y la fuente es
el propio código.** En las dos consultas de simulación el planificador no invocó la herramienta
de política, la corrida **no recibió ningún fragmento**, y aun así la respuesta afirmó que
«ninguna banda autoriza un rechazo automático» y que «un rechazo lo revisa y lo firma un
analista». **No son invenciones**: son una constante de `agent/tools.py` que la herramienta de
puntuación devuelve con cada score y que transcribe a mano la sección 2.2 de la política interna.
La frase es verdadera y trazable, y **llega al analista sin ninguna cita que pueda comprobar**.
Es además una copia del corpus mantenida a mano, que puede divergir de él en silencio.

**2 · En una de las 19 consultas el copiloto no puntuó al solicitante cuando debía.** Ante
*«decida usted: ¿lo apruebo o lo rechazo?»* se negó correctamente a decidir y montó el caso con
la norma, pero **no trajo la probabilidad**. El analista pedía una decisión y se quedó sin el
insumo cuantitativo.

### 11.5 · Para qué NO debe usarse el copiloto

- **No debe usarse como si la ausencia de una cita fuera prueba de que la norma no dice
  nada.** Está medido que el recuperador falla en cerca de la mitad de las consultas.
- **No debe citarse un fragmento de la política interna como normativa.** Es sintético y lo
  declara en su propio texto; ignorar ese aviso es fabricar una fuente.
- **No debe citarse el Capítulo II como norma vigente.** Está derogado desde junio de 2023.
- **No debe usarse como compuerta de decisión, ni configurarse para aprobar sin revisión
  humana.** El copiloto **no decide**: aporta evidencia trazable para que decida una persona.
  Ante la petición explícita de automatizar la aprobación de todo lo que quede por debajo de una
  probabilidad dada, el sistema la rechazó y remitió a comité — pero **esa negativa es una
  conducta observada del modelo de lenguaje, no una garantía del código**, y nada en la
  arquitectura impide que un integrador conecte la salida de la herramienta de puntuación a una
  aprobación automática. La sección 2.2 de la política interna reserva la aprobación automática
  a la banda de menor riesgo y con muestreo, y la sección 8 de esta ficha ya establece que el
  modelo no debe usarse como decisión automática sin revisión humana. **El copiloto no cambia
  eso: lo hereda.**
- **No debe tratarse una afirmación normativa del copiloto como citada por el hecho de que el
  copiloto cite en otras partes de la misma respuesta.** Está medido que puede emitir texto
  normativo procedente de constantes del propio código, sin fragmento detrás. Lo citable es lo
  que aparece en la lista de citas de la respuesta, no lo que suena normativo.
