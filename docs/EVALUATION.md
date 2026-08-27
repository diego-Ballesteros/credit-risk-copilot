# Registro de evaluación

Una entrada por medición realizada, dejando explícito **qué se midió, con qué
protocolo, contra qué baseline y con qué resultado**: una métrica sin el baseline al
lado y sin el procedimiento que la produjo no es evidencia de nada, porque no se
puede saber si el número es bueno ni se puede reproducir.

## Métricas fijadas

Las fija el **ADR-0002** y no se eligen por entrada. Toda evaluación reporta todas.

| Rol | Métrica |
| --- | --- |
| **Métrica de decisión** | **PR-AUC** (área bajo la curva precisión-recall) |
| Contexto | ROC-AUC |
| Contexto | KS |
| Contexto | Gini |
| Contexto | Brier score (calibración) |
| Contexto | **precision@top-10%** (principal) |
| Contexto | **precision@top-5%** (secundaria) |

Las comparaciones entre modelos se deciden **solo** por PR-AUC. Las métricas de
contexto describen el resultado; no lo eligen.

## Formato de entrada

    ### NNN — Qué se evaluó

    - **Fecha:** AAAA-MM-DD
    - **Fase:** NN-nombre-de-la-fase
    - **Objeto evaluado:** modelo, pipeline o componente, con su versión o commit.
    - **Datos:** dataset, partición (train/valid/test), número de filas y periodo.
    - **Protocolo:** esquema de validación, semilla, y cómo se evitó la fuga de
      información entre particiones.
    - **Métricas:** tabla con las siete métricas fijadas arriba. Cada fila lleva el
      valor obtenido **y su baseline al lado, en la misma fila**, más la desviación
      estándar entre folds cuando aplique.
    - **Baseline — campo obligatorio:** contra qué se compara, por qué es el baseline
      correcto para esta medición, y su valor en las mismas métricas y sobre los mismos
      datos. **Una entrada sin este campo está incompleta y no cuenta como evidencia.**
      Cuando no exista baseline previo, se declara el baseline trivial (clase mayoritaria
      o azar estratificado) y se reporta su valor, nunca "no aplica".
    - **Resultado:** la diferencia en PR-AUC respecto al baseline y si supera el umbral
      de decisión fijado de antemano.
    - **Reproducción:** el comando exacto y el run de MLflow que regeneran el número.
    - **Interpretación:** qué significa el resultado para el problema de negocio y qué
      limitaciones tiene.

Las entradas se numeran de forma consecutiva. Un resultado nunca se edita para
mejorarlo: si una medición resulta inválida, se añade una entrada nueva que la
invalida y explica por qué.

---

<!-- Las entradas van debajo de esta línea, de la más antigua a la más reciente. -->

### 001 — Baselines de la Fase 2: el piso del proyecto

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Objeto evaluado:** tres estimadores sobre el `Pipeline` completo de
  `build_preprocessor`, en la rama `feature/02-modeling`. El preprocesador se ajusta
  **dentro** de cada fold; `data/processed/features.parquet` no se leyó en ningún momento.
- **Datos:** UCI 350 vía `loader.load_dataset`. 30.000 filas, 23 predictores y el target.
  Sin partición train/test: toda la medición es validación cruzada sobre el dataset
  completo. El dataset no tiene fecha de originación, así que **no se puede medir
  generalización fuera de tiempo**; no se afirma nada sobre ella.
- **Protocolo:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, semilla
  tomada de `config.py`. En cada fold se construye un preprocesador nuevo y se ajusta solo
  con las filas de entrenamiento de ese fold, de modo que ningún estadístico aprendido ve
  una fila de validación. Prevalencia de la clase positiva: **0,221200**.

**Métricas — media ± desviación estándar entre los 5 folds**

| Métrica | Piso sin señal | Trivial (clase mayoritaria) | Azar estratificado | Regresión logística L2 |
| --- | --- | --- | --- | --- |
| **PR-AUC** (decisión) | 0,2212 | 0,221200 ± 0,000075 | 0,220047 ± 0,001880 | **0,540173 ± 0,010295** |
| ROC-AUC | 0,5000 | 0,500000 ± 0,000000 | 0,496372 ± 0,005752 | 0,776187 ± 0,007162 |
| KS | 0,0000 | 0,000000 ± 0,000000 | 0,001720 ± 0,002569 | 0,421780 ± 0,018022 |
| Gini | 0,0000 | 0,000000 ± 0,000000 | −0,007255 ± 0,011505 | 0,552373 ± 0,014324 |
| Brier (menor es mejor) | 0,1723 | 0,221200 ± 0,000075 | 0,346000 ± 0,003965 | 0,183357 ± 0,001565 |
| precision@top-10% | 0,2212 | 0,221200 ± 0,000075 | 0,215502 ± 0,009033 | 0,692667 ± 0,010904 |
| precision@top-5% | 0,2212 | 0,221200 ± 0,000075 | 0,215502 ± 0,009033 | 0,742667 ± 0,016566 |
| *accuracy (no decide)* | *0,7788* | *0,7788* | *0,6540* | *0,7565* |

- **Baseline:** el **clasificador trivial de clase mayoritaria**, medido sobre exactamente
  los mismos folds y la misma semilla. Es el baseline correcto porque fija el piso absoluto:
  cualquier métrica que no lo supere describe un modelo que no aporta ordenamiento. Se
  añade el **azar estratificado** como segundo baseline porque el trivial no ordena nada —
  da una sola probabilidad a todas las filas — y por lo tanto no puede fijar el piso de una
  métrica de ordenamiento; el azar estratificado sí ordena, sin información.
- **Resultado:** la regresión logística supera al baseline trivial en PR-AUC por
  **+0,3190** (0,540173 contra 0,221200), con una desviación entre folds de 0,0103. La
  diferencia es unas 31 veces esa desviación, así que no es ruido de partición.
- **Reproducción:** `uv run python scripts/run_baselines.py`. Runs de MLflow en el
  experimento `credit-risk-baselines` (id 0): `a89e11251e5145fa990cb53ab8604d10`
  (trivial), `60e590076f4045ad8ef2bf55afa18026` (azar), `7055d04bc1334d8188ecb801c87060fc`
  (logística). Cada run lleva la tabla por fold como artefacto `fold_metrics.csv`.
- **Interpretación:**
  - **El argumento del ADR-0002 queda medido, no citado.** El clasificador trivial gana en
    accuracy (0,7788) contra la logística (0,7565) y no identifica ni un solo incumplidor:
    su PR-AUC está exactamente sobre el piso. Un criterio basado en accuracy habría elegido
    el modelo inútil.
  - **La logística está lejos del piso.** precision@top-10% de 0,6927 contra un piso de
    0,2212 significa que revisar el decil peor puntuado encuentra 3,1 veces más
    incumplidores que revisar un decil al azar. Ese es el número con lectura de negocio.
  - **El Brier del trivial es peor que el del piso** (0,2212 contra 0,1723) y esto no es
    una contradicción: el clasificador de clase mayoritaria predice probabilidad 0, no la
    prevalencia. Predecir constantemente 0,2212 daría 0,1723. Ordena igual de mal en ambos
    casos; calibra mucho peor prediciendo 0.
  - **Límite explícito:** estas cifras son de validación cruzada sobre todo el dataset. No
    hay conjunto de test retenido, así que no son una estimación de desempeño en
    producción, y no hay eje temporal con el que medir deriva.

### 002 — Prueba del target barajado: verificación de ausencia de leakage

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Objeto evaluado:** el **mismo** `Pipeline` de la entrada 001 —preprocesador más
  `LogisticRegression(l2, class_weight="balanced")`— entrenado contra un target permutado.
  No es la evaluación de un modelo: es un diagnóstico del pipeline.
- **Datos:** los mismos 30.000 registros. El target se permuta con `random_state=42`;
  10.370 de 30.000 etiquetas cambian de fila y la prevalencia queda intacta en 0,221200,
  porque una permutación mueve etiquetas y no crea ni destruye ninguna.
- **Protocolo:** idéntico al de la entrada 001, sin ninguna variación. Esa identidad es la
  condición de validez de la prueba: si el pipeline bajo test no fuera el mismo objeto, el
  resultado no diría nada sobre el pipeline real.

**Métricas — media ± desviación estándar entre los 5 folds**

| Métrica | Valor con target permutado | Piso sin señal (baseline) | Desviación |
| --- | --- | --- | --- |
| **PR-AUC** | 0,214976 ± 0,003514 | 0,221200 | −0,006224 |
| ROC-AUC | 0,488232 ± 0,007280 | 0,500000 | −0,011768 |
| KS | 0,008602 ± 0,005490 | 0,000000 | +0,008602 |
| Gini | −0,023537 ± 0,014560 | 0,000000 | −0,023537 |
| Brier | 0,250725 ± 0,000525 | 0,172270 | +0,078455 |
| precision@top-10% | 0,201667 ± 0,021311 | 0,221200 | −0,019533 |
| precision@top-5% | 0,209333 ± 0,024766 | 0,221200 | −0,011867 |

- **Baseline:** el **piso sin señal**, que no es el mismo número para cada métrica: la
  prevalencia (0,2212) para PR-AUC y las dos de precisión en el tope, 0,5 para ROC-AUC y 0
  para KS y Gini. Es el baseline correcto porque la hipótesis bajo prueba es precisamente
  que el desempeño colapsa hasta él.
- **Criterios de aprobación, fijados antes de calcular el resultado:**
  1. `|media ROC-AUC − 0,5| ≤ 0,020`
  2. `|media PR-AUC − 0,2212| ≤ 0,015`

  Ambos umbrales están **medidos, no elegidos**: se corrió el pipeline completo contra
  **ocho permutaciones distintas** del target, obteniendo para la media de 5 folds
  ROC-AUC 0,502030 con desviación 0,00593 y PR-AUC 0,223199 con desviación 0,00335. Los
  umbrales son 3,4 y 4,5 errores estándar medidos respectivamente, y la peor desviación
  observada en esas ocho permutaciones fue 0,0117 y 0,0082 — dentro de ambos.
- **Resultado:** **APROBADA**. ROC-AUC 0,488232 (desviación −0,011768, tolerancia ±0,020) y
  PR-AUC 0,214976 (desviación −0,006224, tolerancia ±0,015). El script terminó con código
  de salida 0.
- **Reproducción:** `uv run python scripts/run_leakage_check.py`. Run de MLflow
  `969a18ba369a402aab0f9000851044c4`, etiquetado `run_type=leakage-check-shuffled-target`
  e `is_real_result=false` para que no se confunda con un resultado. **Filtrar por esa
  etiqueta antes de comparar modelos.**

  > **Nota de estado, 2026-08-26.** El run original de esta entrada era
  > `d90d6c32398b4d839201d019794c2000`. Al migrar el estimador de `penalty="l2"` a
  > `l1_ratio=0` se volvió a correr la prueba para verificar que la garantía seguía en
  > pie, y el run nuevo reemplazó al anterior, que se eliminó del experimento para no
  > dejar dos diagnósticos idénticos. **Ninguna cifra de esta entrada cambió**: la
  > equivalencia de las dos parametrizaciones se verificó bit a bit antes de migrar y el
  > resultado se reprodujo exacto. Lo único que cambió es el identificador del run.
- **Interpretación:**
  - El desempeño colapsó. Con el target destruido, las siete métricas quedan dentro del
    ruido de su piso, mientras el mismo pipeline con el target real alcanza PR-AUC 0,5402 y
    ROC-AUC 0,7762. La distancia entre las dos situaciones es de 95 y 47 errores estándar.
  - **Las dos desviaciones cayeron del mismo lado, por debajo del piso** (−1,98 y −1,86
    errores estándar medidos respecto del piso; −2,33 y −2,45 respecto del centro empírico
    de la nula). Se verificó si eso era un sesgo sistemático de la validación
    cruzada bajo la hipótesis nula corriendo ocho permutaciones: el centro quedó en +0,0020
    sobre el piso en ambas métricas, es decir **por encima**, no por debajo. La coincidencia
    de signo en la semilla 42 es ruido de esa permutación concreta y no un efecto del
    protocolo.
  - Lo que la prueba **no** dice: que el modelo sea bueno, ni que no exista ninguna otra
    forma de fuga. Descarta que información de una fila de validación llegue al objeto que
    la transforma, que es la fuga que el diseño del pipeline existe para impedir.

### 003 — Contraste de la hipótesis principal: comportamiento contra demografía

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Objeto evaluado:** tres brazos que comparten estimador, preprocesador, particionador y
  semilla, y **difieren únicamente en el subconjunto de columnas que llega al modelo**. El
  estimador sale de `build_logistic_regression()`, que no acepta parámetros justamente para
  que dos brazos no puedan divergir. La selección ocurre sobre la *salida* del
  preprocesador, no sobre la tabla cruda.
- **Datos:** UCI 350 vía `loader.load_dataset`. 30.000 filas, prevalencia 0,221200. Sin
  partición train/test: validación cruzada sobre el dataset completo.
- **Protocolo:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, semilla de
  `config.py`. Un preprocesador nuevo por fold, ajustado solo con las filas de
  entrenamiento de ese fold. El reparto de columnas se verifica en cada `fit`: las 110
  columnas de la matriz se reparten en **12 demográficas y 98 de comportamiento**,
  disjuntas y cubriendo el total.

**Métricas — media ± desviación estándar entre los 5 folds**

| Métrica | Trivial (baseline) | Solo demografía | Solo comportamiento | Todas |
| --- | --- | --- | --- | --- |
| **PR-AUC** (decisión) | 0,2212 ± 0,0001 | 0,3056 ± 0,0064 | **0,5362 ± 0,0132** | 0,5402 ± 0,0103 |
| ROC-AUC | 0,5000 ± 0,0000 | 0,6242 ± 0,0043 | 0,7728 ± 0,0094 | 0,7762 ± 0,0072 |
| KS | 0,0000 ± 0,0000 | 0,1968 ± 0,0089 | 0,4212 ± 0,0166 | 0,4218 ± 0,0180 |
| Gini | 0,0000 ± 0,0000 | 0,2485 ± 0,0085 | 0,5457 ± 0,0187 | 0,5524 ± 0,0143 |
| Brier (menor es mejor) | 0,2212 ± 0,0001 | 0,2395 ± 0,0012 | 0,1841 ± 0,0021 | 0,1834 ± 0,0016 |
| precision@top-10% | 0,2212 ± 0,0001 | 0,3508 ± 0,0178 | 0,6927 ± 0,0146 | 0,6927 ± 0,0109 |
| precision@top-5% | 0,2212 ± 0,0001 | 0,3641 ± 0,0171 | 0,7433 ± 0,0178 | 0,7427 ± 0,0166 |
| *Columnas que ve el modelo* | *110* | *12* | *98* | *110* |

- **Baseline:** el **clasificador trivial de clase mayoritaria**, medido sobre exactamente
  los mismos folds y la misma semilla dentro de este mismo script, para que la comparación
  no dependa de una corrida anterior. Es el baseline correcto porque los tres brazos son
  modelos de ordenamiento y el trivial fija el piso que ninguno de ellos puede no superar
  sin ser inútil. Su run en MLflow es el de la entrada 001; aquí se remide y no se vuelve a
  registrar, para no duplicarlo.

**Distancia al baseline trivial, en PR-AUC**

| Brazo | PR-AUC | Desviación entre folds | Ventaja sobre el piso | Ventaja / desviación |
| --- | ---: | ---: | ---: | ---: |
| Solo demografía | 0,3056 | 0,0064 | +0,0844 | 13,2 |
| Solo comportamiento | 0,5362 | 0,0132 | +0,3150 | 23,9 |
| Todas | 0,5402 | 0,0103 | +0,3190 | 31,0 |

**Incrementos, pareados fold a fold**

Pareado porque los tres brazos comparten partición: comparar las diferencias por fold
elimina la variación que viene del corte y deja la que viene de las columnas.

| Incremento | Media | Desv. de las diferencias | Error estándar | t pareado | Folds con signo positivo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Comportamiento sobre demografía | **+0,2306** | 0,0117 | 0,0052 | **44,1** | 5 de 5 |
| Todas sobre demografía | +0,2346 | 0,0100 | 0,0045 | 52,2 | 5 de 5 |
| **Demografía sobre comportamiento** | **+0,0040** | 0,0034 | 0,0015 | **2,66** | 4 de 5 |

- **Resultado:** la hipótesis principal **se sostiene, y con holgura**. El comportamiento de
  pago supera a la demografía en PR-AUC por **+0,2306**, unas 44 veces el error estándar de
  la diferencia pareada, con las cinco diferencias por fold del mismo signo. El camino
  inverso es de otro orden: añadir demografía a un modelo que ya ve comportamiento aporta
  **+0,0040**, que es 0,3 desviaciones entre folds en la comparación no pareada.
- **Reproducción:** `uv run python scripts/run_hypothesis_contrast.py`. Runs de MLflow en
  el experimento `credit-risk-baselines` (id 0), etiquetados `run_type=hypothesis-contrast`:
  `bb5dcba3001e4320a6b39a70321c758e` (demografía), `b47cd4c5eeff446e827eb94d1d59e0c3`
  (comportamiento), `ed0646b0eb7a4b01aa562d61cf54c1f1` (todas).
- **Interpretación:**
  - **La demografía no es inútil, es insuficiente.** Con 12 columnas alcanza PR-AUC 0,3056
    contra un piso de 0,2212: 13 desviaciones por encima del azar, y precision@top-10% de
    0,3508 contra 0,2212. Ordena de verdad. Lo que no hace es competir: el comportamiento
    saca más del triple de ventaja sobre el mismo piso.
  - **El brazo "todas" reproduce exactamente el baseline logístico de la entrada 001**
    —0,540173 y 0,776187, fold por fold— lo cual era la comprobación de que el selector con
    el conjunto completo no altera nada y de que los tres brazos son comparables.
  - **El incremento de la demografía merece una lectura cuidadosa y no un titular.** No
    pareado cae dentro del ruido entre folds (0,3 desviaciones). Pareado da t = 2,66 con
    cuatro de cinco folds positivos, lo que sugiere un efecto pequeño y real más que ruido
    puro. Las dos lecturas no se contradicen: el efecto es **consistente en signo y
    minúsculo en magnitud**. Además, los folds de una validación cruzada no son
    independientes, así que ese t es anticonservador y no debe leerse como un valor p.
    La conclusión defendible es que la demografía aporta, sobre el comportamiento, **algo
    del orden de 0,004 de PR-AUC**, que es una centésima parte de lo que aporta el
    comportamiento sobre la demografía.
  - **Límite del contraste, explícito.** Los dos grupos reparten las *columnas* pero no
    reparten del todo la *información*. `UTILIZATION_*` es un saldo dividido por
    `LIMIT_BAL`, y `LIMIT_BAL` está contado como demográfico mientras las utilizaciones
    están contadas como comportamiento; como el brazo de comportamiento también tiene los
    saldos, el cupo es en principio recuperable de él. Un modelo lineal no puede hacer esa
    división, así que el efecto práctico es acotado, pero el contraste mide **estas columnas
    contra aquellas**, no demografía contra comportamiento como conceptos separables.
  - **Límite de alcance.** Todo esto es validación cruzada sobre el dataset completo, con un
    único estimador lineal. No se afirma que el orden se conserve con modelos no lineales,
    ni fuera de esta población.

### 004 — Comparación de modelos: ¿supera algo a la logística?

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Objeto evaluado:** tres modelos con hiperparámetros por defecto documentados en
  `models/estimators`, sobre el mismo `Pipeline`, la misma partición y la misma semilla.
  Ninguno está tuneado: el tuning es la entrada 006.
- **Datos:** UCI 350 vía `loader.load_dataset`. 30.000 filas, prevalencia 0,221200.
- **Protocolo:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, semilla de
  `config.py`. Preprocesador ajustado dentro de cada fold.
- **Umbral de decisión, fijado ANTES de ver los resultados:** una diferencia menor a
  **0,02 en PR-AUC** no es distinguible del ruido de partición con 5 folds, porque la
  desviación entre folds es del orden de 0,010. Vive en
  `evaluation.PRACTICAL_SIGNIFICANCE_THRESHOLD`, en un solo sitio.

**Métricas — media ± desviación estándar entre los 5 folds**

| Métrica | Trivial (piso) | Logística L2 (referencia) | Random Forest | Hist Gradient Boosting |
| --- | --- | --- | --- | --- |
| **PR-AUC** (decisión) | 0,2212 ± 0,0001 | 0,5402 ± 0,0103 | **0,5605 ± 0,0096** | 0,5565 ± 0,0078 |
| ROC-AUC | 0,5000 ± 0,0000 | 0,7762 ± 0,0072 | 0,7873 ± 0,0077 | 0,7810 ± 0,0070 |
| KS | 0,0000 ± 0,0000 | 0,4218 ± 0,0180 | 0,4379 ± 0,0148 | 0,4315 ± 0,0153 |
| Gini | 0,0000 ± 0,0000 | 0,5524 ± 0,0143 | 0,5747 ± 0,0153 | 0,5621 ± 0,0140 |
| Brier (menor es mejor) | 0,2212 ± 0,0001 | 0,1834 ± 0,0016 | 0,1740 ± 0,0020 | **0,1692 ± 0,0021** |
| precision@top-10% | 0,2212 ± 0,0001 | 0,6927 ± 0,0109 | 0,6953 ± 0,0214 | 0,7017 ± 0,0135 |
| precision@top-5% | 0,2212 ± 0,0001 | 0,7427 ± 0,0166 | 0,7620 ± 0,0227 | 0,7600 ± 0,0312 |

- **Baseline:** doble. El **clasificador trivial** sigue siendo el piso absoluto (entrada
  001), y la **regresión logística L2** es la referencia de este turno: es el modelo más
  simple que ya funciona, así que es contra ella y no contra el piso como se decide si la
  complejidad adicional se paga.
- **Resultado, contra la logística y al umbral de 0,02:**

| Modelo | PR-AUC | Desv. entre folds | Diferencia | ¿Supera el umbral? |
| --- | ---: | ---: | ---: | --- |
| Random Forest | 0,5605 | 0,0096 | **+0,0203** | **Sí, por 0,0003** |
| Hist Gradient Boosting | 0,5565 | 0,0078 | +0,0163 | No — dentro del ruido |

- **Reproducción:** `uv run python scripts/run_model_comparison.py`. Runs de MLflow
  etiquetados `run_type=model-comparison`: `8f8e4decef22422aa37df69ba08dc695` (logística),
  `d34ebec22a7743289e4c134de5ef950f` (random forest),
  `60d744bc27554fc9af14cf14d88d6e48` (hist gradient boosting).
- **Interpretación:**
  - **El random forest supera el umbral por 0,0003.** Es el resultado literal y hay que
    leerlo como lo que es: está **en el borde**, no cómodamente por encima. Una partición
    distinta podría dejarlo del otro lado. La lectura defendible no es "el random forest
    es mejor" sino "es el único cuya ventaja alcanza justo el mínimo que este protocolo
    puede resolver, y por el margen más estrecho posible".
  - **La ganancia real es pequeña en términos absolutos.** +0,0203 de PR-AUC sobre 0,5402
    es una mejora relativa del 3,8%, a cambio de pasar de un modelo lineal con coeficientes
    legibles a un conjunto de 300 árboles.
  - **Donde los árboles ganan sin ambigüedad es en calibración.** Brier baja de 0,1834 a
    0,1740 (random forest) y a 0,1692 (boosting). El boosting es el mejor calibrado de los
    tres y **no** el mejor en PR-AUC, que es exactamente la disociación por la que el
    ADR-0002 reporta ambas.
  - **precision@top-10% apenas se mueve:** 0,6927 → 0,6953 → 0,7017. En el decil que el
    negocio realmente revisaría, los tres modelos encuentran prácticamente los mismos
    incumplidores.
  - **Límite explícito:** el tercer modelo debía ser LightGBM y no se pudo usar; ver el
    reporte del turno. `HistGradientBoostingClassifier` es de la misma familia pero **no es
    LightGBM**, y esta tabla no dice nada sobre LightGBM.

### 005 — Estrategias de desbalance: ¿ayuda SMOTE?

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Objeto evaluado:** el random forest de la entrada 004, con los mismos hiperparámetros,
  bajo tres tratamientos del desbalance. Lo único que cambia es el tratamiento.
- **Datos y protocolo:** idénticos a la entrada 004.
- **Cómo se impidió que SMOTE tocara un fold de validación:** el remuestreador vive dentro
  de un `imblearn.pipeline.Pipeline`, que llama a `fit_resample` en `fit` y **salta** los
  remuestreadores en `predict`. Verificado, no supuesto: una subclase espía registró
  exactamente **5 llamadas a `fit_resample`, cada una sobre 24.000 filas** —el fold de
  entrenamiento— y ninguna sobre las 30.000.

**Métricas — media ± desviación estándar entre los 5 folds**

| Métrica | Sin tratamiento (referencia) | class_weight balanceado | SMOTE |
| --- | --- | --- | --- |
| **PR-AUC** (decisión) | **0,5614 ± 0,0104** | 0,5605 ± 0,0096 | 0,5557 ± 0,0108 |
| ROC-AUC | 0,7872 ± 0,0078 | 0,7873 ± 0,0077 | 0,7835 ± 0,0074 |
| KS | 0,4385 ± 0,0182 | 0,4379 ± 0,0148 | 0,4302 ± 0,0119 |
| Gini | 0,5745 ± 0,0155 | 0,5747 ± 0,0153 | 0,5670 ± 0,0148 |
| **Brier** (menor es mejor) | **0,1336 ± 0,0020** | 0,1740 ± 0,0020 | 0,1576 ± 0,0023 |
| precision@top-10% | 0,6983 ± 0,0222 | 0,6953 ± 0,0214 | 0,6963 ± 0,0191 |
| precision@top-5% | 0,7640 ± 0,0172 | 0,7620 ± 0,0227 | 0,7620 ± 0,0217 |

- **Baseline:** el **modelo sin tratamiento**, medido sobre los mismos folds. Es el
  baseline correcto porque la pregunta es si *añadir* un tratamiento aporta algo; el piso
  trivial (PR-AUC 0,2212) sigue siendo el de la entrada 001.
- **Resultado:** **ninguna de las dos estrategias mejora nada, y ambas empeoran la
  calibración.**

| Estrategia | Δ PR-AUC | ¿Supera 0,02? | Δ Brier | Efecto en calibración |
| --- | ---: | --- | ---: | --- |
| class_weight balanceado | −0,0009 | No | **+0,0404** | **Peor** |
| SMOTE | −0,0058 | No | **+0,0239** | **Peor** |

- **Reproducción:** `uv run python scripts/run_imbalance_comparison.py`. Runs etiquetados
  `run_type=imbalance-comparison`: `c1148ce4339a49e9ac0a70ed0484e7fd` (sin tratamiento),
  `fef0f2218ab64f1e97f56e848221cb7b` (class_weight), `dddd752db36f42059b24f040bf17026b`
  (SMOTE).
- **Interpretación:**
  - **SMOTE no ayuda: no hace nada al ordenamiento y estropea la probabilidad.** Baja
    PR-AUC en 0,0058 —dentro del ruido, así que la lectura honesta es "no mejora", no
    "empeora el ordenamiento"— y sube Brier en 0,0239, que **sí** es un efecto grande al
    lado de una desviación entre folds de 0,002.
  - **`class_weight="balanced"` es el que más daña la calibración**, +0,0404 de Brier, sin
    comprar nada en PR-AUC (−0,0009). El mecanismo es directo: reponderar la clase positiva
    hace que el modelo prediga probabilidades más altas que la tasa real de la población,
    y el Brier mide exactamente esa distancia.
  - **Consecuencia para los turnos anteriores.** Toda la fase venía usando
    `class_weight="balanced"` por defecto, incluidos los baselines de la entrada 001 y el
    contraste de la 003. Esta medición dice que esa elección **no compró ordenamiento y
    costó calibración**. No invalida ninguna comparación anterior —todas eran internamente
    consistentes, con la misma configuración en todos los brazos— pero sí significa que el
    modelo que se lleve a producción no debería llevar `class_weight`.
  - **Con 22% de positivos, el desbalance no es el problema.** Estas técnicas se diseñaron
    para prevalencias de 1% o menos. A 22% hay 6.636 casos positivos, de sobra para que un
    modelo aprenda la clase minoritaria sin ayuda.
  - **Límite explícito sobre SMOTE en esta matriz.** SMOTE interpola entre filas vecinas, y
    74 de las 110 columnas son indicadores one-hot. Interpolarlos produce valores
    fraccionarios como 0,37 en una columna que significa "el nivel educativo es
    universitario", que no corresponde a ningún cliente que pueda existir. `SMOTENC` es la
    variante hecha para datos mixtos y **no se usó**, porque el turno pedía SMOTE. Parte del
    resultado negativo puede deberse a esto y no a SMOTE en general.

### 006 — Tuning con Optuna en validación cruzada anidada

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Objeto evaluado:** `RandomForestClassifier` con `class_weight=None` —la configuración
  que ganó la entrada 005— con los hiperparámetros estructurales buscados por Optuna.
- **Datos:** UCI 350 vía `loader.load_dataset`. 30.000 filas, prevalencia 0,221200.
- **Protocolo — anidado, y por qué:** el bucle **externo** es
  `StratifiedKFold(5, shuffle=True, random_state=42)` y **solo puntúa**; dentro de cada
  fold externo, un estudio de Optuna con `StratifiedKFold(3)` sobre las filas de
  entrenamiento de ese fold **solo elige**. Ningún trial ve jamás el fold externo con el
  que se le puntúa. Sin anidar, se reportaría el máximo de 30 extracciones ruidosas como
  si fuera el desempeño esperado de un modelo, con un sesgo optimista que después nadie
  puede acotar.
- **Espacio de búsqueda y presupuesto:** `n_estimators` ∈ {200,300,400,500}, `max_depth` ∈
  {6,8,…,24}, `min_samples_leaf` ∈ [1,80] en escala logarítmica, `max_features` ∈
  {sqrt, log2, 0,3}. **30 trials** por estudio con muestreador TPE sembrado desde
  `config.py`. Coste: 5 × 30 × 3 = **450 ajustes** anidados más 90 del estudio final.

**Métricas — media ± desviación estándar entre los 5 folds externos**

| Métrica | Logística (referencia global) | Forest sin tunear (referencia) | **Forest tuneado, CV anidada** |
| --- | --- | --- | --- |
| **PR-AUC** (decisión) | 0,5402 ± 0,0103 | 0,5614 ± 0,0104 | **0,5642 ± 0,0080** |
| ROC-AUC | 0,7762 ± 0,0072 | 0,7872 ± 0,0078 | 0,7863 ± 0,0089 |
| KS | 0,4218 ± 0,0180 | 0,4385 ± 0,0182 | 0,4392 ± 0,0191 |
| Gini | 0,5524 ± 0,0143 | 0,5745 ± 0,0155 | 0,5726 ± 0,0177 |
| Brier (menor es mejor) | 0,1834 ± 0,0016 | 0,1336 ± 0,0020 | **0,1334 ± 0,0021** |
| precision@top-10% | 0,6927 ± 0,0109 | 0,6983 ± 0,0222 | 0,7063 ± 0,0176 |
| precision@top-5% | 0,7427 ± 0,0166 | 0,7640 ± 0,0172 | 0,7687 ± 0,0090 |

- **Baseline:** el **mismo forest sin tunear**, con `class_weight=None`, medido sobre la
  misma partición dentro de este mismo script. Es el baseline correcto porque la pregunta
  es qué añade el tuning, no qué añade el forest.
- **Las tres estimaciones, y cuál es honesta:**

| Estimación | PR-AUC | Desv. | Qué significa |
| --- | ---: | ---: | --- |
| Forest sin tunear, 5 folds | 0,5614 | 0,0104 | la referencia |
| **Forest tuneado, CV anidada** | **0,5642** | **0,0080** | **insesgada: lo que vale el tuning** |
| Forest tuneado, 5 folds ordinarios | 0,5640 | 0,0075 | **OPTIMISTA** — misma partición que la búsqueda |

- **Resultado:** el tuning aporta **+0,0028** de PR-AUC contra una desviación entre folds
  de 0,0080. **No supera el umbral de 0,02.** El tuning no produjo una ganancia
  distinguible del ruido de partición.
- **Reproducción:** `uv run python scripts/run_tuning.py`. Runs de MLflow:
  `66f604ed459f4f979539c6750fae4ef7` (`tuning-nested-cv`, la estimación honesta, etiquetada
  `estimate_is_unbiased=true`) y `3de0934639d0426182c03102d36f4b39`
  (`tuned-random-forest`, la estimación optimista, etiquetada `estimate_is_unbiased=false`).
  El primero lleva como artefactos las métricas por fold externo, los mejores parámetros de
  cada fold y **los 150 trials completos**.
- **Mejores hiperparámetros del estudio final**, sobre las 30.000 filas:
  `{'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 18, 'max_features': 0.3}`.
- **Interpretación:**
  - **El tuning no aportó nada distinguible.** +0,0028 es un tercio de la desviación entre
    folds y una séptima parte del umbral. Con este espacio, este presupuesto y este
    protocolo, la respuesta es que los hiperparámetros por defecto ya estaban donde la
    métrica se aplana.
  - **El sesgo optimista resultó ser nulo: −0,0003.** Es un hallazgo por derecho propio.
    La CV anidada existe para corregir un sesgo que aquí, medido, **no apareció**. La
    lectura coherente es que la métrica está en una meseta: si ninguna configuración es
    apreciablemente mejor que otra, elegir la mejor de 30 sobre una partición no puede
    sobreajustar esa partición, porque no hay nada que sobreajustar. Esto **no** significa
    que la CV anidada fuera innecesaria: sin ella, el cero no se podría afirmar.
  - **Los cinco folds eligen hiperparámetros distintos**, y eso es la misma conclusión
    vista de otro modo: `n_estimators` ∈ {300, 500}, `max_depth` ∈ {8, 10},
    `min_samples_leaf` ∈ {1, 3, 6}. Un parámetro sobre el que cinco folds no se ponen de
    acuerdo es un parámetro que los datos no restringen. El estudio final eligió
    `min_samples_leaf=18`, un valor **fuera del rango** que eligió cualquier fold externo,
    lo que confirma que en esa dimensión la métrica no separa.
  - **La única excepción es `max_features=0,3`, elegido por los cinco folds y por el
    estudio final.** Ése sí es un parámetro que los datos constriñen, y es el único
    hallazgo accionable del turno en materia de hiperparámetros.
  - **Acumulado contra la logística:** el conjunto completo —forest, sin tratamiento del
    desbalance, tuneado— alcanza 0,5642 contra 0,5402, es decir **+0,0240**, que sí supera
    el umbral. Pero de esos 24 puntos de diezmilésima, el tuning aporta 3.
  - **Límite explícito:** 30 trials en cuatro dimensiones no agotan el espacio. La
    afirmación defendible es "no se encontró ganancia con este presupuesto", no "no existe
    ganancia".

### 007 — Calibración: ¿sirve calibrar el random forest?

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Objeto evaluado:** `build_production_forest()` —random forest tuneado, `class_weight=None`—
  crudo y envuelto en dos mapas de calibración. Lo único que cambia es el mapa.
- **Datos:** UCI 350, 30.000 filas, prevalencia 0,221200.
- **Protocolo:** `StratifiedKFold(5, shuffle=True, random_state=42)`. El calibrador se ajusta
  **dentro** de cada fold, sobre 3 particiones internas de las filas de entrenamiento de ese
  fold, con `ensemble=False`. Ninguna fila participa en el ajuste del mapa que después la
  puntúa. La curva de fiabilidad se dibuja con probabilidades **fuera de fold**, por la misma
  razón: una curva dibujada en muestra se apoya en la diagonal por un motivo que no tiene que
  ver con que el modelo esté bien calibrado.

**Métricas — media ± desviación estándar entre los 5 folds**

| Métrica | Sin calibrar (referencia) | Sigmoide (Platt) | Isotónica |
| --- | --- | --- | --- |
| PR-AUC | **0,5640 ± 0,0075** | **0,5640 ± 0,0075** | 0,5507 ± 0,0078 |
| ROC-AUC | 0,7874 ± 0,0083 | 0,7874 ± 0,0083 | 0,7868 ± 0,0076 |
| KS | 0,4388 ± 0,0197 | 0,4388 ± 0,0197 | 0,4338 ± 0,0170 |
| Gini | 0,5748 ± 0,0166 | 0,5748 ± 0,0166 | 0,5736 ± 0,0152 |
| **Brier** (menor es mejor) | **0,133228 ± 0,0021** | 0,134009 ± 0,0022 | 0,133551 ± 0,0019 |
| precision@top-10% | 0,7070 ± 0,0177 | 0,7070 ± 0,0177 | 0,7055 ± 0,0169 |
| precision@top-5% | 0,7713 ± 0,0223 | 0,7713 ± 0,0223 | 0,7665 ± 0,0126 |

- **Baseline:** el **forest sin calibrar**, sobre los mismos folds. Es el baseline correcto
  porque la pregunta es qué añade calibrar, no qué añade el forest.

**Fiabilidad fuera de fold, 10 bins por cuantiles — peor brecha absoluta en cualquier bin**

| Arm | Peor \|brecha\| | Media predicha | Prevalencia | Sesgo |
| --- | ---: | ---: | ---: | ---: |
| **Sin calibrar** | **0,0106** | 0,221243 | 0,221200 | +0,000043 |
| Sigmoide | 0,0481 | 0,221889 | 0,221200 | +0,000689 |
| Isotónica | 0,0157 | 0,222037 | 0,221200 | +0,000837 |

- **Resultado:** **la calibración no ayudó.** El forest crudo obtuvo el mejor Brier de los
  tres y la menor brecha en cada decil. Ninguna de las dos ganancias esperadas apareció.
- **Reproducción:** `uv run python scripts/run_calibration.py`. Runs etiquetados
  `run_type=calibration`: `12b5fce218db42e5a5d17422b8398fa7` (sin calibrar),
  `346d56bae88642c0932cd3cf9ea51677` (sigmoide), `b773e8f6119d42f788433b2a0c7a5a2b`
  (isotónica), y `efbfafdbf5a3462ea6fa261f56f43646` con la curva de fiabilidad como
  artefacto.
- **Interpretación:**
  - **El forest ya estaba calibrado, y hay un mecanismo detrás.** 300 árboles promediando
    frecuencias de hoja producen una probabilidad, no una proporción de votos. La media
    predicha coincide con la prevalencia con un sesgo de **+0,000043**. No quedaba mucho que
    corregir.
  - **La isotónica costó 0,0133 de PR-AUC, y eso merece explicación.** Un mapa monótono no
    puede reordenar nada. Pero la regresión isotónica es solo **no decreciente**: colapsa
    rangos de score en un valor constante, lo que **crea empates**, y los empates son de lo
    que están hechas `precision@top-k%` y la precisión media. La sigmoide es estrictamente
    creciente y dejó las siete métricas de ordenamiento **bit-idénticas**, lo que confirma el
    mecanismo por contraste.
  - **La sigmoide se conserva a pesar de todo**, y la decisión es del Verificador. Su costo
    está medido y es despreciable: **+0,0008 de Brier —dentro de la desviación entre folds de
    0,002— y exactamente 0,0000 de PR-AUC**. Un mapa de dos parámetros es un seguro barato si
    la distribución de scores se mueve. Quitarla es una alternativa defendible.
  - **Límite:** la curva se midió sobre esta población. Un modelo bien calibrado aquí no lo
    está necesariamente sobre otra mezcla de clientes.

### 008 — Umbral operativo desde una matriz de costos

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Objeto evaluado:** el modelo productivo —forest + calibración sigmoide— barrido sobre 201
  umbrales de 0,000 a 1,000.
- **Datos:** las 30.000 filas, con probabilidades **fuera de fold**: cada fila puntuada por un
  modelo que no se ajustó con ella. Elegir un umbral sobre probabilidades en muestra lo
  ajustaría a filas memorizadas y la matriz de confusión describiría el conjunto de
  entrenamiento.
- **Protocolo:** `StratifiedKFold(5, shuffle=True, random_state=42)`. Costo esperado
  `ratio·FN + FP`. Empates resueltos hacia el umbral **más alto**, que rechaza a menos gente:
  a igual costo esperado, intervenir sobre menos personas es estrictamente preferible.
- **Supuesto de costos:** un falso negativo cuesta **5 veces** un falso positivo. Solo importa
  el cociente. El umbral **no** se eligió maximizando F1 ni ningún criterio interno del
  modelo; el ADR-0002 descarta F1 justamente por eso.

**Resultado a 5:1 — umbral 0,160**

| | Predicho: paga | Predicho: incumple |
| --- | ---: | ---: |
| **Pagó de verdad** | 16.498 | 6.866 |
| **Incumplió de verdad** | 1.867 | 4.769 |

Recall 0,7187 · Precisión 0,4099 · Costo esperado 16.201 unidades de falso positivo.

**Sensibilidad al supuesto**

| FN:FP | Umbral | Rechazados | Atrapados | Perdidos | Recall | Precisión |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3:1 | 0,220 | 7.768 | 3.942 | 3.826 | 0,5940 | 0,5075 |
| **5:1** | **0,160** | **11.635** | **4.769** | **6.866** | **0,7187** | **0,4099** |
| 10:1 | 0,105 | 22.329 | 6.200 | 16.129 | 0,9343 | 0,2777 |

- **Baseline:** los dos extremos degenerados del propio barrido, que el grid incluye en vez de
  asumir. En el umbral **0,000** se rechaza a todo el mundo —`scores >= 0` siempre es cierto—
  y el costo es `FP = 23.364`. En el umbral **1,000** se acepta a todo el mundo y el costo es
  `5 × 6.636 = 33.180`. El umbral elegido cuesta **16.201**, un **30,7% menos que rechazar a
  todos** y un **51,2% menos que aceptar a todos**. Un umbral que no batiera a ambos extremos
  no valdría la pena calcularlo.
- **Resultado en lenguaje de negocio:** sobre 30.000 clientes se rechazarían **11.635** (38,8%
  del libro); de ellos **4.769** habrían incumplido de verdad y **6.866** habrían pagado. Se
  atrapan **4.769 de 6.636** incumplidores (71,9%) y **1.867** pasan.
- **Reproducción:** `uv run python scripts/run_threshold_selection.py`. Run
  `494ca908b3594bd6977b2eac305bbcfc`, etiquetado `run_type=threshold-selection`, con el
  barrido completo y la curva de costo como artefactos.
- **Interpretación:**
  - **La sensibilidad es el hallazgo, no el umbral.** Mover el cociente de 3:1 a 10:1 mueve el
    umbral 0,115 y **14.561 clientes, el 48,5% del libro**. Quien fije el cociente está tomando
    una decisión mucho mayor que la de elegir el modelo. Todo lo medido en esta fase —modelos,
    desbalance, tuning, calibración— mueve menos negocio que ese único supuesto.
  - **A 5:1 el 59% de los rechazos son clientes que habrían pagado.** No es un defecto del
    umbral sino aritmética de la prevalencia: con 22% de positivos y un modelo de PR-AUC 0,56,
    atrapar al 72% de los incumplidores exige rechazar al 29% de los pagadores. Presentar el
    modelo como "acierta el 41% de los rechazos" y no como "atrapa el 72% de los
    incumplidores" es la misma cifra leída desde el otro lado.
  - **Límite:** el cociente 5:1 es un supuesto sin respaldo empírico en este proyecto. No hay
    datos de exposición, de recuperación ni de margen, así que no se puede medir; declararlo
    es más útil que inventarlo.

### 009 — Modelo productivo registrado

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling
- **Objeto evaluado:** el `Pipeline` completo —preprocesador + forest calibrado— registrado
  como `credit-risk-default-probability` versión 1.
- **Datos:** ajustado sobre las **30.000 filas**. Las métricas asociadas **no salen de ese
  ajuste**: son las de validación cruzada de la entrada 006.
- **Protocolo:** ninguno. **Este registro no mide nada.** Un ajuste sobre todo el dataset es
  correcto para producir el artefacto y sería fuga si de él se leyera una métrica; la
  distinción está escrita en el docstring del script y en la sección 2 del Model Card.

**Métricas asociadas al modelo en el registro, con su baseline al lado**

| Métrica | Piso trivial | Baseline logística | **Modelo registrado** |
| --- | ---: | ---: | ---: |
| **PR-AUC** | 0,221200 | 0,540173 | **0,564230 ± 0,007962** |
| ROC-AUC | 0,500000 | 0,776187 | 0,786279 |
| KS | 0,000000 | 0,421780 | 0,439205 |
| Gini | 0,000000 | 0,552373 | 0,572558 |
| Brier (menor es mejor) | 0,221200 | 0,183357 | 0,133408 |
| precision@top-10% | 0,221200 | 0,692667 | 0,706333 |
| precision@top-5% | 0,221200 | 0,742667 | 0,768667 |

- **Baseline:** los dos baselines viajan **dentro del registro**, como métricas del propio
  modelo (`cv_baseline_trivial_pr_auc` y `cv_baseline_logistic_pr_auc`). Una entrada de
  registro que guarde 0,564 y no el piso de 0,221 contra el que se lee invita a citar el
  número solo, que es exactamente el modo de falla que la sección 7.3 de la metodología
  describe.
- **Resultado:** registrado como versión **1**, con firma inferida de 23 columnas crudas y
  salida `(-1, 2)` en `float64`. Prueba de ida y vuelta superada: el artefacto cargado desde
  el registro reproduce sus propias predicciones con una diferencia máxima de **1,1×10⁻¹⁶**.
- **Reproducción:** `uv run python scripts/register_production_model.py`. Run
  `418c633718074bae9473e9ddbbef26d6`, etiquetado `run_type=production-candidate`.
- **Interpretación:**
  - **El artefacto es el pipeline entero y no el clasificador.** Un consumidor le entrega las
    23 columnas crudas que devuelve `loader.load_dataset` y nunca construye la matriz él
    mismo. Es la garantía de la sección 6.3 de la metodología convertida en objeto.
  - **Las predicciones no son reproducibles bit a bit, y quedó medido.** Con `n_jobs=-1` el
    forest acumula los votos de 300 árboles entre hilos; la suma en coma flotante no es
    asociativa y el mismo objeto llamado dos veces difiere en **5×10⁻¹⁶**. Con `n_jobs=1` la
    diferencia es exactamente cero, lo que identifica la causa. Es quince órdenes de magnitud
    por debajo del umbral de 0,160 y no puede cambiar una decisión, pero **invalida cualquier
    test que exija igualdad de bits sobre predicciones**.
  - **La firma declara enteros y eso es un riesgo abierto para la fase de API.** MLflow
    advierte que un entero de Python no puede representar un faltante: si una petición llega
    con un nulo, el enforcement de esquema fallará. El validador garantiza que el dataset no
    tiene nulos; una API recibe lo que le mandan.
  - **Límite:** el modelo está registrado como **candidato**, no promovido a producción. Lo
    que falta antes de un despliegue real está en la sección 8 del Model Card, empezando por
    la validación fuera de tiempo que el ADR-0001 declaró imposible con estos datos.
