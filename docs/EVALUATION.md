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
  `d90d6c32398b4d839201d019794c2000`, etiquetado `run_type=leakage-check-shuffled-target`
  e `is_real_result=false` para que no se confunda con un resultado. **Filtrar por esa
  etiqueta antes de comparar modelos.**
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
