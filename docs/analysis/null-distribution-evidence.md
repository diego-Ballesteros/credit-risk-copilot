# Evidencia — Distribución nula de las métricas de ordenamiento

Medición de referencia para los umbrales que fija el **ADR-0006**. Es un registro de
evidencia: describe qué se midió y qué salió, y **no saca conclusiones sobre el modelo**.
La lectura de los resultados del modelo vive en `docs/EVALUATION.md`.

- **Fecha:** 2026-08-26
- **Reproducción:** `uv run python scripts/measure_null_distribution.py`
- **Objeto medido:** el pipeline completo del proyecto —preprocesador de
  `build_preprocessor` más el estimador de `build_logistic_regression`— entrenado contra
  un target permutado.
- **Datos:** UCI 350 vía `loader.load_dataset`, 30.000 filas. Prevalencia **0,221200**,
  idéntica en las ocho permutaciones: una permutación mueve etiquetas entre filas y no
  crea ni destruye ninguna.
- **Protocolo:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, semilla de
  `config.py`. El preprocesador se ajusta una vez por fold, sobre las filas de
  entrenamiento de ese fold. Ocho permutaciones independientes, identificadas por los
  índices 0 a 7, que son la semilla de cada permutación y nada más. En total **40 ajustes
  completos** del pipeline.

## Por qué se mide con el modelo ajustado y no con puntuaciones aleatorias

La pregunta es cuánto se aparta cada métrica de su piso cuando el target no lleva
información. La forma barata de responderla —extraer puntuaciones al azar y calcular la
métrica— **omite los dos efectos que aquí la mueven**: que el modelo *se ajusta* sobre las
etiquetas permutadas y puede perseguir ruido, y que los cinco conjuntos de entrenamiento
de un 5-fold **se solapan al 80%**, de modo que los cinco resultados por fold no son
extracciones independientes.

Ambas mediciones están abajo para que la diferencia quede registrada.

## Resultado — media sobre los 5 folds, una fila por permutación

| Permutación | ROC-AUC | Desviación de 0,5 | PR-AUC | Desviación de la prevalencia |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0,500533 | +0,000533 | 0,222450 | +0,001250 |
| 1 | 0,503147 | +0,003147 | 0,221089 | −0,000111 |
| 2 | 0,497178 | −0,002822 | 0,220404 | −0,000796 |
| 3 | 0,508309 | +0,008309 | 0,227564 | +0,006364 |
| 4 | 0,492821 | −0,007179 | 0,220945 | −0,000255 |
| 5 | 0,501364 | +0,001364 | 0,221498 | +0,000298 |
| 6 | 0,501178 | +0,001178 | 0,222280 | +0,001080 |
| 7 | 0,511708 | +0,011708 | 0,229359 | +0,008159 |

## La distribución nula

| | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Piso sin señal | 0,500000 | 0,221200 |
| Centro empírico | 0,502030 | 0,223199 |
| Desplazamiento respecto del piso | +0,002030 | +0,001999 |
| **Error estándar de la media entre folds** | **0,005929** | **0,003352** |
| Peor desviación absoluta observada | 0,011708 | 0,008159 |

## Comparación con la estimación ingenua

| Método | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Puntuaciones aleatorias independientes, 4.000 extracciones | 0,004030 | 0,002440 |
| **Pipeline ajustado, 8 permutaciones** | **0,005929** | **0,003352** |
| Factor de subestimación de la estimación ingenua | 1,47× | 1,37× |

## Qué implica para los umbrales del ADR-0006

| Métrica | Tolerancia | En errores estándar medidos | Frente a la peor desviación observada |
| --- | ---: | ---: | ---: |
| ROC-AUC | ±0,020 | 3,4 | 1,7× |
| PR-AUC | ±0,015 | 4,5 | 1,8× |

## Nota de alcance

Ocho permutaciones sitúan el centro y la dispersión con tres cifras significativas, y
**no alcanzan para caracterizar las colas**. Por eso el criterio del ADR-0006 se enuncia
en errores estándar y no en cuantiles: un cuantil estimado con ocho puntos afirmaría una
precisión que esta medición no tiene.

Las cifras se derivaron **sobre este dataset y este pipeline**, con `n_splits=5`. El
solapamiento del 80% entre conjuntos de entrenamiento es una propiedad de ese número de
folds y entra directamente en la dispersión medida, de modo que un cambio de esquema de
validación obliga a repetir la medición.
