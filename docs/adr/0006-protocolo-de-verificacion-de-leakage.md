# ADR 0006 — Protocolo de verificación de leakage

- **Status:** Accepted
- **Date:** 2026-08-26

---

## Contexto

La prueba del target barajado es la defensa de nivel 2 de la sección 6.5 de
`docs/METHODOLOGY.md`: se permuta el target al azar, se entrena el pipeline completo
contra esa permutación y se exige que el desempeño colapse al azar. Si no colapsa, hay
fuga de información.

Esa prueba **necesita un criterio de aprobación fijado de antemano**, y el criterio es
donde la prueba se gana o se pierde. Un umbral demasiado laxo la convierte en decorativa:
aprueba siempre, incluida una fuga real. Uno demasiado estricto la hace fallar por ruido
de partición sobre un pipeline sano, y una prueba que falla sin motivo se termina
desactivando.

El problema es que el umbral no se puede razonar desde un formulario: hay que saber
cuánto se mueve cada métrica **cuando no hay señal**, sobre este dataset y este pipeline.

---

## Decisión

### 1 · La tolerancia se fija en 0,020 para ROC-AUC y 0,015 para PR-AUC

Ambas medidas como desviación absoluta de la media entre folds respecto del **piso sin
señal**, que no es el mismo número para cada métrica: 0,5 para ROC-AUC y la prevalencia
—0,2212— para PR-AUC.

Los umbrales se derivaron **midiendo la distribución nula**: ocho permutaciones
independientes del target, con el pipeline completo ajustado una vez por fold en cada
una. El error estándar real resultó ser **0,00593 para ROC-AUC y 0,00335 para PR-AUC**,
de modo que las tolerancias equivalen a **3,4 y 4,5 errores estándar** respectivamente.

#### La primera estimación era incorrecta, y queda registrada

Una primera estimación, basada en **puntuaciones aleatorias independientes** en vez del
modelo ajustado, daba 0,00403 y 0,00244. Es decir, **subestimaba la dispersión en cerca
de un tercio**.

El mecanismo del error: simular puntuaciones aleatorias mide el ruido de muestreo de la
métrica, pero omite las dos cosas que aquí la mueven. Primero, el modelo **se ajusta**
sobre las etiquetas permutadas, de modo que puede perseguir ruido. Segundo, los cinco
conjuntos de entrenamiento de un 5-fold **se solapan al 80%**, así que los cinco
resultados por fold no son extracciones independientes y su media no reduce la varianza
por `sd/√5`.

Queda registrado porque el error es reutilizable: la estimación correcta **exige el
modelo ajustado**, no una simulación de puntuaciones. Un umbral derivado de la cifra
ingenua habría sido demasiado estricto y habría terminado fallando sobre un pipeline
honesto.

#### Cuánta señal separa un caso del otro

Medido sobre los mismos folds y la misma semilla, el pipeline con el **target real**
frente al piso sin señal:

| Métrica | Piso sin señal | Pipeline con target real | Distancia | En errores estándar | En múltiplos de la tolerancia |
| --- | ---: | ---: | ---: | ---: | ---: |
| ROC-AUC | 0,500000 | 0,776187 | 0,276187 | **46,6** | 13,8× |
| PR-AUC | 0,221200 | 0,540173 | 0,318973 | **95,2** | 21,3× |

La constancia que deja esta tabla es la que hace que el criterio no sea decorativo: la
banda de aceptación es entre catorce y veintiún veces más estrecha que la distancia a una
señal real. Hay sitio de sobra para el ruido de permutación y ninguno para una fuga.

### 2 · `precision@top-k%` resuelve los empates por valor esperado

El corte del decil se resuelve devolviendo el **valor esperado sobre el grupo empatado**
—las filas estrictamente por encima del corte cuentan enteras, y las plazas restantes se
llenan del grupo empatado a la tasa de positivos de ese grupo— en vez de tomar las
primeras `k` filas tras ordenar.

Razón medida, y son dos casos reales de este proyecto, no hipotéticos:

- El **clasificador aleatorio estratificado** produce solo dos puntuaciones distintas, de
  modo que **ambos cortes**, el del 10% y el del 5%, caen dentro del mismo grupo de
  empates. Se observa en la salida de los baselines: las dos métricas valen exactamente
  lo mismo, 0,215502.
- El **clasificador trivial** produce una sola puntuación para toda la población, así que
  el corte cae íntegramente dentro de un único grupo de empates que abarca las 30.000
  filas.

Con la política de "primeras `k` filas", el número resultante dependería del orden en que
las filas llegaron al arreglo, **que no es una propiedad del modelo**: las mismas filas
barajadas darían otro número.

Con la política del valor esperado, el clasificador trivial devuelve **exactamente la
prevalencia**, que es la respuesta correcta: un modelo que no ordena nada no puede
concentrar incumplidores en ningún decil. Y cuando ningún empate cruza el corte —el caso
ordinario de un modelo real— la política devuelve la precisión top-k de toda la vida, de
modo que no cuesta nada cuando no hace falta.

---

## Alternativas consideradas

**Tolerancia de cinco errores estándar sobre la estimación simulada.** Descartada porque
la estimación simulada era incorrecta, por el mecanismo de la decisión 1. Los mismos
umbrales numéricos, leídos contra la medición real, son 3,4 y 4,5 errores estándar. Se
conservaron los números y se corrigió su justificación, no al revés.

**Desempate por orden de llegada en `precision@top-k%`.** Descartada por la decisión 2:
produce un número que cambia si las filas se barajan.

**Umbral expresado como cuantil de la distribución nula** en vez de como múltiplo del
error estándar. Descartada por el tamaño de la muestra: ocho permutaciones sitúan bien el
centro y la dispersión, pero no alcanzan para caracterizar las colas, y un cuantil
estimado con ocho puntos afirmaría una precisión que la medición no tiene.

---

## Consecuencias

### Positiva

La prueba de leakage tiene un criterio **derivado de una medición y no de una intuición**,
y por lo tanto se puede citar. `scripts/measure_null_distribution.py` regenera las cifras
y `docs/analysis/null-distribution-evidence.md` las transcribe, de modo que un tercero
puede reproducir el umbral en vez de aceptarlo.

### Negativa

La medición de la nula es **cara**: ocho ajustes completos del pipeline, cuarenta ajustes
contando los folds. Se corre **cuando cambia el pipeline**, no en cada turno. La prueba de
leakage en sí sigue siendo barata —un solo ajuste— y esa es la que se corre siempre.

### Riesgo

Los umbrales se derivaron **sobre este dataset y este pipeline**. Un cambio sustancial en
cualquiera de los dos —otra fuente de datos, otro esquema de validación, un preprocesador
con pasos nuevos— obliga a remedirlos. El número de folds entra directamente en el
argumento: el solapamiento del 80% entre conjuntos de entrenamiento es una propiedad de
`n_splits=5`, y con otro valor la dispersión cambia.
