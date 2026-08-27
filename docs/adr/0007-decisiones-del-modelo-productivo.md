# ADR 0007 — Decisiones del modelo productivo

- **Status:** Accepted
- **Date:** 2026-08-26

---

## Contexto

El cierre de la fase 2 dejó tres decisiones sobre el artefacto que se registra como modelo
productivo. Las tres comparten una propiedad que conviene declarar de entrada: **ninguna se
apoya en una ganancia de desempeño medida**. Dos de ellas se sostienen porque su costo está
acotado y medido, y la tercera porque preserva una garantía de seguridad.

Registrarlas juntas es deliberado. Una decisión que no mejora nada y cuesta poco es
exactamente el tipo de decisión que seis meses después nadie recuerda por qué se tomó, y que
alguien revierte o refuerza sin saber que la evidencia ya existía.

---

## Decisión

### 1 · El modelo productivo lleva calibración sigmoide, pese a que la medición muestra que no aporta

**La decisión de calibrar se tomó antes de medirla, y la evidencia la contradice.** Eso queda
registrado aquí de forma explícita en vez de presentarse como una mejora.

La entrada 007 de `docs/EVALUATION.md` comparó los dos métodos contra el forest crudo, con el
calibrador ajustado dentro de cada fold y la curva dibujada con probabilidades fuera de fold:

| | Brier | Peor brecha en un decil |
| --- | ---: | ---: |
| **Sin calibrar** | **0,133228** | **0,0106** |
| Sigmoide | 0,134009 | 0,0481 |
| Isotónica | 0,133551 | 0,0157 |

El forest **sin calibrar obtuvo el mejor Brier de los tres** y la menor brecha en cada decil.
La diferencia de la sigmoide es **+0,0008**, dentro de la desviación entre folds de **0,002**,
y las **siete métricas de ordenamiento quedan bit-idénticas**.

Hay un mecanismo detrás del resultado: 300 árboles promediando frecuencias de hoja ya
producen una probabilidad y no una proporción de votos. La media predicha coincide con la
prevalencia con un sesgo de **+0,000043**. No quedaba mucho que corregir.

**Se mantiene como decisión conservadora, no como mejora.** Un mapa de dos parámetros es un
seguro barato si la distribución de puntuaciones se desplaza en producción, y su costo está
acotado y medido: ocho diezmilésimas de Brier y cero de PR-AUC. Quitar la calibración es una
alternativa defendible con la misma evidencia.

#### Alternativa descartada: calibración isotónica

Costó **0,0133 de PR-AUC**, y el mecanismo se midió en vez de suponerse.

Un mapa monótono no puede reordenar nada, así que a primera vista no debería mover ninguna
métrica de ordenamiento. Pero la regresión isotónica es **no decreciente y no estrictamente
creciente**: colapsa rangos de puntuación en un valor constante, y eso **crea empates**.
`precision@top-k%` y la precisión media se calculan precisamente sobre esos empates, de modo
que aplanar el score degrada ambas sin haber invertido el orden de ningún par.

La sigmoide, **estrictamente creciente**, dejó las siete métricas de ordenamiento
bit-idénticas. Ese contraste es lo que confirma el mecanismo: dos mapas monótonos, uno
estricto y otro no, y solo el no estricto mueve el número.

### 2 · El modelo productivo usa los hiperparámetros del tuning, pese a que la ganancia estuvo dentro del ruido

La entrada 006 midió la ganancia del tuning en **+0,0028 de PR-AUC** contra una desviación
entre folds de **0,0080**, con validación cruzada anidada. Está dentro del ruido y muy por
debajo del umbral de significancia práctica de 0,02.

**Se adoptan igualmente porque adoptarlos no cuesta nada** —son la misma familia de modelo y
el mismo tiempo de ajuste— **y porque el estudio dejó al menos un hallazgo que los datos sí
restringen**: `max_features = 0,3`, elegido por los **cinco folds internos y por el estudio
final**. El default sin tunear usa `"sqrt"`, que sobre 110 columnas es aproximadamente 0,1.
Esa es una dimensión donde la evidencia habló.

**Los demás hiperparámetros no están restringidos por los datos, y queda registrado.** Los
cinco folds eligieron valores distintos de `min_samples_leaf` —1, 1, 6, 3 y 1— y el estudio
final eligió **18**, un valor **fuera del rango que eligió cualquiera de ellos**. Lo mismo con
`n_estimators` (300 y 500) y `max_depth` (8 y 10). Un parámetro sobre el que cinco folds no se
ponen de acuerdo es un parámetro que la métrica no separa.

### 3 · La serialización declara los tipos de confianza en vez de volver a cloudpickle

MLflow 3 serializa modelos de scikit-learn con **`skops`** en vez de pickle, y `skops` se
niega a reconstruir cualquier clase que no se le haya indicado como confiable. El registro
del modelo falló con esa negativa, nombrando las clases propias del proyecto:
`AttachPaymentBehaviourFeatures`, `CollapseEducation`, `PercentileClipper` y
`PaymentBehaviourFeatures`, más tres internos de scikit-learn y numpy.

**La negativa es la funcionalidad, no el defecto.** Cargar un pickle ejecuta lo que el archivo
diga que se ejecute, y un artefacto de modelo es una vía de ataque real.

Se declara la lista de **siete tipos** en `skops_trusted_types`. Los cuatro primeros son
transformadores definidos en este repositorio y revisados como cualquier otro código de aquí;
los tres restantes son internos de scikit-learn y un tipo de numpy que el forest calibrado
contiene. Nada de terceros y nada dinámico entra en la lista.

#### Alternativa descartada: volver a `serialization_format="cloudpickle"`

Funciona, nunca falla, y **confía en todo**. Renuncia a la garantía completa para evitar
escribir siete líneas.

Nombrar los tipos tiene además una propiedad que cloudpickle no tiene: **hace legible el
contenido del artefacto**. Un cambio en el pipeline que añada un paso propio **fallará
ruidosamente** en el registro, y la corrección será leer el nombre nuevo y decidir si
corresponde. Con cloudpickle ese cambio pasaría en silencio.

---

## Consecuencias

### Positivas

**Las tres decisiones tienen su evidencia citable y su alternativa descartada por escrito.**
Ninguna se apoya en una intuición, y las dos que contradicen lo que se esperaba —la
calibración que no calibra mejor, el tuning que no mejora— quedan registradas como tales en
vez de presentarse como logros.

**El artefacto que se despliega es auditable.** La lista de tipos de confianza enumera lo que
el archivo contiene, y una adición futura no puede colarse.

### Negativas

**El modelo productivo carga un paso que no mejora nada.** La calibración sigmoide añade tres
ajustes internos al entrenamiento y un mapa más que mantener, a cambio de un seguro cuyo
beneficio no se ha observado en estos datos y solo se materializaría si la distribución se
desplazara.

**La lista de tipos de confianza es frágil por diseño.** Cambiar el pipeline rompe el
registro hasta que alguien actualice la lista. Es fragilidad deliberada —falla ruidosamente en
vez de en silencio— pero es trabajo recurrente.

**Los hiperparámetros adoptados dan una falsa impresión de precisión.** `min_samples_leaf=18`
parece un valor elegido, y la evidencia dice que la métrica no distingue ese valor de 1, 3 o
6. Quien lo lea sin la entrada 006 al lado creerá que está afinado.

### Riesgo — y es mayor que las tres decisiones juntas

**El umbral operativo de 0,160 depende de un supuesto de costos de 5:1 que no tiene respaldo
empírico en este dataset.** No hay datos de exposición, de recuperación ni de margen, así que
el cociente no se puede medir: se declaró.

Y la decisión es extraordinariamente sensible a él. Medido en la entrada 008:

| FN:FP | Umbral | Clientes rechazados |
| --- | ---: | ---: |
| 3:1 | 0,220 | 7.768 |
| **5:1** | **0,160** | **11.635** |
| 10:1 | 0,105 | 22.329 |

Mover el cociente entre 3:1 y 10:1 **desplaza a 14.561 clientes, el 48,5% del libro**. Para
comparar: **todo lo ganado por modelado en esta fase** —cambiar de regresión logística a
random forest, quitar el tratamiento de desbalance y tunear— **suma +0,0240 de PR-AUC**.

**El supuesto de costos mueve más negocio que todo el trabajo de modelado.** Quien fije ese
cociente toma una decisión mucho mayor que la de elegir el modelo, y ahora mismo el cociente
es una declaración y no una medición. Conseguir datos de exposición y recuperación que
permitan medirlo tiene más valor esperado que cualquier mejora adicional del modelo.
