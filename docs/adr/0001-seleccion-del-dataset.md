# ADR 0001 — Selección del dataset

- **Status:** Accepted
- **Date:** 2026-08-24

---

## Contexto

El proyecto necesita un dataset tabular para un problema de clasificación binaria de
riesgo de crédito.

Los criterios de selección se fijaron **antes de mirar candidatos**, para no elegir
primero el dataset y justificarlo después:

- **(a) Fricción mínima de adquisición.** El enunciado del entregable indica
  explícitamente que los procesos complejos de ingeniería de datos no serán
  evaluados, así que el esfuerzo invertido en conseguir los datos no devuelve nota.
- **(b) Tamaño por debajo de 100 MB**, según recomendación del mismo enunciado.
- **(c) Presencia de variables categóricas y numéricas**, para poder ejercitar
  one-hot encoding, bucketing y escalado.
- **(d) Un diccionario de datos documentable en un esfuerzo razonable**, ya que es un
  requisito explícito del README.
- **(e) Desbalance de clase suficiente para que la accuracy resulte engañosa**,
  porque eso es parte de lo que el proyecto busca demostrar.

## Decisión

Se usa **"Default of Credit Card Clients"** del UCI Machine Learning Repository
(ID 350):

| Propiedad | Valor |
| --- | --- |
| Registros | 30.000 |
| Variables predictoras | 23, más el target |
| Tamaño | ~3 MB |
| Clase positiva | ~22% |

## Alternativas consideradas

**Give Me Some Credit (Kaggle).** 150.000 filas y solo 10 variables, todas
numéricas. Descartado porque la ausencia total de variables categóricas impide
ejercitar one-hot encoding y feature hashing, que son parte del temario del curso.

**Home Credit Default Risk, tabla `application_train`.** Descartado por dos razones.
El CSV crudo supera el límite recomendado de 100 MB, y sus aproximadamente 120
columnas convierten el requisito del diccionario de datos en trabajo mecánico de
transcripción sin valor de aprendizaje.

**Lending Club.** Descartado por tamaño y porque exige una limpieza previa de
columnas post-desenlace. Esa limpieza es una lección valiosa pero de una sola vez, y
el costo de adquisición no se justifica frente al beneficio.

## Consecuencias

### Positivas

- **Adquisición trivial** vía la librería `ucimlrepo`.
- **Estructura de panel** con seis meses de historial de pago, que permite derivar
  features de comportamiento con significado real. Ahí está el aprendizaje de feature
  engineering.
- **Diccionario de datos de 24 filas**, documentable en una sesión.

### Negativas

- Es un dataset **muy modelado públicamente**, por lo que el proyecto no aporta
  originalidad en la elección del problema. El valor está en la ejecución.

### Limitación aceptada

El dataset **no contiene fecha de originación del crédito**, por lo que no es posible
una validación out-of-time.

La consecuencia directa es que la estrategia de validación será **Stratified K-Fold y
no un corte cronológico**. Esto se documentará también en el Model Card como
limitación para un despliegue real, donde sí se exigiría validación fuera de tiempo.

### Riesgo que invalidaría esta decisión

Descubrir durante el EDA que las **variables de comportamiento no tienen poder
predictivo suficiente**, lo que dejaría sin sustento la hipótesis principal del
proyecto.
