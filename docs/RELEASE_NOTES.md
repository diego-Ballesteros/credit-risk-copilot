# Credit Risk Copilot — v1.0.0

**Proyecto final del Curso II · Especialización MLE 2026** · 31 de agosto de 2026

---

## Qué es

Un sistema de evaluación de riesgo crediticio con dos componentes integrados: un **modelo de
clasificación** que estima la probabilidad de que un cliente de tarjeta incumpla el pago del
mes siguiente, y un **copiloto agéntico** que usa ese modelo como una de sus cuatro
herramientas, contrasta el resultado contra un corpus normativo y responde citando el
fragmento que sostiene cada afirmación. El modelo no es un artefacto paralelo al copiloto: es
una herramienta suya, y el copiloto decide cuándo puntuar, cuándo explicar, cuándo simular y
cuándo consultar la norma. Todo lo que el sistema afirma sobre sí mismo está medido contra un
baseline —el modelo contra un clasificador trivial y una regresión logística, el copiloto
contra el mismo modelo de lenguaje sin herramientas—, y los resultados negativos se reportan
con el mismo detalle que los positivos.

---

## Componentes entregados

| Componente | Qué es | Dónde vive |
| --- | --- | --- |
| **Modelo productivo** | Un único `Pipeline` de la fila cruda a la probabilidad: preprocesador de 23 → 110 columnas, random forest calibrado. Registrado y versionado | MLflow Model Registry, `credit-risk-default-probability` **versión 1** |
| **Copiloto agéntico** | Grafo LangGraph con estado tipado, aristas condicionales y ciclo de replanificación acotado en 3 iteraciones. Cuatro herramientas con contrato Pydantic y RAG sobre 4 documentos normativos | `src/credit_copilot/agent/`, `src/credit_copilot/rag/` |
| **API REST** | Dos servicios FastAPI. Modelo: `/predict`, `/explain`, `/simulate`, `/health`, `/ready`, `/model-info`. Copiloto: `/chat`, `/health`, `/ready` | `src/credit_copilot/api/` |
| **Contenedores** | Dos imágenes separadas, multi-stage, usuario no-root, construidas y verificadas por CI antes de publicarse | GHCR: `model` **1,13 GB**, `agent` **2,47 GB** |
| **Scripts de ejecución** | 23, uno por medición, más preprocesamiento, entrenamiento y predicción | `scripts/` |
| **Documentación** | 10 ADRs, 13 mediciones registradas, 10 errores con su mecanismo, 6 documentos de evidencia reproducible | `docs/` |

**Calidad:** 380 tests, **82% de cobertura**, ruff, ruff-format y mypy en verde. La CI corre
las cuatro compuertas en cada push y en cada pull request.

**Reproducibilidad verificada, no supuesta:** un clon limpio —sin entorno, sin credenciales y
sin datos— llega hasta el dataset procesado siguiendo solo el README, y con credenciales
reproduce la prueba de ausencia de fuga, las puntuaciones del artefacto anclado y las 380
pruebas.

---

## Resultados principales

### El modelo, con su baseline al lado

Validación cruzada de 5 folds, preprocesador ajustado **dentro** de cada fold,
`random_state=42`. Ninguna de estas cifras se calculó sobre las filas con las que se ajustó el
artefacto.

| Métrica | Piso trivial | Baseline logística L2 | **Modelo productivo** |
| --- | ---: | ---: | ---: |
| **PR-AUC** (métrica de decisión) | 0,2212 | 0,5402 ± 0,0103 | **0,5642 ± 0,0080** |
| ROC-AUC | 0,5000 | 0,7762 | 0,7863 ± 0,0089 |
| Brier *(menor es mejor)* | 0,2212 | 0,1834 | **0,1334 ± 0,0021** |
| precision@top-10% | 0,2212 | 0,6927 | 0,7063 ± 0,0176 |

Revisar el decil peor puntuado encuentra **3,2 veces más incumplidores** que revisar un decil
al azar. Todo el trabajo de modelado —cambiar de logística a forest, quitar el tratamiento de
desbalance y tunear— suma **+0,0240 de PR-AUC**.

### La hipótesis principal se sostiene

El comportamiento de pago reciente supera a la demografía estática en **+0,2306 de PR-AUC**
(0,5362 contra 0,3056, sobre un piso de 0,2212), con las cinco diferencias por fold del mismo
signo. El camino inverso aporta **+0,0040**. SHAP lo confirma por una vía independiente: el
comportamiento concentra el **95,5%** de la atribución total.

### La hipótesis secundaria va en la dirección esperada

19 consultas de analista anotadas a mano, tres brazos, 57 corridas sin error:

| | **copiloto** | mismo modelo sin herramientas |
| --- | ---: | ---: |
| Afirmaciones normativas emitidas | 171 | 42 |
| … sostenidas por un fragmento verificado | **151** | 0 |
| Sin respaldo, por consulta | **1,05** | 2,21 |
| Afirmaciones sobre el mundo, sin fragmento | **8** | 40 |
| Banda correcta en consultas numéricas | **4 de 4** | 0 |
| Separación entre abstener cuando debe y cuando no | **+0,562** | 0,000 |
| Costo por consulta | 0,209 USD | 0,059 USD |

El copiloto **no gana callándose**: emite cuatro veces más afirmaciones normativas y sostiene
151 de 171 con una cita comprobada literalmente contra el fragmento.

### El sistema desplegado

3.000 peticiones HTTP reales contra la API, con un 5% deliberadamente malformado:

| | |
| --- | --- |
| Latencia p50 · 1 worker, concurrencia 1 | **75,6 ms** |
| Latencia p50 · 4 workers, concurrencia 8 | 152,3 ms, **53,2 req/s** |
| Tasa de error | 4,40% — **el 100% son los 422 esperados**, 0 fallos sobre 2.868 peticiones válidas |
| Drift máximo (PSI) | 0,0092, once veces por debajo del umbral de ruido |

Una petición rechazada cuesta **2,0 ms** contra 75,6 ms de una válida: el contrato la rechaza
antes de llegar al artefacto.

### El umbral operativo, y el supuesto que lo sostiene

**0,160**, derivado de suponer que un falso negativo cuesta **5 veces** un falso positivo. Ese
supuesto **fue declarado, no medido**. Mover el cociente entre 3:1 y 10:1 desplaza a **14.561
clientes, el 48,5% del libro** — más negocio que todo el trabajo de modelado junto.

---

## Limitaciones conocidas

Van sin suavizar, porque un evaluador que solo lea esta página debe enterarse de las cuatro
primeras.

1. **No hay validación fuera de tiempo, y es bloqueante para un despliegue real.** El dataset
   no contiene fecha de originación, así que la validación es `StratifiedKFold` y no un corte
   cronológico. Ninguna métrica de arriba dice nada sobre cómo se comportaría el modelo seis
   meses después ni ante un cambio de ciclo económico.

2. **El modelo trata de forma distinta a distintos grupos demográficos, y está medido.** En el
   umbral operativo la razón de impacto dispar cae por debajo de 0,80 para `EDUCATION`
   (**0,7364**) y `AGE` (**0,7796**), con hasta **10,3 puntos porcentuales** de diferencia en
   la tasa de rechazo por error **entre clientes que habrían pagado**. Este proyecto
   **cuantificó** la disparidad y **no la mitigó**. Está medido además que cegar el modelo a
   las variables protegidas elimina solo entre el **9% y el 30%** de la brecha: la mayor parte
   viaja por proxies en las features de comportamiento.

3. **No existe un holdout limpio.** El artefacto se ajustó sobre las 30.000 filas, así que el
   PR-AUC del flujo servido (**0,672144** contra 0,564230) es **optimismo por memorización y
   no desempeño**. Una degradación real habría dado un número menor.

4. **El corpus normativo incluye un documento sintético y uno derogado.** La política interna
   se redactó para este proyecto y **no representa la política de ninguna entidad financiera
   real**; el Capítulo II de la Circular Básica está **derogado desde el 1 de junio de 2023**.
   Los dos avisos viajan incrustados en el texto indexado, de modo que ningún fragmento
   recuperado puede presentarse sin ellos. Los cuatro documentos son extractos parciales.

5. **El recuperador falla en cerca de la mitad de las consultas** (hit@5 de 0,538) y **la
   ausencia de una cita no es prueba de que la norma no diga nada**. El sistema tampoco puede
   declarar por score que no sabe.

6. **El copiloto puede emitir texto normativo sin cita comprobable**, procedente de constantes
   del propio código. Lo citable es lo que aparece en la lista de citas, no lo que suena
   normativo.

7. **El desempeño es modesto en términos absolutos.** PR-AUC 0,5642 sobre un piso de 0,2212:
   con el umbral de 0,160, el **59% de los clientes rechazados habría pagado**.

8. **Población y periodo muy concretos.** Taiwán, 2005, tarjeta de crédito. El periodo
   **precede a la crisis de 2008**, así que el modelo no ha visto un ciclo adverso.

9. **Las predicciones no son reproducibles bit a bit.** Con `n_jobs=-1` la suma en coma
   flotante de 300 árboles difiere en 5×10⁻¹⁶ entre llamadas, y dos ajustes independientes
   difieren hasta **2,1×10⁻⁹** por el optimizador del calibrador. Quince y ocho órdenes de
   magnitud por debajo del umbral: no puede cambiar una decisión, pero invalida cualquier test
   de igualdad de bits.

> **El modelo está registrado como candidato y NO está desplegado.** No debe usarse para
> decisiones de crédito reales sin validación fuera de tiempo, ni como decisión automática sin
> revisión humana, ni sin asumir explícitamente la disparidad medida. La lista completa de usos
> desaconsejados está en la sección 8 de `docs/MODEL_CARD.md`.

---

## Enlaces

| Recurso | Dirección |
| --- | --- |
| **Repositorio** | <https://github.com/diego-Ballesteros/credit-risk-copilot> |
| **Experimentos de MLflow** *(públicos, sin credenciales)* | <https://dagshub.com/diego-Ballesteros/credit-risk-copilot.mlflow> |
| **Imagen del modelo** | `docker pull ghcr.io/diego-ballesteros/credit-risk-copilot/model:latest` |
| **Imagen del copiloto** | `docker pull ghcr.io/diego-ballesteros/credit-risk-copilot/agent:latest` |

Los cuatro experimentos son `credit-risk-baselines`, `credit-risk-retrieval`,
`credit-risk-agent` y `credit-risk-online`, y `docs/EVALUATION.md` cita el identificador de
cada run. **Usa el enlace terminado en `.mlflow`**: la página del repositorio en DagsHub no es
la interfaz de MLflow.

**Por dónde empezar a leer:** el `README.md` es el entregable y se lee entero;
`docs/MODEL_CARD.md` para qué hace el modelo y para qué no debe usarse; `docs/EVALUATION.md`
para el protocolo de cada medición; `docs/adr/` para por qué el proyecto es como es.

---

## Qué queda fuera del alcance, y por qué

**Nada de esto está pendiente. Son decisiones, y se registran para que nadie las confunda con
olvidos.**

| Fuera de alcance | Razón |
| --- | --- |
| **Reto ML 2** — Azure Container Apps y Terraform | **No intentado.** Al llegar a la fase 4 quedaban dos días, y la regla de corte estaba fijada de antemano: nunca sacrificar el 100% obligatorio por el 20% extra. Su mitad de CI/CD sí se hizo, dentro del Reto ML 1 |
| **Boxplots en el EDA univariado** | Los outliers ya están caracterizados por otra vía, y la decisión que dependía de ellos —`RobustScaler`— se tomó midiendo asimetría y curtosis |
| **Variante L1 de la logística** | La logística es el baseline, no el candidato. L1 aportaría selección de features sobre un modelo ya descartado |
| **Intervalos de confianza** | Las decisiones se tomaron contra un umbral de significancia práctica fijado antes de ver resultados, y las comparaciones que importaban son **pareadas fold a fold** |
| **Análisis de errores dedicado** | Lo que habría destapado —trato distinto por grupo— está medido con más rigor en el análisis de equidad |
| **`docs/ARCHITECTURE.md`** | El README cubre el contrato de comportamiento con sus dos diagramas; un documento aparte duplicaría contenido contra la regla de no solapamiento |
| **Mitigación de la disparidad medida** | Reponderar, umbralizar por grupo o quitar variables son decisiones con alternativas y costos. Ninguna se tomó, y cegar el modelo **ya está descartado por medición** |

Los detalles están en la sección 11 bis de `docs/ROADMAP.md`.

---

## Qué haría falta para un despliegue real

En orden de importancia, y ninguno es trabajo de modelado:

1. **Un conjunto de validación posterior en el tiempo.** Es la limitación bloqueante y no se
   puede construir con estos datos.
2. **Datos de exposición, recuperación y margen** para medir el cociente de costos en vez de
   declararlo. Mueve más negocio que cualquier mejora del modelo.
3. **Una decisión explícita sobre la disparidad medida**: mitigarla, aceptarla por escrito, o
   no desplegar.
4. **Revisión humana obligatoria.** Con el umbral de 0,160, 4 de cada 10 rechazos aciertan.
5. **Cerrar la grieta de las constantes normativas sin cita**, recuperándolas por identificador
   o marcándolas como no citables.
6. **Un holdout limpio**, para que la degradación en el flujo se pueda medir en vez de leerse
   como optimismo.
