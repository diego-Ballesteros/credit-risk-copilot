# Evidencia — Métricas online del sistema desplegado

Medición de registro. Describe **qué se midió y qué salió**. No decide nada: qué hacer con
una latencia de 527 ms o con un PR-AUC en muestra es una decisión con alternativas, y vive
en la entrada 013 de `docs/EVALUATION.md`.

- **Fecha:** 2026-08-31
- **Reproducción:**
  `uv run uvicorn credit_copilot.api.model_app:app --port 8000` y en otra terminal
  `uv run python scripts/run_online_simulation.py`;
  `uv run uvicorn credit_copilot.api.agent_app:app --port 8001` y
  `uv run python scripts/run_agent_latency.py --n 5`
- **Objeto medido:** el **sistema desplegado**, no el modelo. Las peticiones cruzan HTTP y
  pagan el parseo de JSON, las 23 validaciones Pydantic, la verificación del contrato de
  datos, el middleware de correlación y la serialización de la respuesta.
- **Dónde corrió:** **uvicorn en la máquina de desarrollo**, no en el contenedor. Esa máquina
  no tiene daemon de Docker (ver el reporte del turno 2 de la fase 4), así que las cifras
  describen este proceso en este equipo y **no son una predicción del contenedor**.
- **Runs de MLflow:** experimento `credit-risk-online`.
  - `online-simulation`, concurrencia 8, n=3.000: `3c2b27398e6b42deae77e6b923169d40`
  - `agent-latency`, n=5: `967e9d209f714720b775763438f9dcce`

---

## 0 · El flujo NO es un holdout limpio, y eso condiciona una de las cinco métricas

El artefacto productivo se ajustó sobre **las 30.000 filas**. No es una inferencia: el script
que lo registró deja la afirmación en un tag del run, y `run_online_simulation.py` **lo lee**
antes de reportar nada.

```
registry tag `fitted_on` = 'all 30000 rows - artefact, not an estimate'
```

Consecuencia, tabla por tabla:

| Métrica | ¿Afectada por la contaminación? | Por qué |
| --- | --- | --- |
| Latencia | **No** | El tiempo de servir una fila no depende de si el modelo la vio |
| Throughput | **No** | Ídem |
| Tasa de error | **No** | La rechaza el contrato de entrada, antes del modelo |
| Drift por feature | **Degenerada** | El flujo es una submuestra de la referencia: PSI ≈ 0 por construcción |
| **PR-AUC del flujo** | **Sí, y de forma decisiva** | Es una medición de memorización, no de desempeño |

---

## 1 · Latencia y throughput del servicio del modelo

Tres puntos de operación sobre el mismo servicio y el mismo artefacto. **Ninguna cifra de
latencia es interpretable sin la concurrencia y el número de workers a su lado.**

| Despliegue | Concurrencia | n | p50 | p95 | p99 | máx | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 worker | 1 | 500 | **75,6 ms** | 84,1 ms | 91,3 ms | 193,3 ms | 14,1 req/s |
| 1 worker | 8 | 3.000 | 527,4 ms | 643,2 ms | 692,4 ms | 870,4 ms | 15,8 req/s |
| **4 workers** | 8 | 3.000 | **152,3 ms** | 247,1 ms | 279,6 ms | 633,1 ms | **53,2 req/s** |

### Peticiones rechazadas, medidas por separado

| Despliegue | Concurrencia | p50 | p95 | p99 | máx |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 worker | 1 | **2,0 ms** | 2,5 ms | 2,5 ms | 2,5 ms |
| 1 worker | 8 | 68,5 ms | 136,6 ms | 180,9 ms | 222,8 ms |
| 4 workers | 8 | 2,9 ms | 33,3 ms | 42,2 ms | 57,3 ms |

**A concurrencia 1 una petición rechazada cuesta 2,0 ms contra 75,6 ms de una válida: 38 veces
menos.** El contrato la rechaza antes de tocar el artefacto. A concurrencia 8 con un worker
sube a 68,5 ms, que es cola detrás de peticiones válidas y no trabajo propio.

### Lectura de las tres filas

De 1 a 8 en concurrencia con un worker, **el throughput no se mueve** —14,1 a 15,8 req/s— y la
latencia se multiplica por 7. Es la firma de un cuello serializado: cada petición añadida
espera, ninguna añade capacidad. El tiempo de servicio real es el p50 de la primera fila,
**75,6 ms**; los 527 ms de la segunda son 451 ms de cola.

Con **4 workers** el throughput sube a **53,2 req/s (3,4×)** y el p50 baja a 152 ms. El cuello
era el proceso único, no el modelo. Costo declarado: **cada worker carga su propia copia del
artefacto** —cuatro descargas del registro y cuatro bosques de 300 árboles en memoria—,
verificado en el log de arranque (`model.loaded` aparece 4 veces).

---

## 2 · Tasa de error, desglosada por tipo

El flujo lleva una fracción **declarada** de peticiones malformadas (`--invalid-fraction 0.05`).
Sin ella la tasa de error es 0% y el desglose está vacío, que no mide nada.

| | n=3.000, concurrencia 8 |
| --- | ---: |
| Peticiones válidas | 2.868 |
| Peticiones deliberadamente malformadas | 132 |
| **Tasa de error total** | **4,4000%** (132 de 3.000) |
| **Fallos inesperados** (solicitante válido que no puntuó) | **0** |
| `invalid_request` (422) | 132 |
| Fallos de transporte | 0 |
| Timeouts | 0 |

Los cuatro tipos inyectados —campo faltante, nulo explícito, valor fuera de rango, categoría
desconocida— **se rechazaron los 132, con 422, y ninguno devolvió una probabilidad**. La tasa
de error del sistema sobre tráfico válido es **0%** en 2.868 peticiones.

---

## 3 · Discriminación sobre el flujo servido

**n = 2.868 filas puntuadas.**

| Métrica | Flujo servido (en muestra) | Referencia offline (CV 5 folds) | Diferencia |
| --- | ---: | ---: | ---: |
| **PR-AUC** | **0,672144** | **0,564230 ± 0,007962** | **+0,107914** |
| ROC-AUC | 0,844692 | 0,786279 | +0,058413 |
| Brier *(menor es mejor)* | 0,117997 | 0,133408 | −0,015411 |
| precision@top-10% | 0,773519 | 0,706333 | +0,067186 |

**Las cuatro se mueven en la dirección de "mejor", y por eso ninguna es un resultado.** El
signo es la evidencia de contaminación: un flujo con degradación real habría dado PR-AUC
*menor* que 0,564. El artefacto vio estas 2.868 filas durante el ajuste, así que +0,1079 mide
**optimismo por memorización**, no desempeño. El script lo etiqueta así en el reporte y en el
tag `pr_auc_gap_reading = optimism` del run de MLflow.

La magnitud es acotada y vale la pena registrarla: **+0,108 sobre 0,564 es un 19% de
optimismo**, no un colapso. Es consistente con un bosque de `max_depth=10` y
`min_samples_leaf=18`, que está regularizado y no memoriza fila a fila.

---

## 4 · Drift por feature entre entrenamiento y flujo servido

**Prueba:** Population Stability Index sobre 10 bins por cuantiles de la **referencia**, con
Kolmogorov-Smirnov de dos muestras al lado. Las columnas categóricas se comparan sobre los
niveles que declara `schema.py` más los códigos que aceptó el ADR-0004, no sobre cuantiles de
un código arbitrario.

**Umbrales, declarados y no derivados de estos datos:** PSI < 0,10 ruido; 0,10–0,25 moderado;
≥ 0,25 accionable.

Referencia n=30.000, flujo n=3.000.

| Feature | PSI | Banda | KS | KS p | media ref. |
| --- | ---: | --- | ---: | ---: | ---: |
| PAY_STATUS_2 | 0,0092 | none | 0,0073 | 9,99e-01 | −0,1 |
| PAY_STATUS_1 | 0,0069 | none | 0,0063 | 1,00e+00 | −0,0 |
| PAY_AMT4 | 0,0039 | none | 0,0135 | 6,98e-01 | 4.826,1 |
| BILL_AMT1 | 0,0036 | none | 0,0159 | 4,93e-01 | 51.223,3 |
| PAY_AMT3 | 0,0033 | none | 0,0132 | 7,24e-01 | 5.225,7 |
| PAY_AMT2 | 0,0032 | none | 0,0223 | 1,31e-01 | 5.921,2 |
| BILL_AMT6 | 0,0032 | none | 0,0150 | 5,69e-01 | 38.871,8 |
| LIMIT_BAL | 0,0026 | none | 0,0123 | 7,99e-01 | 167.484,3 |
| AGE | 0,0024 | none | 0,0111 | 8,84e-01 | 35,5 |
| BILL_AMT3 | 0,0023 | none | 0,0188 | 2,87e-01 | 47.013,2 |
| EDUCATION | 0,0021 | none | 0,0130 | 7,44e-01 | 1,9 |
| PAY_STATUS_4 | 0,0021 | none | 0,0118 | 8,35e-01 | −0,2 |
| PAY_AMT1 | 0,0018 | none | 0,0129 | 7,47e-01 | 5.663,6 |
| BILL_AMT4 | 0,0018 | none | 0,0135 | 6,95e-01 | 43.262,9 |
| PAY_STATUS_3 | 0,0017 | none | 0,0072 | 9,99e-01 | −0,2 |
| PAY_STATUS_6 | 0,0017 | none | 0,0045 | 1,00e+00 | −0,3 |
| BILL_AMT5 | 0,0015 | none | 0,0124 | 7,94e-01 | 40.311,4 |
| PAY_AMT6 | 0,0014 | none | 0,0067 | 1,00e+00 | 5.215,5 |
| PAY_STATUS_5 | 0,0014 | none | 0,0118 | 8,35e-01 | −0,3 |
| BILL_AMT2 | 0,0013 | none | 0,0129 | 7,49e-01 | 49.179,1 |
| PAY_AMT5 | 0,0011 | none | 0,0108 | 9,03e-01 | 4.799,4 |
| MARRIAGE | 0,0010 | none | 0,0072 | 9,99e-01 | 1,6 |
| SEX | 0,0001 | none | 0,0046 | 1,00e+00 | 1,6 |

**Máximo PSI = 0,0092**, 11 veces por debajo del umbral de ruido. Ninguna feature señala.

**Qué significa y qué NO significa.** El flujo es una submuestra aleatoria de la población de
entrenamiento, así que PSI ≈ 0 **es lo esperado por construcción**. Esta corrida es un
**control negativo del instrumento**: muestra que el detector no dispara sobre distribuciones
idénticas. **No es evidencia de que el tráfico de producción no vaya a driftar**, porque no
existe aquí una segunda población contra la cual medirlo.

Que el instrumento sí dispara está verificado aparte, en `tests/test_drift.py`, con controles
positivos: desplazar la media 0,8 σ da PSI ≥ 0,25, desplazarla 2 σ da PSI > 1,0, y el PSI es
monótono en el tamaño del desplazamiento. Un detector que solo se prueba contra el caso sin
señal no está probado.

---

## 5 · Latencia y costo del copiloto

**n = 5 consultas**, las primeras cinco de `data/eval/agent_queries.yaml` en orden de archivo,
enviadas **de a una** a `POST /chat`. No se llamó al juez: la calidad es la entrada 012.

| Consulta | s | llamadas LLM | ciclos | tools | citas | tok. entrada | tok. salida | USD | outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a01 | 39,52 | 3 | 1 | 2 | 5 | 19.538 | 2.726 | 0,1477 | answered |
| a02 | 27,09 | 3 | 1 | 1 | 5 | 19.252 | 1.868 | 0,1254 | answered |
| a03 | 40,35 | 3 | 1 | 2 | 6 | 19.895 | 2.691 | 0,1456 | answered |
| a04 | 30,81 | 3 | 1 | 1 | 5 | 19.665 | 2.281 | 0,1372 | answered |
| a05 | 47,67 | 5 | 2 | 3 | 5 | 30.664 | 3.268 | 0,1949 | answered |

| | |
| --- | ---: |
| Completadas | 5 de 5 |
| Latencia mínima | 27,09 s |
| **Latencia mediana** | **39,52 s** |
| Latencia máxima | 47,67 s |
| Costo total | 0,7509 USD |
| **Costo medio por consulta** | **0,1502 USD** |
| Llamadas al LLM, media | 3,40 (techo 2·3+1 = 7) |
| Citas, media | 5,20 |

**n=5 no permite estimar un percentil**, así que se reportan el mínimo, la mediana y el máximo
y no un p95. El costo es una **estimación** a partir de una tabla de precios transcrita en
`scripts/run_agent_latency.py`, no una factura.

### Contraste con la entrada 012

| | Entrada 012 (19 consultas) | Esta corrida (5 consultas) |
| --- | ---: | ---: |
| Llamadas al LLM, media | 5,11 | 3,40 |
| Costo por consulta | 0,209 USD | 0,150 USD |

**Las cinco primeras consultas son más baratas que la media de las diecinueve porque usan
menos ciclos**: cuatro de las cinco convergieron en un solo ciclo de planificación (3 llamadas)
y solo `a05` usó el segundo (5 llamadas). La entrada 012 promedia sobre las diecinueve, que
incluyen las consultas donde el evaluador pidió otra vuelta. **No es una mejora del sistema: es
un subconjunto distinto**, y por eso las dos cifras se reportan una al lado de la otra en vez
de sustituir la anterior.

---

## 6 · El arranque de cada servicio, medido de paso

| Servicio | `load_seconds` | Qué carga |
| --- | ---: | --- |
| Modelo | 13,341 | Artefacto del registro + `TreeExplainer` sobre 300 árboles |
| Copiloto | 20,670 | Lo anterior + modelo de embeddings + índice Chroma |

Ninguno bloquea la aceptación de conexiones: ambos responden `/health` con `phase: loading`
desde el primer segundo (ADR-0010, decisión 3).
