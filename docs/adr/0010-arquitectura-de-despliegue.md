# ADR 0010 — Arquitectura de despliegue: dos imágenes y una carga que no bloquea

- **Status:** Accepted
- **Date:** 2026-08-31

---

## Contexto

El sistema tiene dos componentes con perfiles de dependencia muy distintos. El modelo
necesita scikit-learn y sus artefactos; el copiloto arrastra `torch` y una base de datos
vectorial por el modelo de embeddings. La diferencia no es una impresión: **importar el
módulo de herramientas del copiloto carga 7.697 módulos, frente a los 3.661 que carga la
aplicación del modelo**, y las dependencias exclusivas del copiloto suman unos **604 MB**
medidos sobre el entorno de desarrollo —`torch` 468 MB, `chromadb` 64 MB, `transformers`
52 MB, `langgraph` 7,6 MB, `anthropic` 7,3 MB, `sentence-transformers` 4,6 MB—.

Las tres decisiones de abajo se registran juntas porque responden a la misma pregunta
formulada sobre superficies distintas: **qué parte del sistema paga el costo de otra.** La
primera lo pregunta sobre la imagen, la segunda sobre el grafo de imports, y la tercera
sobre el tiempo de arranque.

---

## Decisión

### 1 · El sistema se despliega como dos servicios con imágenes separadas

**Razón:** la aplicación que devuelve una probabilidad no debe arrastrar un framework de
deep learning para hacerlo. Un cambio en el modelo de embeddings no puede ser motivo para
volver a desplegar el que puntúa clientes, y una petición de `P(default)` no debe pagar por
un transformer que nunca llama.

**La separación no es una intención: está garantizada por un test.**
`tests/test_api.py::test_model_app_does_not_import_the_agent_stack` lanza un **subproceso
limpio**, importa allí la aplicación del modelo y verifica la ausencia de los cinco módulos
prohibidos en `sys.modules`. Tiene que ser un subproceso porque dentro del proceso de pytest
los tests del copiloto ya los importaron. Eso sitúa la garantía en el **nivel 2** de la
jerarquía de la sección 6.5 de `docs/METHODOLOGY.md` —un test lo detecta— y no en el nivel 3,
donde solo se cumple mientras alguien lo recuerde.

El workflow `.github/workflows/docker.yml` repite la misma prueba **dentro de la imagen
construida**, que es donde importa de verdad, y añade la afirmación complementaria: la
imagen del agente sí importa los cinco. Un import de conveniencia que reintrodujera el
acoplamiento rompe el build antes de que se publique nada.

Para que la separación sea instalable y no solo importable, `pyproject.toml` se parte en un
**set base** —lo que necesita el servicio del modelo— y dos **grupos de dependencias**
(PEP 735), `agent` y `research`. Resuelto sobre el lockfile: **103 paquetes contra 195**, y la
imagen del modelo evita además los **19 paquetes de CUDA y NVIDIA** que `torch` arrastra en
Linux.

Grupos y no `optional-dependencies`, y la razón se midió en vez de suponerse: los extras no
los instala un `uv sync` pelado, y esta versión de `uv` no tiene ajuste `default-extras`
—verificado: declarados como extras, `uv sync` desinstaló todo el stack del copiloto—. Eso
habría roto en silencio cada comando de reproducción que `docs/EVALUATION.md` documenta con
`uv run python scripts/...`. Los grupos sí tienen `tool.uv.default-groups`, así que el entorno
de un contribuidor queda completo y **cada imagen se sale de esa comodidad con
`--no-default-groups`**.

#### Alternativa descartada: una sola imagen

Es más simple de operar: un Dockerfile, un tag, un despliegue. Se descarta porque su tamaño
estaría dominado por dependencias que la mitad del sistema no usa, y porque el acoplamiento
volvería a ser invisible —nada fallaría el día que el módulo del modelo importara algo del
copiloto—.

### 2 · El vocabulario de decisión y el contrato del solicitante viven en el paquete de modelos y no en el del copiloto

**Razón:** `ApplicantRecord`, el umbral operativo y las frases que viajan con toda
probabilidad —`COST_ASSUMPTION`, `DECISION_CAVEAT`, `CAUSAL_NOTE`, `DIRECTION_NOTE`— eran
definiciones **del dominio** alojadas en un módulo que importa el cliente del modelo de
lenguaje y la base vectorial. Cualquier consumidor que las necesitara arrastraba todo ese
stack, de modo que la decisión 1 era inalcanzable mientras vivieran ahí.

**Alternativa descartada:** duplicar los textos en la capa de API con un test de deriva. Es
el patrón que el proyecto ya usa para `ApplicantRecord` frente a `schema.py` y para
`POLICY_BANDS` frente al corpus, así que era coherente. Se descarta porque **una copia de
texto normativo mantenida a mano es exactamente el modo de falla que la sección 11.4 ter de
`docs/MODEL_CARD.md` ya registra** como fallo medido del sistema: `DECISION_CAVEAT` transcribe
la sección 2.2 de la política interna y puede divergir de ella en silencio. Añadir una tercera
copia de esa transcripción para resolver un problema de empaquetado habría sido pagar el
problema conocido para evitar uno nuevo.

**El movimiento se verificó mecánicamente y queda registrado.** De las 108 líneas eliminadas
del módulo de origen, **99 reaparecen literalmente** en los módulos de destino; las **nueve**
restantes son estructurales —una línea de import, dos de comentario de sección, tres sitios de
llamada renombrados y las dos de una función privada que pasó a pública con docstring—.
**Ninguna frase de contenido normativo cambió**, y los **281 tests previos pasan sin
editarse**, porque el módulo de origen reexporta cada nombre bajo `__all__`.

Consecuencia menor y declarada: cuatro documentos —el ADR-0009, la entrada 012 de
`docs/EVALUATION.md`, `docs/analysis/agent-evaluation-evidence.md` y la sección 11.4 ter del
Model Card— describen `DECISION_CAVEAT` como *«una constante de `agent/tools.py`»*. Sigue
siendo cierto al nivel del import; lo que cambió es dónde está la asignación.

### 3 · El modelo se carga en segundo plano y el proceso arranca de inmediato

**Evidencia, medida y no supuesta.** Con el registro de modelos inalcanzable, la carga tarda
**263,2 segundos** porque el cliente de MLflow reintenta con retroceso exponencial. Ese tiempo
transcurría **dentro del ciclo de vida de arranque de la aplicación**, y un `lifespan` de ASGI
que no ha retornado impide que uvicorn acepte ninguna conexión: ni siquiera al endpoint de
salud. El síntoma es un servicio que no responde durante más de cuatro minutos **sin lanzar
ninguna excepción y sin escribir ninguna línea de error**.

**Consecuencia directa sobre el contenedor:** un healthcheck con un periodo de gracia menor
que el presupuesto de reintentos marcaría el servicio como no sano y lo reiniciaría, y el
reinicio reiniciaría la carga. El resultado es **un ciclo de reinicios causado por el propio
healthcheck**, cuyo síntoma visible sería *«el contenedor no arranca»* y no *«el registro no
responde»* —es decir, un diagnóstico que apunta al lugar equivocado—.

La carga pasa a un **hilo demonio**. El `lifespan` lo arranca y retorna. Medido después del
cambio, en las mismas condiciones y cronometrando **desde antes de lanzar el proceso**:
**4,3 segundos hasta la primera respuesta de `/health`**, contra los 263,2 segundos de la
carga. Y de esos 4,3 segundos, prácticamente todo es Python importando `mlflow`, `shap` y
`scikit-learn`: en el log, `load.started` y `Application startup complete` salen seguidos, así
que **la contribución del `lifespan` es de milisegundos**. El registro dejó de gobernar el
arranque.

Tres estados sustituyen al booleano anterior, porque colapsarlos pierde las dos preguntas que
un operador hace de verdad:

| Fase | `/health` | Endpoints que necesitan el artefacto | Qué significa |
| --- | --- | --- | --- |
| `loading` | 200, `status: starting` | 503 `model_loading` + `Retry-After` | **Todavía no.** Reintenta |
| `ready` | 200, `status: ok` | Responden | Listo |
| `degraded` | 200, `status: degraded` + motivo | 503 `model_unavailable` | **No se pudo.** Reintentar no ayuda |

La transición completa está observada de punta a punta contra un registro inalcanzable:
`loading` a los 4,3 s, `degraded` a los **262,7 s** con el motivo en `/health`; y contra el
registro real, `loading` inmediato y `ready` con `load_seconds: 12,784`.

**`/health` responde 200 en las tres fases y `/ready` es la compuerta.** Es el punto de la
decisión: el healthcheck del contenedor apunta a `/health`, de modo que **un servicio vivo
pero cargando no es un servicio no sano**. Si el healthcheck fuera la compuerta de
disponibilidad, cualquier orquestador que reinicie contenedores no sanos —Swarm, Kubernetes—
reintroduciría el ciclo de reinicios por otra vía.

**Un hilo y no una tarea de asyncio,** porque el trabajo es bloqueante y síncrono: una
descarga HTTP dentro del cliente de MLflow, una deserialización de `skops` y 300 árboles
recorridos por SHAP. En el bucle de eventos bloquearía cada petición durante ese tiempo, que
es el mismo error movido del arranque al régimen permanente.

**Ningún cerrojo protege el estado,** y no hace falta: el estado es un dataclass congelado, así
que el hilo publica un objeto nuevo y reasigna un atributo en vez de mutar lo que un manejador
está leyendo. La reasignación de un atributo es atómica bajo el GIL, de modo que un lector ve
el estado viejo completo o el nuevo completo. **La inmutabilidad es la sincronización.**

#### Alternativas descartadas

- **Acortar el presupuesto de reintentos de MLflow.** Mueve el umbral sin resolver el
  acoplamiento: el arranque sigue dependiendo de la disponibilidad del registro, solo que
  falla antes. Y elegir el nuevo valor es elegir a partir de qué latencia del registro el
  servicio deja de arrancar, que es una pregunta que no queremos tener que responder.
- **Fijar un periodo de gracia largo en el healthcheck.** Evita el ciclo de reinicios y hace
  que **un fallo real tarde el mismo tiempo en detectarse**: un proceso muerto al arrancar
  quedaría sin diagnosticar durante los mismos cuatro minutos y medio.
- **Fallar al arrancar si el registro no responde.** Es defendible y se descarta porque un
  contenedor que sale en el arranque reporta su fallo solo en los logs del orquestador,
  mientras que un proceso que sigue en pie responde *«¿por qué no se está puntuando?»* a
  cualquiera que alcance `/health`.

---

## Consecuencias

**Tamaño de las imágenes: NO MEDIDO en este turno.** La máquina de desarrollo **no tiene
daemon de Docker**, así que ninguna imagen se construyó localmente. El workflow `docker.yml`
publica el tamaño de cada una en el resumen de su ejecución, y esta sección se completa con
esas dos cifras cuando la primera ejecución termine. **Escribir aquí un número estimado sería
inventar una medición**, y la regla del proyecto es que una cifra que no se pudo medir se
declara como no medida.

Lo que sí está medido es **qué las domina**. Sobre el entorno de desarrollo (Windows):
`torch` 468 MB, `chromadb` 64 MB, `transformers` 52 MB, `langgraph` 7,6 MB, `anthropic`
7,3 MB y `sentence-transformers` 4,6 MB — unos **604 MB exclusivos del agente**. En la imagen
del modelo lo dominante es `numba`+`llvmlite` (131 MB, dependencia de SHAP), `scipy` (96 MB),
`pyarrow` (82 MB, requisito duro de MLflow), `mlflow` (47 MB), `pandas` (44 MB) y
`scikit-learn` (34 MB): **la imagen del modelo está dominada por su capacidad de explicar, no
por la de puntuar.**

**Y una advertencia sobre esas cifras: en Linux la del agente será bastante mayor.** La
resolución del lockfile muestra que `torch` arrastra allí **19 paquetes de CUDA y NVIDIA**
—`nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `triton` y otros— que en el entorno
de desarrollo no aparecen. Nada en este sistema usa una GPU: el modelo de embeddings corre en
CPU. Existe por tanto una palanca de tamaño del orden de un gigabyte, que es resolver `torch`
contra el índice de ruedas **solo-CPU**. **No se toma en este ADR**: cambia cómo se resuelve
una dependencia y es una decisión con alternativas, no un movimiento. Queda registrada aquí
para que la primera cifra de tamaño del workflow no se lea como inevitable.

**Lo que se pierde al separar.** Una llamada que antes era en proceso ahora cruzaría la red si
un servicio necesitara al otro. Hoy no ocurre —el copiloto carga el mismo artefacto del
registro en su propio proceso, que es lo que mantiene idéntico el número que reporta `/chat` y
el que devuelve `/predict`— pero el costo está latente: el día que el copiloto llame a
`/predict` en vez de puntuar por su cuenta, aparecen latencia de red, un modo de falla nuevo
(el servicio del modelo caído) y la necesidad de propagar el identificador de correlación
entre servicios. Se acepta a sabiendas y se registra para que el día que se pague no parezca
un descubrimiento.

**Dos artefactos se cargan dos veces.** El modelo del registro se descarga en ambos
contenedores, porque ambos lo necesitan. Es duplicación de descarga, no de versión: los dos
apuntan a `models:/credit-risk-default-probability/1`, que está fijado.

**Los artefactos pesados no viajan en las imágenes.** El modelo llega del MLflow Model
Registry en tiempo de ejecución; el corpus y el índice vectorial llegan **montados juntos**
desde el host. Juntos porque un identificador de fragmento resuelto contra una revisión del
corpus distinta de aquella con la que se construyó el índice devuelve el fragmento equivocado
**sin ningún error**, y ese desajuste silencioso es el que este proyecto se niega a hacer
posible. La consecuencia operativa es que el servicio del agente arranca `degraded` si el
índice no está construido, lo dice en `/health`, y responde 503 en `/chat`. No inventa un
índice.

**El árbol de fuentes debe quedarse en `/app/src`.** `config.PROJECT_ROOT` se deriva de la
ubicación del propio módulo, así que instalar el paquete en `site-packages` movería en
silencio cada ruta de datos del proyecto hacia dentro del entorno virtual. Las imágenes
instalan las dependencias con `--no-install-project` y exponen el código por `PYTHONPATH`. Es
una atadura real entre el empaquetado y `config.py`, y queda escrita porque no es evidente.

**La grieta de las constantes normativas sin cita sigue abierta.** La decisión 2 mueve
`DECISION_CAVEAT` y sus tres hermanas; **no las cierra**. Siguen llegando al analista sin
`chunk_id`, ahora también por HTTP, y el ADR-0009 sigue siendo el registro de esa decisión
pendiente.
