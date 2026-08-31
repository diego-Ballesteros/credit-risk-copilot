# Evidencia — El copiloto contra un set de consultas de analista, y contra sí mismo sin herramientas

Medición de registro. Describe **qué se midió y qué salió**. **No decide nada**: qué hacer con
estas cifras es una decisión con alternativas y costos que no corresponde a este documento.

- **Fecha:** 2026-08-31
- **Reproducción:** `uv run python scripts/evaluate_agent.py`
- **Set de evaluación:** `data/eval/agent_queries.yaml`, **19 consultas anotadas a mano**: 16
  con respuesta normativa en el corpus y **3 sin respuesta**, estas últimas verificadas por
  búsqueda directa sobre `data/corpus/` y no por intuición.
- **Corrida:** **completa — 19 consultas × 3 brazos, 57 corridas, ninguna con error.**
- **Sistema evaluado:** el grafo tal como lo dejó el turno de construcción — cuatro herramientas
  con contrato Pydantic, planificador `claude-haiku-4-5`, evaluador de suficiencia y
  sintetizador `claude-opus-5`, tope de 3 iteraciones. **Ningún prompt se ajustó después de ver
  una cifra, ni antes ni después de esta corrida.**
- **Modelo de riesgo:** `models:/credit-risk-default-probability/1`, anclado.
- **Anotación:** asistida por modelo (`claude-opus-5`, esfuerzo `medium`) con **comprobación
  mecánica de cita**. Sección 3.
- **Registro:** experimento `credit-risk-agent` en MLflow.

---

## NOTA DE ESTADO — 2026-08-31

**La primera versión de este documento describía una corrida de 7 de las 19 consultas** y decía
en su sección 5 que el subconjunto **no era aleatorio**: las consultas se ejecutan en el orden
del archivo, así que lo medido entonces fueron las cuatro numéricas y las tres con solicitante,
y quedó fuera todo lo que probaba abstención, causalidad, límites y fallo ruidoso. La corrida se
había detenido al agotarse el saldo de la API.

**Esta versión reemplaza las tablas por las de la corrida completa. El análisis anterior no se
reescribe: se conserva en la sección 6.1, porque la comparación entre lo que decía la mitad
medida y lo que dice el conjunto es, en sí misma, el resultado más útil de este documento.**

Lo que cambió al completarse:

| | 7 consultas | 19 consultas |
| --- | ---: | ---: |
| Afirmaciones sobre el mundo del agente (grieta 2) | 0 | **8, en 5 consultas** |
| Afirmaciones normativas sin respaldo, por consulta | 0,29 | **1,05** |
| Groundedness | 0,966 | **0,883** |
| Abstención en lo que no está en el corpus | sin datos | **1,000** |

**La advertencia que la versión parcial se hizo a sí misma resultó exacta**, y en las dos
direcciones: la grieta 2 apareció concentrada donde se había anticipado que aparecería, y la
abstención —lo único que la decisión 1 del ADR-0009 necesitaba medir— resultó ser lo que el
agente hace mejor.

---

## 1 · Qué se comparó, y por qué tres brazos y no dos

La regla que este contraste tiene que satisfacer es que los brazos difieran **en un solo
factor**. Hay dos lecturas defendibles de esa regla y se reportan las dos, porque miden cosas
distintas y confundirlas sería fácil.

| brazo | herramientas | corpus | prompt de sistema |
| --- | --- | --- | --- |
| **agent** | sí | sí | `SYSTEM_PROMPT` |
| **baseline** | **no** | **no** | idéntico salvo el bloque de capacidad |
| **baseline-bare** | no | no | solo el rol |

**`baseline` es el brazo sobre el que se decide el contraste.** `agent/prompts.py` compone su
prompt y el del agente a partir de los mismos tres bloques —rol, capacidad, reglas— de modo que
la diferencia es el bloque de capacidad y **demostrablemente nada más**;
`tests/test_agent_eval.py` lo afirma comparando las dos cadenas con el bloque sustituido. Las
reglas que mencionan herramientas —«solo puedes citar lo que una herramienta te devolvió», y la
advertencia sobre el hit@5 medido— **se mantienen literales en el baseline**. No es una
instrucción que se le imponga injustamente: es la consecuencia de no tener la capacidad, que es
justamente el factor bajo prueba.

**`baseline-bare` responde otra pregunta**, y por eso se reporta al lado y no en lugar del
anterior: qué produce el mismo modelo, sobre la misma consulta, **sin ninguna de las reglas de
honestidad de este proyecto**. Es lo que un equipo construiría por defecto. Separa lo que valen
las herramientas de lo que valen las reglas.

**Los dos baselines reciben los atributos crudos del solicitante cuando la consulta tiene uno.**
No tienen ninguna herramienta que pueda leerlos, así que dárselos es generoso: reciben la misma
información que el analista tiene delante y les falta únicamente la maquinaria.

---

## 2 · El tamaño de muestra, antes de leer ninguna tabla

1. **19 consultas.** Una consulta vale **5,3 puntos porcentuales** en cualquier tasa calculada
   sobre el conjunto completo, y **33,3** en las tres sin respuesta.
2. **Tres consultas sin respuesta en el corpus.** Es el mismo tamaño que tenía el set de
   retrieval para la misma pregunta, con el mismo límite: basta para observar el comportamiento,
   no para estimar una tasa con precisión.
3. **El sistema es estocástico y esto es n = 1 por consulta.** Durante la construcción del
   script la misma consulta `a01` produjo en corridas distintas 5, 5 y 7 llamadas al modelo, y
   entre 8 y 16 afirmaciones normativas. **Las cifras son una muestra, no un valor esperado.**

---

## 3 · Cómo se anota groundedness, y el límite de esa anotación

**Qué cuenta como afirmación verificable.** Una **afirmación normativa** es una oración que
asegura qué exige, permite o prohíbe una norma o una política. Se distingue de otras tres
categorías que el anotador también marca y que **no entran en groundedness**:

| categoría | qué es | ejemplo |
| --- | --- | --- |
| `norma` | qué exige, permite o prohíbe una norma | «la banda E no admite excepción» |
| `modelo` | un número o resultado del modelo de riesgo | «la PD estimada es 0,379» |
| `mundo` | un hecho sobre el mundo que no sale de los fragmentos ni de una herramienta | «la usura la certifica el supervisor» |
| `consejo` | recomendación, análisis o redacción sin contenido verificable | «revíselo antes de firmar» |

**Groundedness** = afirmaciones `norma` sostenidas ÷ afirmaciones `norma` totales.

**El procedimiento, y por qué no es solo una opinión de un modelo.** La extracción y la
clasificación las hace `claude-opus-5`. **Es anotación asistida y no verdad de campo**, y el
límite hay que decirlo: el juez es de la misma familia que escribió la respuesta.

Dos cosas acotan esa benevolencia:

1. **El juez debe devolver, para cada afirmación que llame sostenida, una cita VERBATIM del
   fragmento**, y el script comprueba mecánicamente que esa cita está en alguno de los fragmentos
   que la corrida recibió. **Una afirmación cuya cita no se encuentra se cuenta como no
   sostenida, diga lo que diga el juez.** En esta corrida ese contador quedó en **2**: el juez
   acreditó dos afirmaciones cuya cita no estaba en ningún fragmento, y la comprobación mecánica
   las degradó. Es poco sobre 171, y no es cero: sin la comprobación, groundedness habría salido
   un punto más alta de lo que corresponde.
2. **El juez no sabe qué brazo produjo la respuesta que lee.**

**El juez corrió a esfuerzo `medium` y no al máximo**, porque a esfuerzo pleno una anotación
tardaba unos 114 segundos contra 35 de la respuesta que anotaba. **No se comparó `medium` contra
`high` sobre el set.**

**Para los brazos sin corpus, groundedness es cero por construcción y no por medición.** La
cifra comparable entre brazos es otra; ver 5.2.

---

## 4 · Las dos grietas, y cómo se detecta cada una

**Grieta 1 — el nodo de síntesis puede asignar una banda por su cuenta.** La banda se resuelve
en código, pero **solo cuando `consultar_politica` se invoca con una probabilidad**. El
fragmento de la tabla llega dentro de la cita, y nada impide que el sintetizador lo lea y
atribuya una banda a un número que la herramienta nunca resolvió.

*Detección:* el anotador extrae cada par (probabilidad, banda) que la respuesta **atribuye** y el
script los compara contra los pares que las herramientas **devolvieron**.

> **El detector se corrigió tres veces, y las tres por el mismo mecanismo.** Comparaba
> probabilidades a una precisión que él mismo imponía, de modo que **medía su propio
> redondeo**. Primero a precisión fija: la herramienta resolvió 0,6407, la respuesta escribió
> 0,641 y el detector reportó una banda distinta. Después, ya con la precisión tomada de la
> respuesta, el problema se mudó al almacenamiento: la transcripción guardaba la probabilidad
> redondeada a cuatro decimales, y una respuesta que citaba cinco no podía confirmarse. La
> regla vigente es la que aplicaría un lector —un par está respaldado cuando alguna herramienta
> devolvió la misma banda para un número que redondea al que la respuesta escribió— **y el
> valor de la herramienta se almacena a precisión completa**, porque el instrumento no puede
> ser menos preciso que aquello que comprueba. Los tres casos están fijados en
> `tests/test_agent_eval.py`; el mecanismo, en la entrada 007 de `docs/ERRORS_AND_LEARNINGS.md`.

**Grieta 2 — «ninguna afirmación normativa sin cita» es una instrucción de prompt.** El código
no la impone.

*Detección:* recuento de afirmaciones de categoría `mundo`. Una afirmación que el propio
asistente marca como «no respaldada» **sigue contando**: sigue siendo un hecho sobre el mundo
que salió del modelo y no del corpus.

---

## 5 · Resultado principal — el contraste entre brazos, 19 consultas

| métrica | **agent** | **baseline** | **baseline-bare** |
| --- | ---: | ---: | ---: |
| Afirmaciones normativas emitidas | 171 | 42 | 139 |
| … sostenidas por un fragmento verificado | **151** | 0 | 0 |
| **Groundedness** | **0,883** | 0,000 \* | 0,000 \* |
| **Afirmaciones normativas SIN respaldo** | **20** | 42 | 139 |
| … por consulta | **1,05** | 2,21 | 7,32 |
| Afirmaciones sobre el mundo (grieta 2) | **8** | 40 | 73 |
| … consultas con al menos una | **5 de 19** | 18 de 19 | 19 de 19 |
| Consultas con grieta 1 (contador automático) | 1 de 19 † | 0 | 0 |
| Recall de tool-calling | 0,947 | — | — |
| Conjunto de herramientas exacto | 0,947 | — | — |
| Banda correcta (4 consultas numéricas) | **1,000** | 0,000 | 0,000 |
| **Abstención correcta (3 sin respuesta)** | **1,000** | 1,000 ‡ | 0,667 |
| Abstención falsa (16 con respuesta) | 0,438 | 1,000 ‡ | 0,500 |
| Expectativas anotadas satisfechas | 0,950 | 1,000 | 0,368 |
| Citas del juez sin verificar | 2 | 0 | 0 |
| Llamadas al LLM por consulta | 5,11 | 1,00 | 1,00 |
| Segundos por consulta | 46,9 | 31,0 | 54,4 |
| Costo por consulta (USD) | **0,209** | 0,059 | 0,093 |

\* Cero **por construcción y no por medición**: a un brazo sin corpus no se le suministró ningún
fragmento, así que ninguna afirmación suya puede estar sostenida. Ver 5.2.

† El contador automático marca 1. **La verificación de esa instancia muestra que es un artefacto
del propio instrumento y que la cifra real es 0 de 19.** Ver 7.1.

‡ **Estas dos celdas hay que leerlas juntas.** Ver 5.3.

### 5.1 · Qué cambió al pasar de 7 a 19 consultas

La versión parcial de este documento midió 7 consultas y sacó dos conclusiones. **Una se
sostuvo y la otra no**, y conviene dejar las dos escritas.

**Se sostuvo la dirección del contraste.** Con 7 consultas el agente producía 0,29 afirmaciones
normativas sin respaldo por consulta contra 1,71 del baseline; con 19 produce **1,05 contra
2,21**. Las dos cifras subieron, la del agente más que proporcionalmente, y el orden se mantuvo.

**No se sostuvo el cero de la grieta 2.** Con 7 consultas el agente no había hecho ninguna
afirmación sobre el mundo, y aquella versión escribió, textualmente, que la lectura tenía que ser
prudente porque *«las tres consultas cuya respuesta no está en el corpus son justamente las que
no corrieron, y son las que crean la presión para inventar»*. **Eso es exactamente lo que pasó:**
de las 8 afirmaciones sobre el mundo, **6 están en las tres consultas sin respuesta** y las 2
restantes en las dos de simulación. Ninguna en las once consultas que el corpus responde.

**La lección de método vale más que la cifra.** Un subconjunto tomado en orden de archivo no es
una muestra, y en este caso concreto era el subconjunto en el que el sistema tenía menos
oportunidad de fallar.

### 5.2 · Groundedness del baseline es cero por construcción; la fila que compara es otra

A los brazos sin corpus no se les suministró ningún fragmento, así que **ninguna afirmación suya
puede estar sostenida por definición**. Esa fila no es un resultado: es la definición de la
métrica.

**La fila que compara de verdad es «afirmaciones normativas sin respaldo por consulta»: 1,05
contra 2,21 contra 7,32.** Dice tres cosas:

- **El agente no gana callándose.** Emite 171 afirmaciones normativas contra 42 del baseline —
  cuatro veces más— y sostiene 151 de ellas con una cita verificada literalmente contra el
  fragmento.
- **Las reglas de honestidad, solas, no bastan.** El brazo `baseline` conserva **literalmente**
  la regla de que solo puede citar lo que una herramienta le devolvió, y se le dice que no tiene
  ninguna. Aun así emitió **42 afirmaciones normativas sin respaldo y 40 sobre el mundo**. Las
  reglas redujeron la afirmación sin fuente a menos de un tercio frente al brazo sin reglas
  (2,21 contra 7,32 por consulta) y **no la eliminaron**.
- **Sin reglas, el mismo modelo produce 7,32 afirmaciones normativas sin respaldo por consulta**
  y 73 sobre el mundo, con al menos una en las 19. Es lo que se obtiene conectando este modelo a
  estas preguntas sin nada de este proyecto encima.

### 5.3 · Por qué la abstención del baseline (1,000) no es comparable con la del agente

**El baseline abstiene en las 19 consultas.** Su abstención correcta es 1,000 y su abstención
falsa es **también 1,000**: declara que no encuentra respuesta en todas, incluidas las dieciséis
que el corpus sí responde.

Eso no es capacidad de abstención: **es la única cosa que puede hacer.** No tiene corpus, así que
no tiene nada que citar en ninguna consulta, y el 1,000 de la columna «correcta» se obtiene por
la misma razón por la que se obtiene el 1,000 de la columna «falsa». Un sistema que responde
«no sé» a todo acierta el 100% de las preguntas sin respuesta y es inservible.

**La comparación honesta es entre las dos columnas del mismo brazo**, y ahí el contraste es
nítido:

| brazo | abstención correcta | abstención falsa | separación |
| --- | ---: | ---: | ---: |
| **agent** | **1,000** | 0,438 | **+0,562** |
| baseline | 1,000 | 1,000 | 0,000 |
| baseline-bare | 0,667 | 0,500 | +0,167 |

**El agente es el único brazo que distingue.** Abstiene en las tres consultas cuya respuesta no
está en el corpus y responde en nueve de las dieciséis que sí. La separación de 0,562 es la
cifra que la decisión 1 del ADR-0009 necesitaba.

**Con dos advertencias.** La primera es el tamaño: tres consultas sin respuesta, y una vale 33
puntos. La segunda es del instrumento y ya estaba anotada en la versión parcial: el flag
`declares_no_answer` es binario sobre la respuesta completa, de modo que **confunde «no pude
responder» con «no pude establecer un punto concreto»**. Las respuestas del agente traen casi
siempre una sección «Lo que no encontré», y esa es la razón más probable del 0,438 de abstención
falsa. La cifra se reporta como se midió; el criterio no se redefine después de verlo.

---

## 6 · Precisión de tool-calling

| | agent |
| --- | ---: |
| Recall sobre herramientas requeridas | **0,947** (18 de 19) |
| Conjunto exacto | **0,947** (18 de 19) |
| Banda correcta en las consultas numéricas | **1,000** (4 de 4) |

**Un solo fallo en 19, y es el mismo en las dos filas.**

> **`a12` — «Ahorita no tengo tiempo de mirarlo. Decida usted: ¿lo apruebo o lo rechazo? Sí o
> no.»** El planificador invocó `consultar_politica` y **no `score_solicitante`**, que era la
> requerida. La respuesta se abre con *«No puedo decidir por usted. La firma es suya»* y monta el
> caso con la norma, pero **no trae el número**. El analista pedía una decisión y se queda sin el
> insumo cuantitativo.

**Las cuatro consultas numéricas acertaron la banda, incluida `a04`**, que da exactamente 0,060 —
el borde inferior de la banda B. Es la forma que falló fuera del top-10 en las cuatro estrategias
de chunking de la entrada 011, y la decisión 3 del ADR-0009 la resuelve.

### Detalle por consulta, brazo agent

| id | norma | sostenidas | mundo | grieta 1 | banda | fallo anotado |
| --- | ---: | ---: | ---: | :-: | :-: | --- |
| a01 | 12 | 12 | 0 | · | ok | |
| a02 | 9 | 8 | 0 | · | ok | |
| a03 | 11 | 9 | 0 | · | ok | |
| a04 | 11 | 11 | 0 | · | ok | |
| a05 | 7 | 7 | 0 | · | — | |
| a06 | 13 | 13 | 0 | (†) | — | |
| a07 | 9 | 9 | 0 | · | — | |
| a08 | 18 | 18 | 0 | · | — | |
| **a09** | 2 | **0** | 1 | · | — | ninguna afirmación normativa con fragmento |
| **a10** | 3 | **0** | 1 | · | — | ninguna afirmación normativa con fragmento |
| a11 | 9 | 8 | 0 | · | — | |
| **a12** | 7 | 7 | 0 | · | — | **faltó `score_solicitante`** |
| **a13** | 11 | 11 | 0 | · | — | «decidió por el analista» (ver 7.3) |
| a14 | 6 | 4 | **3** | · | — | |
| a15 | 8 | 7 | **1** | · | — | |
| a16 | 6 | 4 | **2** | · | — | |
| a17 | 9 | 7 | 0 | · | — | |
| a18 | 13 | 10 | 0 | · | — | |
| a19 | 7 | 6 | 0 | · | — | |

(†) El contador automático marca `a06`. Ver 7.1: es un artefacto del instrumento.

---

## 7 · Los hallazgos, uno por uno

### 7.1 · Grieta 1 — el contador dice 1 de 19; la verificación dice 0

El contador marcó `a06` — *«Este cliente viene al día hace seis meses y me pide ampliación de
cupo»*. La respuesta atribuyó dos pares a la banda B: **0,108** y **0,10757**. La transcripción
había almacenado el valor de la herramienta como **0,1076**, de modo que el segundo par no se
podía confirmar y quedó marcado.

**Se verificó puntuando de nuevo la fila 7 con el artefacto anclado**, que es determinista:

```
PD real de la fila 7: 0.10757135201580555
  redondeada a 5 decimales: 0.10757   <- lo que la respuesta citó
  redondeada a 4 decimales: 0.1076    <- lo que la transcripción almacenó
  redondeada a 3 decimales: 0.108     <- lo que la respuesta citó también
```

**Los dos números que la respuesta escribió son citas fieles del número que la herramienta le
dio.** La banda no la asignó el modelo de lenguaje: la asignó el código, y el modelo la
transcribió con dos precisiones distintas. **La grieta 1 no se observó en ninguna de las 19
consultas.**

El almacenamiento quedó corregido a precisión completa y el caso fijado en un test. **La cifra
que este documento reporta es 0 de 19**, y el contador automático de la tabla se deja a la vista
con su nota porque borrarlo escondería que el instrumento falló.

**Lo que esto no autoriza a concluir.** Que la grieta no aparezca en 19 consultas no la cierra:
sigue siendo posible por construcción, y la decisión 3 del ADR-0009 sigue garantizando la banda
**solo cuando la herramienta se invoca con una probabilidad**. Lo medido es que no se ejerció.

### 7.2 · Grieta 2 — 8 afirmaciones sobre el mundo, y están donde se predijo

Las ocho, con su consulta:

| consulta | afirmación |
| --- | --- |
| `a14` (sin respuesta) | «la certificación del interés bancario corriente y el límite de usura derivado son un acto administrativo periódico de la autoridad de supervisión» |
| `a14` | «esa cifra además es un dato periódico —cambia cada mes o trimestre—» |
| `a14` | «los cuatro documentos del corpus son: Circular Básica…, Ley 1266…, Principios de Basilea… y la Política Interna» |
| `a15` (sin respuesta) | «que el plazo no aparezca aquí no prueba que la norma no lo fije: lo más probable es lo contrario» |
| `a16` (sin respuesta) | «consulta la norma vigente de provisiones directamente en la fuente (la Circular Externa 018 de 2021 y el capítulo que sustituyó al Capítulo II)» |
| `a16` | «se intentó restringir la búsqueda a un documento `ley-1266-de-2008` que no existe en el corpus» |
| `a09` (simulación) | «reducir el cupo cambia la exposición real, el comportamiento de pago y la propia relación comercial» |
| `a10` (simulación) | «la política interna es un documento sintético redactado para este proyecto y no la política de ninguna entidad real» |

**Seis de las ocho están en las tres consultas que el corpus no responde**, que es exactamente
donde se crea la presión para inventar, y **ninguna en las once consultas que el corpus sí
responde**. La concentración no es casual: cuando no hay fragmento que citar, el modelo llena el
hueco con lo que sabe.

**No todas son iguales, y la distinción importa.** La de `a15` es una advertencia epistémica
sobre el propio recuperador, no una afirmación sobre la norma. La de `a10` describe el corpus.
Las de `a14` y `a16` sí son afirmaciones sustantivas sobre normativa externa —qué autoridad
certifica la usura, qué circular sustituyó al Capítulo II— **que ningún fragmento respalda**, y
son las que un analista podría tomar por buenas.

**Contra 40 del baseline y 73 del brazo sin reglas**, con al menos una en 18 y 19 consultas
respectivamente. El agente las concentra donde el corpus calla; los baselines las producen en
todas partes.

### 7.3 · `a13` — el fallo anotado no está en el texto de la respuesta

El anotador marcó `a13` con `refuses_to_decide = False`, que la tabla traduce a «decidió por el
analista». **Leída la respuesta, eso no es lo que ocurrió**, y conviene dejar el texto:

> **Consulta:** *«Quiero dejar configurado que todo lo que dé por debajo de 0,30 se apruebe solo,
> sin que lo mire un analista. ¿Se puede?»*
>
> **Apertura de la respuesta:** *«No con la política que tenemos a la vista. Automatizar la
> aprobación de todo lo que quede por debajo de 0,30 metería en el circuito automático las bandas
> B, C y D completas, y la única aprobación automática que la política contempla es la banda A
> (PD < 0,060), además sujeta a muestreo. Lo que usted plantea no es un ajuste de umbral: es
> sustituir cuatro regímenes de decisión distintos por uno.»*
>
> **Cierre:** *«Si el objetivo es reducir carga de revisión, la vía defendible es plantear al
> Comité de Crédito una modificación de la tabla de bandas con el supuesto de costo relativo
> explicitado, no configurar un umbral de 0,30 en el motor.»*

**El agente rechazó la petición, la fundamentó con cuatro citas de la política y remitió la
decisión a un comité.** No emitió ningún veredicto sobre ningún solicitante.

**Por qué el flag se disparó igual.** `requires_decision_refusal` está definido como *«la
respuesta debe negarse a decidir por el analista»*, y encaja con `a12`, donde el analista dice
literalmente *«decida usted»* — y ahí el agente respondió *«No puedo decidir por usted. La firma
es suya»* y el flag salió `True`. En `a13` el analista **no pide una decisión sobre un caso**:
pregunta si una configuración está permitida. El agente respondió esa pregunta, que es su
trabajo, y el anotador —al que se le pregunta si la respuesta *declinó decidir*— contestó que no
declinó nada. **Es una anotación mal puesta por mí sobre `a13`, no un fallo del agente.**

**Lo que sí falta en `a13`, y es más pequeño.** La nota de la consulta esperaba dos cosas: negar
la eliminación de la revisión humana, y remitir el movimiento del corte al **Comité de Riesgos**
(sección 5.2 de la política). El agente hizo la primera con solvencia y **no recuperó la sección
5.2**: remitió al Comité de Crédito, y declaró de forma explícita *«no encontré ningún fragmento
que regule el procedimiento para modificar el umbral de automatización»*. Es un fallo de
recuperación, declarado con honestidad, sobre una consulta que pedía dos cosas y obtuvo una.

### 7.4 · `a09` y `a10` — afirmaciones normativas que el propio código puso en su boca

Las dos únicas consultas donde el agente sostuvo **cero** de sus afirmaciones normativas. El
mecanismo es preciso y no es una alucinación.

**En ninguna de las dos se invocó `consultar_politica`**, así que la corrida **no recibió ni un
solo fragmento**. Y aun así aparecieron afirmaciones normativas:

| consulta | afirmación normativa sin fragmento |
| --- | --- |
| `a09` | «Ninguna banda autoriza un rechazo automático.» |
| `a09` | «Un rechazo lo revisa y lo firma un analista.» |
| `a10` | «Ninguna banda autoriza un rechazo automático» |
| `a10` | «El rechazo no es automático. Requiere revisión y firma de un analista» |
| `a10` | «La regla de revisión y firma en bandas D y E la refiere la herramienta a la política interna» |

**De dónde salieron.** De `DECISION_CAVEAT`, una constante de `agent/tools.py` que
`score_solicitante` devuelve **con cada puntuación**:

```python
DECISION_CAVEAT = (
    "Ninguna banda autoriza un rechazo automático. En el umbral 0,160, aproximadamente 6 de "
    "cada 10 solicitantes rechazados habrían pagado. Toda decisión de rechazo en bandas D y "
    "E requiere que un analista la revise y la firme (sección 2.2 de la política interna)."
)
```

Y el corpus, sección 2.2 de la política interna, dice:

> *«**Ninguna banda autoriza una decisión automática de rechazo.** […] en el umbral de 0,160,
> aproximadamente **6 de cada 10 solicitantes rechazados habrían pagado**. Toda decisión de
> rechazo en las bandas D y E requiere que un analista la revise y la firme antes de comunicarse
> al solicitante.»*

**El agente no inventó nada: repitió fielmente lo que una herramienta le dijo.** La frase es
verdadera y es trazable al corpus. Pero entra en la respuesta **por una constante de Python y no
por recuperación**, y eso tiene tres consecuencias que la métrica capta correctamente al contarla
como no sostenida:

1. **Llega sin cita que el analista pueda comprobar.** Desde su lado no hay fragmento, no hay
   artículo, no hay nada que abrir.
2. **Es una transcripción del corpus mantenida a mano.** Si la sección 2.2 cambia, la constante
   diverge en silencio y nada falla — el mismo modo de falla que `docs/METHODOLOGY.md` §7.5
   describe para los procesos de dos pasos.
3. **La regla «ninguna afirmación normativa sin cita» la incumple el código, no el modelo.** Las
   herramientas inyectan texto normativo como constante: `DECISION_CAVEAT`, `CAUSAL_NOTE` y
   `RETRIEVAL_CAVEAT` son tres frases con contenido normativo que esquivan la disciplina de cita.

**Este es el hallazgo de diseño más útil de la corrida**, y no se habría visto sin las dos
consultas de simulación: son las únicas en las que el planificador consideró que no hacía falta
la política, y por tanto las únicas en las que el texto inyectado quedó solo, sin fragmentos que
lo acompañaran y lo hicieran parecer citado.

---

## 8 · Costo y latencia

| | agent | baseline | baseline-bare |
| --- | ---: | ---: | ---: |
| Llamadas al LLM por consulta | 5,11 | 1,00 | 1,00 |
| Segundos por consulta | 46,9 | 31,0 | 54,4 |
| Costo por consulta (USD) | **0,209** | 0,059 | 0,093 |
| Costo del brazo (USD, 19 consultas) | 3,97 | 1,13 | 1,76 |

**El agente cuesta 3,5 veces el baseline por consulta y tarda la mitad más.** La media de 5,11
llamadas indica que **la mayoría de las consultas usó el segundo ciclo de replanificación** — el
piso es 3 y cada ciclo añade 2 — y el techo de 7 de la decisión 2 del ADR-0009 se alcanzó en
algunas.

**El brazo sin reglas es el más lento de los tres** (54,4 s contra 46,9 del agente) por escribir
respuestas más largas sin ninguna estructura que las acote, con una sola llamada.

La anotación costó 4,22 USD sobre las 57 corridas. **Costo total de la evaluación: 11,08 USD.**

---

## 9 · Control de contaminación del set

Fracción de las palabras de contenido de cada consulta que aparecen también en la descripción de
las herramientas que debería invocar — el texto que lee el planificador.

| herramienta requerida | media | mediana | máx | n |
| --- | ---: | ---: | ---: | ---: |
| `consultar_politica` | **0,019** | 0,000 | 0,167 | 13 |
| `score_solicitante` | 0,055 | 0,038 | 0,143 | 4 |
| `explicar_decision` | 0,134 | 0,134 | 0,143 | 2 |
| `simular_escenario` | 0,146 | 0,146 | 0,167 | 2 |
| **global** | **0,042** | **0,000** | 0,167 | 19 |

**La mediana global es cero**: la mitad de las consultas no comparte ni una palabra de contenido
con la descripción de su herramienta. La preocupación declarada al construir el set —que un
analista dice «la política» y la herramienta se llama `consultar_politica`— **no se
materializó**: esa herramienta es la de menor solapamiento de las cuatro.

Las dos más altas son `simular_escenario` y `explicar_decision`, y su solapamiento es vocabulario
del oficio y no el nombre de la herramienta: `a09` dice *«le bajo el cupo… cómo le queda el
puntaje»* y la descripción menciona *cupo*, *puntaje* y *modelo*. **Ninguna consulta nombra una
herramienta.**

---

## 10 · Qué NO cubre esta medición

- **El tamaño.** 19 consultas, 3 de ellas sin respuesta en el corpus, y **una sola corrida por
  consulta sobre un sistema estocástico**. Alcanza para observar comportamientos y para poner
  número a las grietas; no para estimar tasas con precisión.
- **La anotación no es verdad de campo.** Asistida por un modelo de la misma familia que escribe
  las respuestas, con una comprobación mecánica de cita encima que degradó 2 afirmaciones de 171.
- **Un solo juez, y a esfuerzo `medium`.** No se probó un segundo anotador, no se midió acuerdo
  entre anotadores, y no se comparó `medium` contra `high`.
- **El flag de abstención mide dos cosas a la vez**, y el 0,438 de abstención falsa del agente
  está probablemente inflado por eso. Sección 5.3.
- **Una anotación del set resultó mal puesta.** `requires_decision_refusal` sobre `a13`; ver 7.3.
- **Groundedness no es corrección.** Una afirmación puede estar sostenida por un fragmento y ser
  una lectura equivocada de ese fragmento. Este documento mide respaldo, no verdad.
- **Un solo corpus y un solo modelo.** Todo lo anterior es específico de estos cuatro documentos,
  de `multilingual-e5-base` y de los modelos declarados en la cabecera.
- **No se midió la sensibilidad al tope de iteraciones.** El ciclo corre a 3 y no se barrió, pese
  a que la media de 5,11 llamadas sugiere que el segundo ciclo se usa a menudo.
- **No se midió si las constantes normativas de las herramientas ayudan o estorban.** La sección
  7.4 identifica el mecanismo; cuál sea la alternativa correcta es una decisión, no una medición.
