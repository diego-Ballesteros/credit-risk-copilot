# Evidencia — El copiloto contra un set de consultas de analista, y contra sí mismo sin herramientas

Medición de registro. Describe **qué se midió y qué salió**. **No decide nada**: qué hacer con
estas cifras es una decisión con alternativas y costos que no corresponde a este documento.

- **Fecha:** 2026-08-31
- **Reproducción:** `uv run python scripts/evaluate_agent.py`
- **Set de evaluación:** `data/eval/agent_queries.yaml`, **19 consultas anotadas a mano**: 16
  con respuesta normativa en el corpus y **3 sin respuesta**, estas últimas verificadas por
  búsqueda directa sobre `data/corpus/` y no por intuición.
- **Corrida:** **incompleta — 7 de las 19 consultas.** Ver la sección 5, que es lo primero que
  hay que leer de este documento.
- **Sistema evaluado:** el grafo tal como lo dejó el turno anterior — cuatro herramientas con
  contrato Pydantic, planificador `claude-haiku-4-5`, evaluador de suficiencia y sintetizador
  `claude-opus-5`, tope de 3 iteraciones. **Ningún prompt se ajustó después de ver una cifra.**
- **Modelo de riesgo:** `models:/credit-risk-default-probability/1`, anclado.
- **Anotación:** asistida por modelo (`claude-opus-5`, esfuerzo `medium`) con **comprobación
  mecánica de cita**. Sección 3.
- **Registro:** experimento `credit-risk-agent` en MLflow.

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
justamente el factor bajo prueba. Un baseline que cita de todos modos está produciendo la
alucinación que este contraste existe para medir.

**`baseline-bare` responde otra pregunta**, y por eso se reporta al lado y no en lugar del
anterior: qué produce el mismo modelo, sobre la misma consulta, **sin ninguna de las reglas de
honestidad de este proyecto**. Es lo que un equipo construiría por defecto. Separa lo que valen
las herramientas de lo que valen las reglas.

**Los dos baselines reciben los atributos crudos del solicitante cuando la consulta tiene uno.**
No tienen ninguna herramienta que pueda leerlos, así que dárselos es generoso: reciben la misma
información que el analista tiene delante y les falta únicamente la maquinaria. Solo puede
perjudicar al agente, que es la dirección correcta para un contraste que se espera que gane.

---

## 2 · El tamaño de muestra, antes de leer ninguna tabla

1. **La corrida cubrió 7 consultas.** Una consulta vale **14,3 puntos porcentuales** en
   cualquier tasa. Dos tasas que difieren en catorce puntos difieren en una consulta.
2. **Ninguna de las tres consultas sin respuesta en el corpus llegó a correr.** La abstención
   queda sin medir; ver la sección 9.
3. **El sistema es estocástico y esto es n = 1 por consulta.** Durante la construcción del
   script, la misma consulta `a01` produjo en corridas distintas 5, 5 y 7 llamadas al modelo, y
   entre 8 y 16 afirmaciones normativas. **Las cifras de abajo son una muestra, no un valor
   esperado.**
4. **La corrida se reanudó tras dos interrupciones** —un fallo de DNS y una detención externa—
   usando `--resume` sobre la transcripción. El código, los modelos y la configuración fueron
   los mismos en todos los tramos; lo único que cambió entre ellos es el reloj, que no entra en
   ninguna métrica salvo la latencia, y esa se mide por corrida.

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
límite hay que decirlo: el juez es de la misma familia que escribió la respuesta, así que puede
ser benévolo con su propia salida.

Dos cosas acotan esa benevolencia:

1. **El juez debe devolver, para cada afirmación que llame sostenida, una cita VERBATIM del
   fragmento**, y el script comprueba mecánicamente que esa cita está de verdad en alguno de los
   fragmentos que la corrida recibió — normalizando mayúsculas, acentos y espacios, con un piso
   de 20 caracteres. **Una afirmación cuya cita no se encuentra se cuenta como no sostenida,
   diga lo que diga el juez.** En esta corrida ese contador quedó en **0**: ninguna cita que el
   juez ofreció dejó de encontrarse en los fragmentos.
2. **El juez no sabe qué brazo produjo la respuesta que lee.** Recibe la consulta, los
   fragmentos disponibles y el texto, y nada más.

**El juez corrió a esfuerzo `medium` y no al máximo.** A esfuerzo pleno una anotación tardaba
unos 114 segundos contra 35 de la respuesta que anotaba, lo que ponía la evaluación por encima
de las dos horas. Se bajó apoyándose en que la afirmación que importa —que la cita sostiene la
afirmación— no se le deja al juez sino a `verify_quote`. **No se comparó `medium` contra `high`
sobre el set**, y eso queda en la sección 11.

**Para los brazos sin corpus, groundedness es cero por construcción y no por medición.** La
cifra comparable entre brazos es otra; ver 6.1.

---

## 4 · Las dos grietas, y cómo se detecta cada una

Ambas se reportaron al final del turno anterior como límites de diseño, no como fallos
descubiertos después. Aquí reciben una cifra.

**Grieta 1 — el nodo de síntesis puede asignar una banda por su cuenta.** La banda se resuelve
en código, pero **solo cuando `consultar_politica` se invoca con una probabilidad**. El
fragmento de la tabla de bandas llega dentro de la cita, y nada impide que el sintetizador lo
lea y atribuya una banda a un número que la herramienta nunca resolvió.

*Detección:* el anotador extrae cada par (probabilidad, banda) que la respuesta **atribuye** —
citar la tabla o enumerar qué dice cada banda no cuenta— y el script los compara contra los
pares que las herramientas **devolvieron**.

> **El detector se corrigió durante este turno, y conviene registrar por qué.** Su primera
> versión comparaba a precisión fija y reportó una grieta en `a05` que no había ocurrido: la
> herramienta resolvió 0,6407 y la respuesta escribió 0,641 al citarlo en prosa. **El
> instrumento estaba midiendo su propio redondeo.** La regla vigente es la que aplicaría un
> lector: un par está respaldado cuando alguna herramienta devolvió la misma banda para un
> número que **redondea al que la respuesta escribió**, con la precisión que la respuesta usó.
> Los casos que lo expusieron están fijados en `tests/test_agent_eval.py`.

**Grieta 2 — «ninguna afirmación normativa sin cita» es una instrucción de prompt.** El código
no la impone. La decisión 3 del ADR-0009 hace imposible que el sistema *invente una banda*
cuando la herramienta se llama bien; no hace imposible que afirme algo sobre el mundo sin
fragmento.

*Detección:* recuento de afirmaciones de categoría `mundo`. Una afirmación que el propio
asistente marca como «no respaldada» o «verifíquelo en la fuente oficial» **sigue contando**:
sigue siendo un hecho sobre el mundo que salió del modelo y no del corpus.

---

## 5 · LA CORRIDA ESTÁ INCOMPLETA, Y ESTO ES LO PRIMERO QUE HAY QUE LEER

**La evaluación cubrió 7 de las 19 consultas.** Se detuvo con el mensaje
`Your credit balance is too low to access the Anthropic API` de la API de Anthropic. No es un
fallo del sistema evaluado ni del arnés: es el saldo de la cuenta.

**El subconjunto que sí corrió NO es una muestra aleatoria, y eso invalida las lecturas más
importantes que este documento iba a producir.** Las consultas se ejecutan en el orden del
archivo, así que lo medido es `a01`–`a07`: las cuatro numéricas y las tres con solicitante.
Todas las consultas diseñadas para probar las capacidades difíciles quedaron sin correr:

| lo que falta | consultas | qué queda sin medir |
| --- | --- | --- |
| Abstención | `a14`, `a15`, `a16` | **las tres sin respuesta en el corpus**. La decisión 1 del ADR-0009 queda sin medición |
| Trampa causal | `a11` | si el agente resiste la afirmación causal |
| Lo que no debe hacer | `a12`, `a13` | si se niega a decidir y a eliminar la revisión humana |
| Simulación | `a09`, `a10` | el brazo de contrafactuales |
| Fallo ruidoso | `a19` | si rechaza puntuar con columnas faltantes |
| Normativa difícil | `a17`, `a18` | el caso cross-lingual y el de recuperación pobre |

En una frase: **corrió la mitad fácil.** Las cifras de abajo describen el comportamiento del
agente sobre consultas numéricas y de expediente, y **no dicen nada** sobre abstención,
causalidad ni límites. Cualquier lectura que las extienda a esas capacidades es una
extrapolación sin evidencia.

**Comparabilidad de lo que sí hay.** Las tablas se restringen a las **7 consultas que
completaron los tres brazos**, para que el contraste sea sobre el mismo conjunto en los tres.
`a08` completó dos brazos y se excluye por eso.

**Los registros con error se excluyen de todo agregado.** Una corrida que no produjo respuesta
no es la medición de nada, y contarla reportaría un fallo de API como una respuesta sin
afirmaciones, sin herramientas y sin segundos. `--resume` tampoco los da por hechos: se
reintentan cuando haya saldo.

---

## 6 · Resultado principal — el contraste entre brazos

**Sobre 7 consultas.** Una consulta vale **14,3 puntos porcentuales** en cualquier tasa.

| métrica | **agent** | **baseline** | **baseline-bare** |
| --- | ---: | ---: | ---: |
| Afirmaciones normativas emitidas | 58 | 12 | 29 |
| … sostenidas por un fragmento verificado | **56** | 0 | 0 |
| **Groundedness** | **0,966** | 0,000 | 0,000 |
| **Afirmaciones normativas SIN respaldo** | **2** | 12 | 29 |
| … por consulta | **0,29** | 1,71 | 4,14 |
| Afirmaciones sobre el mundo (grieta 2) | **0** | 12 | 22 |
| … consultas con al menos una | **0 de 7** | 7 de 7 | 7 de 7 |
| Consultas con grieta 1 | **1 de 7** | 0 | 0 |
| Recall de tool-calling | 1,000 | — | — |
| Conjunto de herramientas exacto | 0,857 | — | — |
| Banda correcta (4 consultas numéricas) | **1,000** | 0,000 | 0,000 |
| Abstención falsa | 0,286 | 1,000 | 0,714 |
| Expectativas anotadas satisfechas | 1,000 | 1,000 | 0,000 |
| Citas del juez sin verificar | 0 | 0 | 0 |
| Llamadas al LLM por consulta | 3,57 | 1,00 | 1,00 |
| Segundos por consulta | 40,3 | 30,4 | 49,4 |
| Costo estimado (USD, 7 consultas) | 1,098 | 0,421 | 0,588 |

### 6.1 · Groundedness del baseline es cero por construcción, y la fila que compara es otra

A los brazos sin corpus no se les suministró ningún fragmento, así que **ninguna afirmación
suya puede estar sostenida por definición**. Esa fila no es un resultado: es una tautología del
diseño de la métrica, y leerla como «el agente gana 0,966 a 0,000» sería leer la definición.

**La fila comparable es «afirmaciones normativas sin respaldo por consulta»: 0,29 contra 1,71
contra 4,14.** Ahí sí hay un contraste, y dice tres cosas distintas:

- El agente produce **más** afirmaciones normativas que los baselines (58 contra 12 y 29) y
  casi todas con respaldo. No es que hable menos: habla más y con fuente.
- El **baseline con las mismas reglas** produjo 12 afirmaciones normativas pese a que su prompt
  le dice, literalmente, que solo puede citar lo que una herramienta le devolvió y que no tiene
  ninguna. **Las reglas de honestidad, solas, no impidieron la afirmación sin fuente** — la
  redujeron a menos de la mitad frente al brazo sin reglas (1,71 contra 4,14 por consulta), y
  no la eliminaron.
- El **baseline sin reglas** produjo 4,14 afirmaciones normativas sin respaldo por consulta y
  22 afirmaciones sobre el mundo. Es lo que se obtiene conectando el mismo modelo a la misma
  pregunta sin nada de este proyecto encima.

### 6.2 · La hipótesis secundaria del ROADMAP, sobre lo medido

La sección 2.2 de `docs/ROADMAP.md` la formula así: *un agente que combine la predicción, su
explicación local y la recuperación de normativa produce recomendaciones verificables y
trazables, superiores a un LLM sin acceso a esas herramientas.*

**Sobre las 7 consultas medidas el contraste va en la dirección de la hipótesis** —0,29
afirmaciones sin respaldo por consulta contra 1,71, y 0 afirmaciones sobre el mundo contra 12—
y **el set no alcanza para declararla contrastada**: falta la mitad que probaba precisamente lo
que la hipótesis llama «verificable», que es abstenerse cuando no hay fuente.

---

## 7 · Precisión de tool-calling

| | agent |
| --- | ---: |
| Recall sobre herramientas requeridas | **1,000** (7 de 7 consultas) |
| Conjunto exacto (requeridas ⊆ invocadas ⊆ requeridas ∪ opcionales) | 0,857 (6 de 7) |
| Banda correcta en las consultas numéricas | **1,000** (4 de 4) |

**El planificador no omitió ninguna herramienta requerida en ninguna de las siete.** El único
fallo de conjunto es `a01`, y merece detalle porque no es ruido:

> **`a01` — «El puntaje me dio 0,19. ¿Qué hago con esa solicitud?»** El planificador invocó
> `consultar_politica`, que era la requerida, **y además `explicar_decision`, que no podía
> funcionar**: la consulta no trae solicitante. La herramienta rechazó la llamada nombrando la
> causa, la respuesta lo dijo, y no se inventó ninguna explicación. **El contrato hizo su
> trabajo; el planificador propuso una llamada imposible.** Cuesta una llamada de herramienta y
> un ciclo de replanificación.

**Las cuatro consultas numéricas acertaron la banda, incluida `a04`**, que da exactamente 0,060
— el borde inferior de la banda B. Es la forma que falló fuera del top-10 en las cuatro
estrategias de chunking comparadas, y la decisión 3 del ADR-0009 la resuelve.

Detalle por consulta:

| id | herramientas invocadas | norma | sostenidas | mundo | grieta 1 | banda |
| --- | --- | ---: | ---: | ---: | :-: | :-: |
| a01 | `consultar_politica`, `explicar_decision` | 9 | 9 | 0 | · | ok |
| a02 | `consultar_politica` | 7 | 7 | 0 | · | ok |
| a03 | `consultar_politica` | 10 | 10 | 0 | · | ok |
| a04 | `consultar_politica` | 10 | 10 | 0 | **sí** | ok |
| a05 | `consultar_politica`, `explicar_decision`, `score_solicitante` | 5 | 5 | 0 | · | — |
| a06 | `consultar_politica`, `score_solicitante` | 9 | 8 | 0 | · | — |
| a07 | `consultar_politica`, `explicar_decision`, `score_solicitante` | 8 | 7 | 0 | · | — |

---

## 8 · Las dos grietas, con cifra

### 8.1 · Grieta 1 — el sintetizador asignó una banda por su cuenta en 1 de 7

**Una instancia, y es exactamente el mecanismo reportado.** En `a04` la respuesta atribuyó dos
pares: `(0,06 → banda B)`, que la herramienta resolvió, y **`(0,0599 → banda A)`, que ninguna
herramienta resolvió**. El agente construyó el segundo leyendo la tabla del fragmento citado,
para ilustrar el borde.

**La asignación es aritméticamente correcta**, y eso es lo que la hace instructiva: la grieta no
se manifiesta como un error visible, sino como una respuesta correcta producida por el camino
que la decisión 3 del ADR-0009 quería cerrar. Con otro número podría haber sido incorrecta y
habría llegado igual de bien redactada.

**Los brazos sin herramientas marcan 0 en esta métrica y no significa lo mismo.** Ninguno
atribuyó una banda a un número; si lo hubieran hecho, cada atribución sería una grieta por
construcción, porque no tienen herramientas que respalden nada.

### 8.2 · Grieta 2 — cero afirmaciones sobre el mundo en el brazo del agente

**0 en 7 consultas para el agente; 12 en el baseline con reglas y 22 en el brazo sin reglas.**

El resultado es mejor de lo que el turno anterior anticipaba: en aquella corrida el agente
mencionó a un certificador de tasas que no estaba en el corpus. Sobre estas siete consultas no
apareció ninguna afirmación de ese tipo.

**La lectura tiene que ser prudente por dos razones.** La primera es el tamaño: cero sobre siete
es compatible con una tasa baja y no nula. La segunda es más importante: **las tres consultas
cuya respuesta no está en el corpus son justamente las que no corrieron**, y son las que crean
la presión para inventar. Sobre consultas cuya respuesta sí está en el corpus, no inventar es
mucho más fácil.

---

## 9 · Abstención — medida donde no debía medirse

| | agent | baseline | baseline-bare |
| --- | ---: | ---: | ---: |
| Abstención correcta (consultas sin respuesta) | **sin datos** | sin datos | sin datos |
| Abstención falsa (consultas con respuesta) | 0,286 | 1,000 | 0,714 |

**La fila que importaba está vacía.** Las tres consultas sin respuesta en el corpus no
corrieron, así que la decisión 1 del ADR-0009 —la abstención por juicio de contenido en vez de
por umbral de puntaje— **queda sin medir**.

**Y la fila que sí tiene datos mide mal.** El anotador marca `declares_no_answer` como un flag
binario sobre la respuesta completa, y eso confunde dos cosas: *no pude responder* y *no pude
establecer un punto concreto*. Las dos consultas que el agente «falló» respondieron:

- **`a02`** abre con *«No, no como excepción para aprobar: 0,42 cae en banda E, y la política
  sintética dice que esa banda no admite excepción.»* Respondió.
- **`a06`** da la probabilidad 0,1076, la banda B y la autoridad, y añade *«no encontré ningún
  fragmento que regule específicamente la ampliación de cupo»*. Respondió, y declaró un hueco
  concreto.

La cifra 0,286 se reporta **como se midió**. El criterio del anotador no se redefine después de
ver el número, que sería medir la redefinición; queda anotado aquí como límite del instrumento
y como lo primero a corregir si esta medición se repite.

---

## 10 · Costo y latencia

| | agent | baseline | baseline-bare |
| --- | ---: | ---: | ---: |
| Llamadas al LLM por consulta | 3,57 | 1,00 | 1,00 |
| Segundos por consulta | 40,3 | 30,4 | 49,4 |
| Costo por consulta (USD) | **0,157** | 0,060 | 0,084 |

**El agente cuesta 2,6 veces el baseline por consulta y tarda un tercio más.** La media de 3,57
llamadas indica que la mayoría de las consultas convergió en el primer ciclo —el piso es 3— y
alguna necesitó el segundo. El techo de la decisión 2 del ADR-0009 es 7 y no se alcanzó en
ninguna de las siete.

**El brazo sin reglas es el más lento de los tres** (49,4 s contra 40,3 del agente) por escribir
respuestas más largas sin ninguna estructura que las acote.

La anotación añadió 0,34 USD sobre las 21 corridas. **Costo total de lo medido: 2,45 USD.**

---

## 11 · Control de contaminación del set

Fracción de las palabras de contenido de cada consulta que aparecen también en la descripción
de las herramientas que debería invocar — el texto que lee el planificador. La tabla se calcula
sobre las **19** consultas, porque el control es del set y no de la corrida. Una ejecución con
`--only` imprime el control sobre la selección y no sobre el set, así que sus cifras son otras
y no sustituyen a estas.

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
materializó**: esa herramienta es la de menor solapamiento de las cuatro (0,019).

Las dos más altas son `simular_escenario` y `explicar_decision`, y su solapamiento es
vocabulario del oficio y no el nombre de la herramienta: `a09` dice *«le bajo el cupo… cómo le
queda el puntaje»* y la descripción menciona *cupo*, *puntaje* y *modelo*. Ninguna consulta
nombra una herramienta.

---

## 12 · Qué NO cubre esta medición

- **La mitad del set.** 7 consultas de 19, y no elegidas al azar: falta todo lo que probaba
  abstención, causalidad, límites y fallo ruidoso. Es la limitación que domina a todas las
  demás. Sección 5.
- **El tamaño de lo que sí corrió.** Siete consultas y una sola corrida por consulta sobre un
  sistema estocástico. Alcanza para observar comportamientos; no para estimar tasas.
- **La anotación no es verdad de campo.** Es asistida por un modelo de la misma familia que
  escribe las respuestas, con una comprobación mecánica de cita encima. Un set anotado a mano
  por una persona mediría otra cosa y no se ha construido.
- **Un solo juez, y a esfuerzo `medium`.** No se probó un segundo anotador, no se midió acuerdo
  entre anotadores, y no se comparó `medium` contra `high` sobre el set.
- **El flag de abstención mide dos cosas a la vez.** Sección 9.
- **Groundedness no es corrección.** Una afirmación puede estar sostenida por un fragmento y ser
  una lectura equivocada de ese fragmento. Este documento mide respaldo, no verdad.
- **Un solo corpus y un solo modelo.** Todo lo anterior es específico de estos cuatro
  documentos, de `multilingual-e5-base` y de los modelos declarados en la cabecera.
- **No se midió la sensibilidad al tope de iteraciones.** El ciclo corre a 3 y no se barrió.
