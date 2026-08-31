# ADR 0009 — Diseño del agente: abstención, ciclo y resolución de banda

- **Status:** Accepted
- **Date:** 2026-08-28

---

## Contexto

El copiloto orquesta cuatro herramientas sobre un grafo con estado tipado. Tres de sus
decisiones no se derivan de la biblioteca ni del framework: se derivan de mediciones que este
proyecto ya tenía cuando el agente se diseñó, y por eso se registran juntas aquí.

Las tres responden a la misma pregunta de fondo, formulada tres veces sobre superficies
distintas: **qué parte del sistema tiene derecho a afirmar algo.** La primera decide quién
puede afirmar que no hay respuesta; la segunda, cuántas veces puede el sistema volver a
intentarlo antes de tener que decirlo; la tercera, quién decide en qué banda cae un número.

El ADR-0008 dejó las tres planteadas y ninguna resuelta. Su decisión 4 descartó el umbral de
similitud y no propuso reemplazo; su sección de consecuencias registró que las preguntas
numéricas no se arreglan con recuperación y dejó escrito que *"la respuesta corresponde a un
filtro por rango resuelto en código… queda para el turno de tools"*. Este ADR cierra las tres.

---

## Decisión

### 1 · El agente declara que no encuentra respuesta por juicio de un nodo evaluador sobre el contenido recuperado, y no por umbral de puntaje de similitud

El ADR-0008, decisión 4, descartó el umbral **sobre evidencia medida**: sobre las tres
preguntas sin respuesta en el corpus, el mejor resultado puntúa hasta 0,8501, **por encima de
24 de las 26 preguntas que sí tienen respuesta**. Las dos nubes se solapan casi por completo
en las cuatro estrategias comparadas.

**La razón de fondo, y es la que gobierna esta decisión:** el puntaje **no lee el texto**. Mide
una distancia entre vectores. Un número que no ha leído el fragmento no puede decir si el
fragmento responde la pregunta, y ninguna elección de corte sobre ese número puede adquirir
esa capacidad.

Un nodo que **lee los fragmentos y responde si sostienen la pregunta** ejerce implicación
textual, que es una operación distinta. Lo importante no es que sea "mejor": es que **su modo
de falla es independiente del que se midió**. El solapamiento de distribuciones que hace
inviable el umbral no dice nada sobre si un lector distingue un fragmento que responde de uno
que no, porque son dos preguntas hechas a dos facultades distintas.

**El nodo evaluador usa el modelo grande y no el del planificador.** Su falla cara es un
veredicto de suficiencia **falso positivo**: declarar que la evidencia basta cuando no basta
produce una respuesta confiada construida sobre fragmentos que no la sostienen — que es
exactamente el fallo que todo este diseño existe para impedir. La otra dirección solo cuesta
un ciclo más.

#### Alternativa descartada: umbral de similitud

Descartada por el **ADR-0008, decisión 4**, con su evidencia y con su límite declarado: tres
preguntas sin respuesta bastan para demostrar que el solapamiento **existe**, que es una
afirmación existencial, y **no bastan para estimar dónde caería un umbral**. Si alguna vez se
reúne un conjunto mucho mayor de preguntas sin respuesta, esa decisión —y con ella esta— debe
revisarse.

### 2 · El ciclo de re-planificación se acota en tres iteraciones

Cada iteración cuesta **dos llamadas al modelo** —planificar y juzgar suficiencia—, de modo que
el techo por consulta es **siete**: tres pares más la síntesis.

**Por qué tres.** Es lo que las herramientas disponibles pueden aprovechar:

1. la primera vuelta **reúne evidencia**;
2. la segunda **reacciona a lo que la primera encontró** — una herramienta que falló y puede
   reintentarse con otros argumentos, o una pregunta que conviene reformular con el vocabulario
   del documento;
3. la tercera es **la última antes de que la respuesta tenga que ser honesta** sobre lo que
   falta.

Más allá de ahí el planificador no converge: interroga al mismo índice con la misma pregunta, y
**el corpus no cambia entre intentos**.

**El tope se aplica en dos lugares independientes**: en el predicado de enrutamiento que sigue
al evaluador, y como límite de recursión del grafo derivado de la misma constante. Un error en
la condición no basta para dejarlo girando. Un tope que vive en un solo `if` es un tope que una
comparación mal escrita elimina.

#### Alternativas descartadas

- **Una sola iteración.** Impide el rescate por reformulación, que está medido: la tabla de
  bandas pasa de fuera del top-8 al puesto 3 cuando la misma pregunta se reformula con el
  vocabulario de la tabla (`docs/analysis/retrieval-evidence.md`, sección 6). Sin segunda
  vuelta, el agente no puede usar lo que la primera le enseñó.
- **Sin límite.** Un agente que cicla sin converger es un agente que cuesta dinero sin producir
  nada.

### 3 · La banda de decisión se resuelve comparando números en código, y el fragmento que la respalda se recupera por identificador y no por similitud

**Evidencia:** las consultas numéricas —dar un valor concreto de probabilidad y preguntar qué
decisión corresponde— **fallan fuera del top-10 en las cuatro estrategias de chunking
comparadas**. Las cuatro disponían del mismo texto. Un recuperador denso empareja superficies y
**no evalúa si un valor cae dentro de un intervalo**; ninguna forma de cortar el documento puede
adquirir esa capacidad.

Que la cita se traiga **por identificador y no por búsqueda** evita reintroducir, en la capa de
la cita, la misma falla que la tabla se movió a código para evitar. Si el fragmento que respalda
la banda dependiera de que una búsqueda acertara, el caso en que la búsqueda falla —que es
precisamente el caso medido— dejaría la banda sin con qué citarse.

**Consecuencia de diseño registrada.** Los nodos de herramientas del grafo se ejecutan en
paralelo dentro de un mismo superstep y **no pueden leerse entre sí**: cuando el planificador
pide un score y una consulta de política en la misma vuelta, la herramienta de política no
puede ver la probabilidad que la de score está calculando a su lado. Por eso **la herramienta de
política vuelve a puntuar al solicitante por su cuenta** en vez de esperar el resultado de la
otra. Volver a puntuar una fila contra el mismo artefacto anclado cuesta microsegundos;
**confiar en que el modelo de lenguaje retransmita el número sería exactamente la falla que los
contratos de las herramientas existen para impedir.**

---

## Consecuencias

Las cifras vienen de la entrada 012 de `docs/EVALUATION.md`: **19 consultas, tres brazos, 57
corridas sin error**, con el sistema tal como estas tres decisiones lo dejaron y **sin ajustar
ningún prompt después de ver una cifra**.

**La decisión 1 tiene ahora respaldo medido, y la cifra es la separación y no el acierto.** El
agente **abstuvo en las tres consultas cuya respuesta no está en el corpus** —abstención correcta
1,000— y respondió en nueve de las dieciséis que sí la tienen. Lo que sostiene la decisión no es
ese 1,000 aislado: el brazo sin herramientas también marca 1,000, **porque abstiene en las
diecinueve**, con una abstención falsa de 1,000. Un sistema que dice «no sé» a todo acierta todas
las preguntas sin respuesta y no sirve para nada. La cifra que importa es la **separación entre
abstener cuando debe y abstener cuando no debe**: **+0,562 en el agente, 0,000 en el baseline**.
El juicio textual distingue; la ausencia de corpus no.

Con dos límites. El primero es el tamaño: tres consultas sin respuesta, y una vale 33 puntos —
el mismo límite que el ADR-0008 declaró al descartar el umbral. El segundo es del instrumento: el
flag de abstención del anotador confunde *no pude responder* con *no pude establecer un punto
concreto*, de modo que la abstención falsa de 0,438 está probablemente inflada.

**La garantía de la decisión 3 no se ejerció, y sigue siendo parcial.** El código resuelve la
banda **solo cuando la herramienta se invoca con una probabilidad**, y el nodo de síntesis puede
leer la tabla del fragmento citado y atribuir una banda por su cuenta. **En 19 consultas no
ocurrió ni una vez.** El contador automático marcó una instancia en `a06`; verificarla puntuando
de nuevo la fila con el artefacto anclado demostró que los dos números que la respuesta escribió
—0,108 y 0,10757— eran **citas fieles del 0,10757135201580555 que la herramienta devolvió**, y
que la marca la produjo el redondeo del propio instrumento. Que la grieta no se ejerza en 19
consultas no la cierra: sigue abierta por construcción.

**Las cuatro consultas numéricas acertaron la banda, incluida la del borde exacto 0,060.** Es la
forma que falló fuera del top-10 en las cuatro estrategias de chunking comparadas, y es el
resultado que esta decisión existía para producir.

**La grieta 2 aparece, y aparece concentrada donde el corpus calla.** De las **8 afirmaciones
sobre el mundo** que el agente produjo en 19 consultas, **6 están en las tres cuya respuesta no
está en el corpus** y las 2 restantes en las dos de simulación. **Ninguna en las once que el
corpus sí responde.** La concentración es el mecanismo hecho visible: cuando no hay fragmento que
citar, el modelo llena el hueco con lo que sabe, y la instrucción del prompt no lo impide. Contra
40 del mismo modelo sin herramientas y 73 sin herramientas ni reglas.

**Una medición parcial anterior había reportado 0 afirmaciones de este tipo, sobre 7 consultas
que excluían justamente esas tres.** Queda registrado porque es la advertencia que aquella
versión se hizo a sí misma y que resultó exacta.

**El código incumple la regla que el prompt impone, y eso se descubrió aquí.** En `a09` y `a10`
el planificador no invocó `consultar_politica`, la corrida no recibió **ningún** fragmento, y aun
así la respuesta contenía cinco afirmaciones normativas. No son invenciones: salen de
`DECISION_CAVEAT`, una constante de `agent/tools.py` que `score_solicitante` devuelve con cada
puntuación y que transcribe a mano la sección 2.2 de la política. **La frase es verdadera y
trazable, y llega sin cita que el analista pueda comprobar.** `DECISION_CAVEAT`, `CAUSAL_NOTE` y
`RETRIEVAL_CAVEAT` son texto normativo inyectado por el código que esquiva la disciplina de cita
que la decisión 1 y el prompt establecen para el modelo. Qué hacer con eso —recuperarlas por
identificador como se hace con la tabla de bandas, o marcarlas como no citables— es una decisión
con alternativas que este ADR no toma.

**«Ninguna afirmación normativa sin cita» sigue siendo una regla de prompt, y ahora se sabe
cuánto vale sola.** El brazo `baseline` conserva esa regla **literalmente** y no tiene
herramientas; aun así emitió 42 afirmaciones normativas sin respaldo y 40 sobre el mundo en 19
consultas. Las reglas redujeron la afirmación sin fuente a menos de un tercio frente al brazo sin
reglas —2,21 contra 7,32 por consulta— y **no la eliminaron**. Es la justificación empírica de
por qué las garantías de este diseño están en los contratos de las herramientas y no en el texto
del prompt.

**El costo del techo de siete llamadas es acotado y ahora conocido.** La media observada fue de
**5,11 llamadas y 0,209 USD por consulta**, contra 0,059 del mismo modelo sin herramientas: el
agente cuesta unas **3,5 veces** el baseline. Una media de 5,11 sobre un piso de 3 significa que
**la mayoría de las consultas usó el segundo ciclo de replanificación**, de modo que la decisión 2
no es un tope decorativo: el ciclo se usa. No se midió la sensibilidad a ese tope.

**Las dos decisiones de enrutamiento son inseparables en la práctica.** El ciclo de la decisión 2
y el evaluador de la decisión 1 son la misma arista vista dos veces, y la entrada 006 de
`docs/ERRORS_AND_LEARNINGS.md` registra qué pasó cuando un camino del grafo —el plan vacío— se
saltaba al evaluador: una respuesta bien redactada, sin herramientas, sin citas y sin ningún
error visible.
