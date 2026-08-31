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

**Las cifras de abajo salen de una medición incompleta, y eso condiciona todas.** La evaluación
de la entrada 012 de `docs/EVALUATION.md` cubrió **7 de las 19 consultas** antes de que se
agotara el saldo de la API. El subconjunto no es aleatorio: es el orden del archivo, es decir
las consultas numéricas y las de expediente. Todo lo que probaba abstención, causalidad y
límites quedó sin correr.

**La garantía de la decisión 3 no es total, su límite es preciso, y ya se observó.** El código
resuelve la banda **solo cuando la herramienta se invoca con una probabilidad**. El nodo de
síntesis recibe el texto de la tabla dentro del fragmento citado, y nada le impide leerla por su
cuenta. **Ocurrió en 1 de las 7 consultas medidas**: ante *«¿y si me da exactamente 0,06?»* la
respuesta atribuyó la banda B a 0,06 —que la herramienta resolvió— y además **la banda A a
0,0599, que ninguna herramienta resolvió**. La atribución es aritméticamente correcta, y eso es
lo instructivo: la grieta no se manifiesta como un error visible sino como una respuesta
correcta producida por el camino que esta decisión quería cerrar. Es una **grieta de diseño
reportada al construir el agente**, no un fallo descubierto después.

**La decisión 1 quedó SIN MEDIR.** Que el juicio textual sea una facultad distinta del puntaje
es un argumento sobre mecanismos, y sigue siéndolo: **las tres consultas sin respuesta en el
corpus son exactamente las que no llegaron a correr**. No hay ninguna cifra en este proyecto
sobre si el nodo evaluador abstiene cuando debe. Lo que sí se observó es un límite del
instrumento: el anotador marca la abstención con un flag binario que confunde *no pude
responder* con *no pude establecer un punto concreto*, de modo que la medición pendiente
necesita además un criterio mejor.

**«Ninguna afirmación normativa sin cita» es una regla de prompt, y el código no la impone.**
La decisión 3 hace imposible que el sistema *invente una banda* cuando la herramienta se llama
bien; no hace imposible que afirme algo sobre el mundo sin fragmento. Sobre lo medido, el brazo
del agente produjo **0 afirmaciones de ese tipo en 7 consultas**, contra 12 del mismo modelo sin
herramientas y 22 sin herramientas ni reglas. **La lectura tiene que ser prudente**: las
consultas que crean la presión para inventar —aquellas cuya respuesta no está en el corpus— son
las que no corrieron.

**El resultado más incómodo del contraste no es sobre el agente sino sobre las reglas.** El
brazo sin herramientas conserva **literalmente** la regla de que solo puede citar lo que una
herramienta le devolvió, y aun así emitió 12 afirmaciones normativas sin respaldo en 7
consultas. Las reglas de honestidad, solas, redujeron la afirmación sin fuente a menos de la
mitad frente al brazo sin reglas y **no la eliminaron**. Es la evidencia más directa de por qué
las garantías de este diseño están en los contratos y no en el prompt.

**El costo del techo de siete llamadas es acotado y ahora conocido.** Una consulta que converge
en la primera vuelta cuesta tres llamadas; una que agota el ciclo, siete. La media observada fue
de **3,57 llamadas y 0,157 USD por consulta**, contra 0,060 del mismo modelo sin herramientas:
el agente cuesta unas **2,6 veces** el baseline. El techo no se alcanzó en ninguna de las siete,
de modo que la decisión 2 se puede revisar contra un presupuesto y no contra una intuición.

**El precio de la decisión 1 se paga en cada consulta.** El nodo evaluador es una llamada al
modelo grande por vuelta, y existe únicamente para poder decir que no se sabe. Un sistema
dispuesto a no abstenerse nunca lo ahorraría entero.

**Las dos decisiones de enrutamiento son inseparables en la práctica.** El ciclo de la decisión
2 y el evaluador de la decisión 1 son la misma arista vista dos veces, y la entrada 006 de
`docs/ERRORS_AND_LEARNINGS.md` registra qué pasó cuando un camino del grafo —el plan vacío— se
saltaba al evaluador: una respuesta bien redactada, sin herramientas, sin citas y sin ningún
error visible.
