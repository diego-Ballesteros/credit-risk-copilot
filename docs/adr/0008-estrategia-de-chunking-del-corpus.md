# ADR 0008 — Estrategia de chunking del corpus normativo

- **Status:** Accepted
- **Date:** 2026-08-27

---

## Contexto

El corpus normativo se indexa para que un agente pueda **citar la fuente de cada afirmación**.
Esa finalidad —citar, no solo recuperar— es la que gobierna las cuatro decisiones de abajo, y
conviene declararla de entrada porque una de ellas no se sostiene en ninguna métrica.

La estrategia inicial se adoptó **por razonamiento, antes de que existiera un set de
evaluación**, y descansaba en tres principios: cortar por unidad estructural del documento,
incrustar la metadata en el texto del chunk, y conservar el encabezado del padre al
subdividir. Los tres se escribieron como si fueran evidentes.

Cuando el set de evaluación existió —29 preguntas anotadas a mano, 26 con respuesta en el
corpus— la medición **contradijo el segundo principio**. Este ADR registra las cuatro
decisiones que resultan de esa medición, incluida la que la evidencia no puede justificar.

---

## Decisión

### 1 · El chunk se corta por unidad estructural del documento, no por longitud fija

**El estado de la evidencia, con precisión.** La comparación pareada sobre las 26 preguntas
con respuesta, a hit@5, entre la unidad estructural y el corte por longitud:

| | aciertan ambas | ninguna | solo unidad estructural | solo corte por longitud |
| --- | ---: | ---: | ---: | ---: |
| Unidad estructural vs. corte por longitud | 15 | 8 | **1** | **2** |

Es un **empate de una pregunta neta**. Y el corte por longitud corrió **con ventaja**: cada
pasaje suyo cubre **1,60 unidades estructurales**, de modo que su top-5 alcanza
aproximadamente **8 unidades frente a 5**. La métrica a nivel de unidad no puede penalizar esa
cobertura extra, así que la favorece.

**La razón de esta decisión no es el rendimiento de recuperación.** Es la **citabilidad**. Un
chunk que coincide con un artículo se puede citar ante un comité: *"según el artículo 13 de la
Ley 1266"*. Una ventana de 700 caracteres no corresponde a ninguna unidad del documento y no
se puede citar como nada.

**Esa propiedad no aparece en ninguna métrica medida, y la decisión se toma sabiéndolo.** Es
exactamente el tipo de decisión que este registro existe para conservar: dentro de seis meses,
alguien que lea solo la tabla de hit@k concluirá que el corte por longitud era mejor y tendrá
razón en lo que la tabla dice.

### 2 · El encabezado de contexto se conserva en la metadata y se retira del texto que se embebe

**Esto revierte el principio 2 de la estrategia original.**

Evidencia, sobre el mismo set y las mismas anotaciones:

| | con encabezado incrustado | sin encabezado |
| --- | ---: | ---: |
| Similitud media entre dos fragmentos del **mismo documento** | **0,9423** | 0,8495 |
| Margen entre el primer y el quinto resultado | 0,0130 | **0,0263** |
| hit@5 a nivel de unidad | 0,423 | **0,615** |

**El efecto es direccional y no uniforme.** Con encabezado, el sistema acierta **mejor el
documento** (doc@1 de 0,692 frente a 0,577) y **peor el artículo dentro de él**. El mecanismo
está medido y no supuesto: el encabezado es un bloque casi idéntico repetido en cada chunk de
un documento, así que empuja todos sus vectores hacia el mismo punto.

Con **cuatro documentos** en el corpus, enrutar al documento correcto es fácil y el costo de no
discriminar entre artículos domina. A otra escala —cuarenta documentos, cuatrocientos— el
balance podría invertirse, y esta decisión debería revisarse.

**La evidencia es direccional y no concluyente por sí sola.** La comparación pareada da **7
discordancias a favor de retirar el encabezado y 2 en contra**, lo que en una binomial de dos
colas sobre 9 pares da **p ≈ 0,18**. Con 26 preguntas, una pregunta vale 3,8 puntos
porcentuales. **Lo que sostiene la decisión no es ese p-valor: es la coincidencia entre ese
resultado y un mecanismo medido de forma independiente** —la homogeneización intra-documento y
la caída del margen— que apunta al mismo sitio por una vía distinta.

#### Alternativa descartada: conservar el encabezado incrustado

El argumento a favor era la comodidad de que **la cita viaje dentro del texto**, de modo que un
agente que cita el fragmento cita su fuente sin consultar ningún campo aparte.

Se descarta porque **la cita puede viajar en la metadata sin costo alguno de recuperación**. El
almacén guarda el texto presentable —encabezado más cuerpo— y codifica solo el cuerpo: un
resultado llega con su documento, su ubicación y su cita completas, y ninguna de las tres se le
cobró al codificador. La comodidad se conserva entera y el costo desaparece.

### 3 · Los avisos de integridad permanecen incrustados en el texto embebido, pese a la decisión 2

Dos avisos, y solo dos: **que un documento es sintético**, y **que un capítulo está derogado**.

**No son contexto: son advertencias que cambian cómo debe leerse el fragmento.** Un fragmento
de la política interna que no declare ser sintético **se citará como normativa real**, y un
capítulo derogado que no lo diga **se citará como norma vigente**. Ninguno de los dos fallos lo
detecta nadie aguas abajo: el fragmento llega bien formado y dice algo falso sobre el mundo.

**El costo de homogeneización que introducen se acepta**, porque el fallo que previenen es
**cualitativamente peor** que un hit@k más bajo. Un hit@k más bajo es un analista que no
encuentra el artículo y lo busca a mano. Una política sintética citada ante un comité como si
fuera la Circular Básica es una decisión de crédito tomada sobre una norma que no existe.

La distinción entre un **aviso** y una **nota de estado** queda tipada en el esquema del corpus
en vez de inferida del texto: `integrity_notice` y `synthetic_notice` se embeben, `status_note`
no. Una nota que dice *"texto vigente, con las modificaciones de la Ley 2157 de 2021"* es
contexto y se queda fuera del vector; buscar la palabra "DEROGADO" con una expresión regular
habría sido adivinar.

### 4 · No se adopta un umbral de score de similitud para declarar que no hay respuesta

Evidencia: sobre las **tres preguntas sin respuesta en el corpus**, el mejor resultado puntúa
hasta **0,8501**, **por encima de 24 de las 26 preguntas que sí tienen respuesta**. Las
distribuciones se solapan casi por completo en las tres estrategias comparadas: en la mejor de
ellas, la pregunta sin respuesta sigue puntuando por encima de 12 de las 26 con respuesta.

**Limitación del tamaño de muestra, y es esencial.** Tres preguntas bastan para demostrar que
**el solapamiento existe**, que es una afirmación existencial y solo necesita un caso. **No
bastan para estimar dónde caería un umbral.** Si se quisiera fijar uno, haría falta un conjunto
mucho mayor de preguntas sin respuesta, y esta decisión debería revisarse con él.

---

## Consecuencias

**El chunk tiene dos textos y confundirlos rompe la decisión 2 en silencio.** `embed_text` es
lo que se convierte en vector; `display_text` es lo que lee una persona. Se construyen dentro
de `chunking.py` y no los ensambla cada consumidor, porque un consumidor que codificara
`display_text` revertiría este ADR y **nada fallaría**: el índice se construiría igual y las
métricas se degradarían sin que ningún error lo dijera. `tests/test_chunking.py` afirma que el
encabezado no llega al texto codificado y que los avisos sí.

**Las cuatro estrategias se siguen midiendo juntas.** `scripts/evaluate_retrieval.py` conserva
los tres brazos que perdieron. Una comparación reejecutada con los brazos perdedores borrados
no la puede verificar nadie.

**Las preguntas numéricas fallan en las cuatro estrategias, fuera del top-10.** Dar un valor
concreto de probabilidad de incumplimiento y preguntar qué decisión corresponde —*"el puntaje
me dio 0,19, ¿qué hago?"*— no recupera la tabla de bandas de la política en ninguna variante.
La pregunta que pide el corte en abstracto, sin dar un número, sí la recupera.

**Eso no es un problema de chunking**, y es importante que quede escrito: las cuatro
estrategias disponían del mismo texto y ninguna lo encontró. Un recuperador denso empareja
superficies y **no evalúa si 0,19 cae dentro de [0,160 ; 0,300)**. Ninguna forma de cortar el
documento puede arreglar eso.

**La respuesta corresponde a un filtro por rango resuelto en código**, dentro de la herramienta
de consulta de política: la herramienta recibe la probabilidad, localiza la banda comparando
números, y recupera el fragmento correspondiente en vez de pedirle al índice que deduzca una
desigualdad. Queda para el turno de tools.

**Efecto colateral favorable sobre el presupuesto del codificador.** Retirar el encabezado bajó
la longitud codificada del peor chunk de 495 a 394 tokens, sobre una ventana de 512. El margen
pasó de **17 a 118 tokens**, de modo que la estrechez señalada al construir el índice deja de
ser una preocupación.

**Nota de remedición, 2026-08-30.** Al implementar la decisión 3 hubo que tipar el aviso de
derogación, que hasta entonces vivía en `status_note`. Ese movimiento cambió una línea del
encabezado de la Circular Básica y, con ella, el brazo A, que pasó de hit@5 0,423 a **0,462**:
una pregunta. Las cifras de las decisiones 2 y 4 de arriba son las que sustentaron la decisión
y **no se reescriben**. La medición posterior a la implementación está en la entrada 011 de
`docs/EVALUATION.md`, e incluye el dato que este ADR no podía tener: **la estrategia adoptada
no reproduce del todo el brazo sin encabezado**. Cuesta dos preguntas de las 26 —hit@5 0,538
frente a 0,615— y esas dos son el precio medido de la decisión 3.
