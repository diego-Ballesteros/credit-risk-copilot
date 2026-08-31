# Evidencia — Recuperación sobre el corpus normativo: tres estrategias de chunking

Medición de registro. Describe **qué se midió y qué salió**. **No decide qué estrategia
adoptar**: elegir entre ellas es una decisión con alternativas y costos que no corresponde a
este documento.

- **Fecha:** 2026-08-30
- **Reproducción:** `uv run python scripts/evaluate_retrieval.py`
- **Set de evaluación:** `data/eval/retrieval_questions.yaml`, 29 preguntas anotadas a mano
  (26 con respuesta en el corpus, 3 sin respuesta).
- **Corpus:** 4 documentos, 89 unidades estructurales bajo la estrategia vigente.
- **Modelo de embeddings:** `intfloat/multilingual-e5-base`, 768 dimensiones, ventana de 512
  tokens, prefijos `query:` / `passage:`.
- **Búsqueda:** **exacta**, producto punto sobre vectores unitarios. **No** se usó el índice
  HNSW de producción: lo que se mide es la estrategia de chunking, y la aproximación del
  índice añadiría ruido ajeno a esa pregunta. La consecuencia hay que decirla: estos números
  describen las estrategias, y el almacén de producción le añade encima su aproximación.
- **Registro:** experimento `credit-risk-retrieval` en MLflow, un run por estrategia.

---

## 1 · Qué se comparó y por qué esas tres

Las tres estrategias se diferencian **en un factor a la vez**, de modo que cada par aísla una
afirmación distinta:

| | Corte | Header incrustado | Qué aísla el par |
| --- | --- | --- | --- |
| **A** | unidad estructural | **sí** | — *(la estrategia vigente)* |
| **B** | unidad estructural | no | **A − B aísla el header** |
| **C** | longitud fija con solape | no | **B − C aísla la estructura** |

**Parámetros de la línea base, elegidos por medición y no por gusto.** La estrategia A produce
89 chunks cuyos cuerpos promedian 591 caracteres. Una ventana de **700 caracteres con 105 de
solape (15%)** produce 97 pasajes sobre el mismo texto, un número comparable. Igualar la
cantidad de vectores es lo que hace que la comparación sea sobre **dónde caen los cortes** y no
sobre cuántos hay.

**La línea base recibe los encabezados.** La ventana se desliza sobre el texto del documento
*incluyendo sus líneas de encabezado*, que es exactamente lo que produciría un chunker ingenuo
apuntado al archivo markdown. Es generoso con la línea base — algunas ventanas se llevan un
encabezado gratis — y es la comparación honesta, porque es lo que un equipo construiría de
verdad y no un muñeco de paja armado para perder.

**Cómo se decide un acierto entre estrategias con formas distintas.** A y B producen un chunk
por unidad estructural; C produce ventanas que atraviesan unidades, así que los identificadores
de chunk no son comparables. La métrica principal se resuelve a nivel de **unidad
estructural**: la anotación nombra chunks, esos chunks resuelven a unidades, cada pasaje
recuperado resuelve al conjunto de unidades cuyo texto solapa —con un piso de 120 caracteres
compartidos, aproximadamente una frase— y hay acierto si la intersección no es vacía.

---

## 2 · Resultado principal — hit@k y MRR a nivel de unidad

| | pasajes | hit@1 | hit@3 | hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A** unidad + header | 89 | 0,346 | 0,423 | 0,423 | 0,413 |
| **B** unidad, sin header | 89 | 0,346 | 0,462 | **0,615** | 0,436 |
| **C** longitud fija + solape | 97 | **0,385** | **0,615** | **0,654** | **0,502** |

Fragmento exacto (solo A y B; la línea base no tiene identificadores comparables):

| | hit@1 | hit@3 | hit@5 |
| --- | ---: | ---: | ---: |
| **A** | 0,346 | 0,423 | 0,423 |
| **B** | 0,346 | 0,462 | 0,615 |

**La estrategia vigente quedó última en hit@3, hit@5 y MRR.**

### 2.1 · El tamaño de muestra, antes de leer la tabla

Con **26 preguntas con respuesta**, una pregunta vale 3,8 puntos porcentuales. Dos tasas que
difieren en cuatro puntos difieren en una pregunta. Por eso se reporta también la comparación
**pareada**, que es la única lectura defendible a este tamaño.

| par | ambas aciertan | ninguna | solo la primera | solo la segunda |
| --- | ---: | ---: | ---: | ---: |
| A vs B | 9 | 8 | **2** (q13, q15) | **7** (q04, q05, q06, q08, q11, q17, q22) |
| A vs C | 10 | 8 | **1** (q15) | **7** (q04, q05, q06, q08, q17, q21, q22) |
| B vs C | 15 | 8 | **1** (q11) | **2** (q13, q21) |

- **A contra B: 7 discordancias a favor de B, 2 a favor de A.** Bajo una prueba binomial de
  dos colas sobre 9 pares discordantes, eso da **p ≈ 0,18**. Es direccional y consistente con
  el mecanismo de la sección 3, y **no alcanza significancia** con este set.
- **B contra C: 1 contra 2.** Es un empate. **La estructura no le ganó al corte por longitud
  en esta medición**, y tampoco perdió.

---

## 3 · Dónde ayuda el header y dónde estorba

La separación que explica el resultado: acertar **el documento** no es lo mismo que acertar **la
unidad dentro de él**.

| | doc@1 | doc@3 | doc@5 | unidad@5 | brecha doc−unidad |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A** | **0,692** | 0,731 | 0,769 | 0,423 | **0,346** |
| **B** | 0,577 | 0,731 | **0,846** | 0,615 | 0,231 |
| **C** | 0,615 | **0,808** | **0,846** | **0,654** | 0,192 |

**A tiene el mejor doc@1 y la peor unidad@5.** El header incrustado enruta mejor al cuerpo
normativo correcto y discrimina peor entre los artículos que hay dentro.

### El mecanismo, medido

Similitud coseno media entre pares de pasajes, dentro del mismo documento y entre documentos
distintos:

| | intra-documento | inter-documento | separación |
| --- | ---: | ---: | ---: |
| **A** | **0,9423** | 0,8422 | 0,1001 |
| **B** | 0,8495 | 0,7820 | 0,0674 |
| **C** | 0,8712 | 0,8081 | 0,0631 |

Con el header, **dos fragmentos cualesquiera del mismo documento se parecen en 0,94**. El
header es un bloque de texto casi idéntico repetido en cada chunk de un documento, así que
empuja todos esos vectores hacia el mismo punto. Sube la separación entre documentos (0,1001
contra 0,0674) y colapsa la distancia dentro de cada uno.

### El mismo efecto visto en el margen

Diferencia entre el score del primer y el quinto resultado, que es la discriminación que
necesitaría cualquier corte:

| | media | mediana | mín | máx |
| --- | ---: | ---: | ---: | ---: |
| **A** | 0,0130 | 0,0105 | 0,0049 | 0,0308 |
| **B** | **0,0263** | 0,0252 | 0,0060 | 0,0562 |
| **C** | 0,0231 | 0,0201 | 0,0082 | 0,0481 |

**El header reduce el margen a la mitad.** La compresión de scores observada en el turno
anterior queda cuantificada aquí.

### Distribución del score del primer resultado

| | mín | p25 | mediana | p75 | máx | rango | desv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A** | 0,8017 | 0,8293 | 0,8348 | 0,8409 | 0,8551 | **0,0534** | 0,0128 |
| **B** | 0,8009 | 0,8472 | 0,8567 | 0,8673 | 0,8838 | 0,0829 | 0,0173 |
| **C** | 0,8026 | 0,8383 | 0,8518 | 0,8569 | 0,8760 | 0,0734 | 0,0152 |

---

## 4 · Un confusor a favor de la línea base, cuantificado

| | unidades por pasaje | caracteres por pasaje | unidades alcanzables en top-5 |
| --- | ---: | ---: | ---: |
| **A** | 1,00 | 1.084,6 | 5,00 |
| **B** | 1,00 | 591,0 | 5,00 |
| **C** | **1,60** | 683,0 | **7,99** |

Una ventana de longitud fija atraviesa unidades, así que **cada pasaje de C cubre 1,60 unidades
en promedio y su top-5 alcanza casi ocho unidades frente a las cinco de A y B**. La métrica a
nivel de unidad no puede ver esa diferencia, y **favorece a C**. Aun con esa ventaja, C le gana
a B por una pregunta neta.

Lo que la métrica tampoco ve: un pasaje de C es una ventana de 700 caracteres que puede
contener la respuesta junto con dos unidades ajenas, y no corresponde a nada que un analista
pueda citar.

---

## 5 · ¿Es viable un umbral de corte para "no sé"?

Score del primer resultado en las 26 preguntas **con** respuesta contra las 3 preguntas **sin**
respuesta en el corpus.

| | con respuesta (mín / mediana / máx) | sin respuesta (mín / mediana / máx) | preguntas con respuesta que quedan por debajo del peor "sin respuesta" |
| --- | --- | --- | ---: |
| **A** | 0,8017 / 0,8348 / 0,8551 | 0,8270 / 0,8500 / 0,8501 | **24 de 26** |
| **B** | 0,8009 / 0,8567 / 0,8838 | 0,8276 / 0,8407 / 0,8582 | 17 de 26 |
| **C** | 0,8026 / 0,8518 / 0,8760 | 0,8212 / 0,8388 / 0,8517 | 12 de 26 |

**Las dos nubes se solapan casi por completo en las tres estrategias.** En A, una pregunta sin
respuesta puntúa más alto que 24 de las 26 preguntas que sí la tienen. Un umbral fijado sobre
el score de similitud **no separa** "hay respuesta" de "no hay respuesta" con este corpus, este
modelo y este set.

Advertencia sobre el tamaño: **n = 3** preguntas sin respuesta. Basta para mostrar el solapamiento
—que es una afirmación existencial— y no para estimar dónde caería un umbral.

---

## 6 · Las preguntas numéricas

Puesto del primer acierto dentro del top-10; "—" significa que no apareció.

| | A | B | C | pregunta |
| --- | ---: | ---: | ---: | --- |
| q18 | — | — | — | El puntaje me dio 0,19. ¿Qué hago con esa solicitud? |
| q19 | — | — | — | Un solicitante quedó en 0,42. ¿Lo puedo mandar a comité para que lo aprueben? |
| q20 | — | — | — | Me salió 0,13 en un solicitante. ¿Se le aprueba lo que pidió o con algún ajuste? |
| q21 | 10 | 8 | **3** | ¿A partir de qué número se rechaza una solicitud? |

**Las tres preguntas que dan un valor concreto de PD fallan en las tres estrategias**, fuera del
top-10. La que pregunta por el corte en abstracto, sin dar un número, sí se recupera.

Las tres estrategias tenían el mismo texto disponible. **El fallo no es del chunking**: es que
un recuperador denso empareja superficies y no evalúa si 0,19 cae dentro de [0,160 ; 0,300).
Cambiar dónde se corta el documento no puede arreglar eso.

### El caso de la tabla de bandas, en detalle

La tabla de bandas de la política interna (`politica-interna-credito::005-2-1-tabla-de-bandas`)
es el fragmento que responde q18, q19 y q20. Medición del turno anterior, con la misma pregunta
reformulada de tres maneras contra la estrategia A:

| formulación | puesto de la tabla de bandas |
| --- | --- |
| "…0,19 de probabilidad… ¿lo apruebo, lo rechazo, comité?" | fuera del top-8 |
| "¿En qué banda cae una probabilidad de 0,19 y qué decisión corresponde?" | 5 |
| "rangos de score y decisión asociada" | 3 |

El fragmento **es recuperable**: aparece en el puesto 3 cuando la pregunta usa el vocabulario de
la tabla. Lo que no ocurre es que una pregunta operativa con un número dentro lo alcance. Sus
celdas son rangos numéricos escuetos, con casi nulo solapamiento léxico con una pregunta en
prosa.

---

## 7 · Preguntas que fallan en las tres estrategias

7 de 26. Fallar en las tres no dice nada sobre el chunking —las tres tenían el mismo texto—;
dice algo sobre el recuperador o sobre el corpus.

| | etiquetas | mejor puesto | pregunta |
| --- | --- | --- | --- |
| q01 | cbcf, otorgamiento | 9 | Antes de aprobarle un cupo a alguien, ¿en qué se supone que me tengo que basar? |
| q10 | ley-1266, habeas-data | 7 | ¿Puedo consultar la central de riesgo para estudiar a un solicitante, o necesito que él me firme una autorización? |
| q14 | **cross-lingual**, basilea | fuera del top-10 | ¿Quién debería confirmar la calificación de riesgo de un crédito: el mismo que lo originó u otra área? |
| q16 | **cross-lingual**, basilea | fuera del top-10 | Si la garantía es muy buena, ¿puedo ser menos exigente con el análisis del deudor? |
| q18 | numérica | fuera del top-10 | El puntaje me dio 0,19. ¿Qué hago con esa solicitud? |
| q19 | numérica | fuera del top-10 | Un solicitante quedó en 0,42. ¿Lo puedo mandar a comité para que lo aprueben? |
| q20 | numérica | fuera del top-10 | Me salió 0,13 en un solicitante. ¿Se le aprueba lo que pidió o con algún ajuste? |

**Dos de las cuatro preguntas cross-lingual fallan en las tres estrategias.** De las cuatro
preguntas formuladas en español cuya respuesta está en el documento en inglés: q14 y q16 fallan
en las tres; **q15 acierta solo en A**; y **q17 acierta en B y C pero no en A** —y q17 es además
la única de las cuatro que tiene también un fragmento anotado en español—. La recuperación entre
idiomas **funciona, y de forma mucho menos fiable de lo que sugería la consulta única del turno
anterior**, que acertó en los tres primeros puestos.

---

## 8 · Control de contaminación del set de preguntas

Fracción de las palabras de contenido de cada pregunta que también aparecen en el texto de su
fragmento anotado. Un set redactado copiando el vocabulario de los chunks puntuaría alto aquí, y
su hit@k mediría la copia y no la recuperación.

| documento del fragmento anotado | media | mediana | máx | n |
| --- | ---: | ---: | ---: | ---: |
| `basilea-principios-riesgo-credito` | 0,100 | 0,000 | 0,400 | 4 |
| `circular-basica-contable-sfc-cap-ii` | 0,276 | 0,286 | 0,500 | 7 |
| `ley-1266-2008-habeas-data` | 0,169 | 0,125 | 0,286 | 9 |
| `politica-interna-credito` | 0,254 | 0,292 | 0,500 | 14 |

Preguntas de mayor solapamiento: q02 (0,500), q24 (0,500), q25 (0,500), q26 (0,429), q17 (0,400).

Dos lecturas:

1. **La debilidad anticipada no apareció.** La política interna la redactó el mismo autor que
   redactó las preguntas, un turno antes, y ese era el punto donde cabía esperar más
   contaminación. Su media (0,254) es **menor** que la de la circular (0,276), que ningún
   participante escribió.
2. **La cifra de Basilea no es comparable.** Sus preguntas están en español y sus fragmentos en
   inglés, así que el solapamiento léxico es cercano a cero por construcción del idioma y no por
   disciplina de redacción. Para esas cuatro preguntas este control no mide nada.

---

## 9 · Qué NO cubre esta medición

- **El tamaño del set.** 26 preguntas con respuesta y 3 sin ella. Alcanza para detectar
  diferencias grandes y para mostrar solapamientos; no alcanza para separar B de C, ni para
  estimar un umbral.
- **Un solo modelo de embeddings.** Todo lo anterior es específico de `multilingual-e5-base`. El
  efecto de homogeneización del header podría comportarse distinto en otro modelo.
- **Un solo corpus de cuatro documentos.** El enrutamiento entre documentos es fácil con cuatro
  y difícil con cuatrocientos; el beneficio del header sobre doc@1 podría valer mucho más a otra
  escala, y el costo sobre unidad@5 también.
- **Un solo juego de parámetros por estrategia.** No se barrieron `max_body_chars`,
  `overlap_chars`, ni el tamaño de ventana de la línea base.
- **La reordenación posterior.** No se probó ningún *reranker* ni búsqueda híbrida con BM25, que
  es la defensa habitual contra el fallo léxico de la sección 6.
- **El índice de producción.** Se midió con búsqueda exacta; el almacén persistente usa HNSW.
- **La citabilidad.** Un chunk de la estrategia A es un artículo y se puede citar; una ventana de
  la estrategia C no corresponde a nada citable. Es una propiedad, no una métrica, y ninguna
  cifra de este documento la refleja.
