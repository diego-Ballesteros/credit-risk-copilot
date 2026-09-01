# Metodología de trabajo — Credit Risk Copilot

> Adaptación de la metodología Arquitecto / Ejecutor a un proyecto de Machine Learning
> con doble objetivo: entregar un sistema completo y **entender lo que se entrega**.
>
> Este documento es el **dueño del contenido**. `CLAUDE.md` es un puntero corto hacia aquí.
> Si algo se contradice entre los dos, este manda.

---

## 1. Los tres roles

| Rol | Quién | Qué hace | Qué NO hace |
|---|---|---|---|
| **Arquitecto** | Claude en el chat del proyecto | Diagnostica, debate, decide, enseña, redacta prompts | **Nunca escribe código.** No ve el repositorio |
| **Ejecutor** | Claude Code, dentro del repo | Implementa, verifica, reporta con evidencia | **Nunca commitea.** No toma decisiones técnicas |
| **Verificador** | Yo | Valido, decido, explico, commiteo | No implemento |

### Por qué el Arquitecto no ve el código

Parece una limitación y es el mecanismo de calidad del proyecto.

**Obliga a que la documentación sea buena**, porque es lo único que el Arquitecto tiene. Un
documento desactualizado se detecta de inmediato: el Arquitecto propone algo que ya existe, o
razona sobre una garantía que no está.

En este proyecto la documentación es además el 10% de la nota y un requisito explícito del
entregable (README, Model Card, diccionario de datos, ADRs). La restricción del Arquitecto y el
requisito de la rúbrica apuntan al mismo lado.

### Por qué solo yo commiteo

Un commit es una afirmación de que algo funciona. Además, en este proyecto **los commits son
evidencia evaluada**: la rúbrica mira el historial como prueba del proceso. Un commit hecho por un
agente es evidencia falsificada.

### La compuerta de aprendizaje

Regla propia de este proyecto, no está en la metodología original:

> **No se mergea nada que yo no pueda explicar en voz alta.**

Si no puedo explicar por qué una función existe, qué hace una transformación o por qué elegimos una
métrica, el paso no está terminado — está *escrito*, que no es lo mismo. En ese caso el siguiente
turno es una pregunta al Arquitecto, no un prompt al Ejecutor.

---

## 2. El ciclo

```
Objetivo del día (viene del roadmap)
   ↓
DIAGNÓSTICO / MEDICIÓN (solo lectura)   ← el paso que más se saltea
   ↓
DEBATE con el Arquitecto                ← decisiones, sin código de por medio
   ↓
PROMPT estructurado                     ← un solo paso, nunca dos
   ↓
EJECUCIÓN + REPORTE con evidencia       ← salida real de comandos, no afirmaciones
   ↓
VERIFICACIÓN + EXPLICACIÓN              ← ¿funciona? ¿lo entiendo?
   ↓
COMMIT
```

### Medir antes de modelar

El equivalente del "diagnóstico antes de diseñar" en un proyecto de ML:

- EDA antes de diseñar el pipeline
- Baseline trivial antes de cualquier modelo
- Medir la estrategia antes de adoptarla (¿SMOTE ayuda? se mide, no se asume)

En la práctica, esta medición casi siempre devuelve algo que invalida el plan.

### Un paso a la vez

**Nunca dos prompts de ejecución juntos.** Un paso puede destapar algo que hay que resolver antes
del siguiente, y con los dos entregados esa información llega tarde.

Cuando el reporte del Ejecutor pide verificación, el Arquitecto la traduce a un guion ejecutable:
qué comando correr, qué debería salir, y qué contaría como incorrecto. **La verificación nunca se
deja enunciada en abstracto.**

---

## 3. Anatomía de un prompt al Ejecutor

```
## CONTEXT
Qué existe, dónde estamos, y POR QUÉ las decisiones ya tomadas son las que son

## TASK
Una frase — qué tiene que ser cierto al terminar

## STRUCTURE TO CREATE (si aplica)
Árbol exacto de archivos

## RULES (do not deviate)
Decisiones técnicas no negociables, incluidos los límites de qué NO tocar

## FILE SPECS
Especificación de cada archivo: qué hace, no cómo se escribe

## AFTER CREATING ALL FILES
Comandos de verificación a correr, con la salida pegada en el reporte
```

### Las razones viajan con las instrucciones

Un prompt que dice *"usá `random_state=42`"* produce código correcto. Uno que dice *"usá
`random_state=42` en todo, y tomalo de `config.py` en vez de hardcodearlo, porque necesitamos que
un tercero reproduzca la métrica exacta desde cero"* produce código correcto **y un Ejecutor capaz
de decidir bien en el caso que no previste**.

### Los límites explícitos

Decir qué **no** tocar es tan importante como decir qué hacer. Frase de uso frecuente:

> *"Si encontrás X, reportalo y NO lo arregles."*

Separa el hallazgo de la decisión, que es de quien lee el reporte.

### Estructura antes que código

Para cualquier trabajo con forma —un módulo nuevo, un pipeline, un grafo— el prompt pide que el
Ejecutor **proponga la estructura al principio del reporte y siga adelante con ella**. Si la
estructura está mal, se ve en la primera línea en vez de después de 400 líneas.

### Enrutamiento de modelo

| Modelo | Cuándo |
|---|---|
| El más liviano | Tareas mecánicas: renombrar, mover, formatear, docstrings |
| El intermedio | La mayoría del trabajo: módulos, tests, scripts |
| El más capaz | Pipeline de preprocesamiento, diseño del grafo del agente, cualquier cosa donde un error se propaga en silencio a todas las métricas |

Criterio: **no es la dificultad, es el costo del error.** Un `ColumnTransformer` mal armado no
rompe nada visible — envenena cada número del proyecto.

### Higiene de chats

- **Un chat de Arquitecto por fase.** El contexto se traspasa por `CONTEXT.md`, no por memoria.
- **Una sesión de Ejecutor por prompt**, salvo que el siguiente dependa de hallazgos del anterior.
- Un chat que arrastra tres temas empieza a mezclar contextos.

---

## 4. Arquitectura de documentos

Cada clase de documento tiene un contrato distinto. **La clave es que no se solapen**: el mismo
contenido en dos lados diverge, porque nadie actualiza dos veces.

| Documento | Contrato |
|---|---|
| `README.md` | El entregable. Qué es el proyecto, para el evaluador y para un tercero. **También el contrato de comportamiento**: qué hace el sistema y qué lo restringe, con sus dos diagramas de flujo. **Cero citas `archivo:línea`** |
| `docs/DATA_DICTIONARY.md` | Contrato de datos: columnas, tipos, rangos, categorías, incluidas las no documentadas por la fuente |
| `docs/MODEL_CARD.md` | Qué hace el modelo, con qué datos, qué NO debe usarse para, y sus limitaciones |
| `docs/adr/` | Una entrada por decisión, con su razón. **Nunca se corrige hacia atrás** |
| `docs/analysis/` | Mediciones de registro sobre los datos, cada una **reproducible por un script del repositorio**. Sus cifras se citan desde los ADR. **No se reescriben hacia atrás** cuando una regla posterior cambia: se les agrega una **nota de estado fechada**, porque son el registro de la evidencia sobre la que se decidió |
| `docs/ERRORS_AND_LEARNINGS.md` | Una entrada por error real, con **el mecanismo** — no solo el síntoma |
| `docs/EVALUATION.md` | Qué se midió, cómo, y con qué resultado |
| `docs/GIT_STRATEGY.md` | Estrategia de ramas, commits, pull requests, releases y convención de idioma |
| `CLAUDE.md` | Reglas de proceso que el Ejecutor lee al empezar. **Corto**, o deja de leerse |
| `docs/METHODOLOGY.md` | Dueño de la metodología de trabajo. Ante contradicción con `CLAUDE.md`, **este manda** |

> **No hay `docs/ARCHITECTURE.md`, y es una decisión y no un olvido (2026-08-31).** Este
> documento planificaba uno separado para el contrato de comportamiento. El README terminó
> cubriéndolo —qué hace el sistema, el recorrido de datos crudos a copiloto, el grafo del
> copiloto, y qué restringe a cada pieza— así que un archivo aparte **duplicaría ese
> contenido**, que es exactamente lo que prohíbe la regla de no solapamiento con la que abre
> esta sección: el mismo contenido en dos lados diverge, porque nadie actualiza dos veces. La
> fila se retira en vez de dejarse apuntando a un archivo que no existe, porque un contrato
> declarado y no cumplido es peor que un contrato ausente.

### Reglas duras

- **Cero números de línea** en cualquier documento. Nombrar un módulo está permitido; `loader.py:42`
  está prohibido. No sobrevive a un solo refactor.
- **Los ADRs no se reescriben.** Cuando una decisión deja de ser válida se marca `superseded` con la
  razón. La decisión fue correcta con la información de entonces; borrar eso destruye el porqué.
- **Las afirmaciones negativas se reverifican siempre.** *"El dataset no tiene valores faltantes"* es
  cierta el día que se escribe y puede ser falsa después de un cambio de fuente. Son las que más
  daño hacen.
- **El razonamiento de un ADR lo produce el Arquitecto, no el Ejecutor.** La justificación sale de
  una decisión discutida en el chat del proyecto. El Ejecutor **transcribe y formatea; nunca redacta
  la justificación desde cero**. La razón: un ADR con justificación inventada se lee bien y no
  registra nada, y es peor que no tenerlo, porque el autor del proyecto lo leerá semanas después
  creyendo que es su propio razonamiento. **Un ADR cuyo contenido no salió de una decisión discutida
  no se escribe.**
- **El idioma de un documento es el de su lector.** En **inglés**: identificadores, funciones,
  clases, variables, docstrings, comentarios en código, nombres de archivos y carpetas, nombres de
  ramas y mensajes de commit. En **español**: `CLAUDE.md`, `docs/METHODOLOGY.md`, `docs/ROADMAP.md`,
  `README.md`, `docs/GIT_STRATEGY.md`, `docs/EVALUATION.md`, `docs/ERRORS_AND_LEARNINGS.md` y todos
  los ADRs, porque los lee el autor del proyecto y un evaluador hispanohablante. Los nombres de
  columnas de los datasets se conservan **como los entrega la fuente**; lo que va en español es su
  descripción.

### Estándares de código

- **Type hints en toda función.** Docstrings en toda función pública.
- **`pathlib.Path` para toda ruta.** Nunca strings concatenados.
- **`random_state` se toma de `config.py`.** Nunca se hardcodea, porque la reproducibilidad exacta
  desde cero es un requisito **verificado y no supuesto**: un tercero clona el repo, corre los
  scripts en orden y las métricas coinciden.
- **Credenciales solo desde `.env`.** Nunca en el código.

Estas cuatro reglas ya están forzadas por las herramientas — `ruff` con las reglas `D` y `PTH`, y
`mypy` con `disallow_untyped_defs` — y aun así se escriben aquí. La razón es de arquitectura de
documentos, no de redundancia: **`CLAUDE.md` es un puntero y no puede ser la fuente de una regla**,
y una regla sin documento dueño no se puede citar ni discutir. Una configuración impide, pero no
explica; cuando alguien pregunte *por qué* no puede concatenar rutas, la respuesta tiene que estar
en algún lado.

---

## 5. Protocolo de auto-documentación

Al final de cada turno, el Ejecutor **evalúa esta lista y reporta cuáles se dispararon**:

1. ¿Se tomó una decisión técnica con alternativas descartadas? → ADR
2. ¿Cambió el esquema de datos o el contrato de una función pública? → actualizar el documento que lo afirma
3. ¿Se agregó o modificó una feature? → actualizar el diccionario de datos
4. ¿Cambió una métrica del modelo? → registrar el run en MLflow y actualizar `EVALUATION.md`
5. ¿Hubo un error real, con diagnóstico? → `ERRORS_AND_LEARNINGS.md`, **con el mecanismo**
6. ¿Se tocó algo que un documento afirma? → releer ese documento y corregirlo
7. ¿Se agregó una dependencia? → justificarla en el reporte

El criterio 6 es el más importante y el más difícil. Formulación ejecutable: *al cerrar un tema,
buscarlo por su nombre en los documentos*. Eso es un `grep`, no un acto de memoria.

---

## 6. Disciplina de verificación

Esta sección es la que separa "hacer un proyecto de ML" de "pedirle código a un modelo".

### 6.1 · Una métrica que no puede fallar no prueba nada

Nuestro dataset tiene ~22% de clase positiva. Un modelo que siempre predice "no incumple" saca 78%
de accuracy. **Ese número no prueba nada.**

Prácticas obligatorias:

- **Baseline trivial primero.** Antes de cualquier modelo, medir el piso. Todo se compara contra él.
- **Prueba del target barajado.** Entrenar el pipeline completo con el target permutado
  aleatoriamente. El desempeño debe colapsar al nivel del azar. **Si no colapsa, hay leakage.** Es
  verificación por mutación aplicada a ML y es la forma más barata de detectar fugas sin buscarlas
  a mano.
- **Cada métrica reportada viene con su baseline al lado.** Un número solo no significa nada.

### 6.2 · Medir, no estimar

*"SMOTE debería ayudar"* no es un resultado. *"SMOTE bajó PR-AUC de 0.54 a 0.51 en CV de 5 folds"*
sí lo es.

Y cuando algo no se puede medir, decirlo: *"no puedo determinar si el modelo generaliza fuera de
tiempo porque el dataset no tiene fecha de originación"* es infinitamente más útil que una
conjetura presentada como hallazgo.

### 6.3 · Un solo pipeline, dos consumidores

El modo de falla más traicionero en ML de producción: el preprocesamiento del notebook y el del
script de servicio divergen. Todo verde en el notebook, predicciones basura en la API.

Defensas:

- **Un solo objeto `Pipeline`**, serializado y versionado en MLflow. El notebook, el script de
  entrenamiento y la API cargan **el mismo artefacto**. Nunca se reimplementa la transformación.
- **Serialización real** en los tests: `dump`/`load` de verdad, no un objeto que se conserva en
  memoria.
- **Test de ida y vuelta**: fila cruda → pipeline cargado desde disco → predicción. Sin construir el
  estado intermedio a mano en ningún punto.

### 6.4 · Una superficie nueva no se cierra sin una llamada real

La API no está lista porque los tests unitarios pasan. Está lista cuando una petición HTTP real,
con un payload real, devuelve una predicción correcta. Lo mismo para el agente: no está listo hasta
que una consulta real recorre el grafo completo.

### 6.5 · El leakage se hace imposible, no se recuerda

Jerarquía, de más fuerte a más débil:

1. **La herramienta lo impide** — el `Pipeline` de sklearn hace imposible escalar antes del split
2. **Un test lo detecta** — la prueba del target barajado
3. **Está escrito en un documento** — se cumple mientras alguien lo recuerde

**El nivel 3 falla.** Antes de escribir una regla, preguntarse si puede subir de nivel.

### 6.6 · Reproducibilidad verificada, no supuesta

`random_state=42` en todo, tomado de `config.py`, nunca hardcodeado. Y al menos una vez antes de
entregar: clonar el repo en un directorio limpio, correr los scripts en orden, y confirmar que las
métricas coinciden.

---

## 7. Modos de falla propios de ML, y sus defensas

### 7.1 · La imputación silenciosa

Un `NaN` convertido en `0` deja de ser *"no sé"* y pasa a ser *"ingreso cero"*, que es un hecho de
negocio. El modelo aprende sobre una mentira bien formada, y ningún chequeo de error la detecta.

**Defensa:** toda imputación se declara explícitamente en el diccionario de datos y en el Model
Card, y cuando aplica se acompaña de una columna indicadora de que el valor fue imputado. La
ausencia de dato suele ser informativa por sí sola.

### 7.2 · La categoría no documentada

Familia distinta de la anterior, y ningún chequeo la detecta **porque el dato llega bien formado**.
Nuestro dataset tiene valores en `PAY_0` y `EDUCATION` que no aparecen en la documentación oficial
de la fuente. Se leen como enteros perfectamente válidos y significan algo que no sabemos.

**Defensa:** el validador de datos verifica el conjunto de categorías contra el diccionario y
**falla ruidosamente** ante una categoría desconocida. Lo que se decida hacer con ella se registra
como ADR, no se resuelve en silencio dentro de una función.

### 7.3 · La métrica sin contexto

Un AUC de 0.78 no dice nada sin saber contra qué se compara, con cuántos folds, con qué varianza y
sobre qué población.

**Defensa:** toda métrica se reporta con baseline, número de folds, desviación estándar entre folds
y tamaño de muestra. Un número solo es una opinión.

### 7.4 · El notebook como fuente de verdad

Un notebook con estado acumulado en memoria produce resultados que nadie puede reproducir, porque
dependen del orden en que se ejecutaron las celdas.

**Defensa:** los notebooks son **exploración y narrativa**, nunca implementación. Toda lógica vive
en `src/` y el notebook la importa. Antes de cada commit: *Restart & Run All*. Si no corre de
arriba a abajo, no se commitea.

### 7.5 · El proceso de dos pasos donde el segundo se olvida

Entrenar y registrar. Cambiar la feature y actualizar el diccionario. Mejorar el modelo y actualizar
el Model Card. **Nada avisa cuando falta el segundo.**

**Defensa:** el protocolo de la sección 5, ejecutado al final de cada turno.

---

## 8. Señales de que la metodología funciona

- El Ejecutor **corrige premisas del Arquitecto con evidencia** en vez de aceptarlas
- Los reportes distinguen **lo que se midió de lo que se infirió**
- Aparecen hallazgos que nadie buscaba
- Se declaran **intercambios** en vez de venderlos como mejoras puras
- Puedo explicar cualquier archivo del repo sin volver a abrirlo

## 9. Señales de que se está degradando

- Los prompts empiezan a mezclar dos temas
- Los reportes dicen "todo verde" sin pegar la salida de los comandos
- Se saltea el EDA o el baseline porque "ya sabemos qué va a dar"
- Aparece lógica de negocio dentro de un notebook
- Se acepta una métrica sin su baseline al lado
- Mergeo algo que no podría explicar

---

*Documento vivo. Actualizar cuando un error real enseñe algo que no está aquí.*