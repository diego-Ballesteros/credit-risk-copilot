# Errores y aprendizajes

Una entrada por error real que costó tiempo, documentando el **mecanismo** que lo
produjo y no solo el síntoma: un registro que dice "fallaba el import" no sirve a
nadie dentro de tres semanas; uno que dice por qué fallaba, sí.

## Formato de entrada

    ### NNN — Título corto del error

    - **Fecha:** AAAA-MM-DD
    - **Fase:** NN-nombre-de-la-fase
    - **Síntoma:** qué se observó, con el mensaje de error literal si lo hubo.
    - **Causa raíz:** el mecanismo subyacente. Por qué el sistema se comportó así.
    - **Diagnóstico:** cómo se llegó a la causa. Qué se descartó por el camino.
    - **Solución:** el cambio concreto que lo resolvió, con el commit o PR.
    - **Prevención:** qué test, hook, tipo o regla impide que vuelva a ocurrir.
    - **Aprendizaje:** la regla generalizable que queda para el resto del proyecto.

Las entradas se numeran de forma consecutiva y no se reescriben ni se borran: una
entrada corregida se corrige con una entrada nueva que referencia a la anterior.

---

### 001 — El filtro de ramas de GitHub Actions silencia el check sin dar error

- **Fecha:** 2026-08-24
- **Fase:** 00-fundacion

- **Síntoma:** la CI no aparece como check en la pull request. No hay mensaje de error,
  no hay workflow en rojo, no hay anotación. Todo lo demás sale en verde: `ruff`, `mypy`
  y `pytest` pasan en local, el YAML es sintácticamente válido y GitHub no se queja de
  nada. La ausencia del check es el único indicio, y es un indicio que hay que ir a
  buscar.

- **Causa raíz:** GitHub evalúa el filtro `branches` del evento **antes** de decidir si
  crea una ejecución. Si el nombre de la rama no coincide con ninguna entrada del
  filtro, **descarta el evento en silencio y no crea ningún check**. No hay error: hay
  ausencia. Y una ausencia se parece mucho a un éxito cuando lo que uno mira es una
  lista de checks sin nada en rojo. En este caso el workflow declaraba
  `branches: [main, development]` mientras la rama de integración se llamaba de otra
  forma, así que el evento nunca produjo ejecución.

- **Diagnóstico:** no se detectó leyendo el YAML, que era correcto, sino **consultando
  el estado real del remoto** con `git ls-remote --heads origin` en vez de dar por
  cierta la premisa del prompt sobre cómo se llamaba la rama. La consulta devolvió los
  nombres de rama que existían de verdad, y de ahí salió la contradicción con el filtro.

- **Solución:** eliminar el filtro `branches` del evento `pull_request` en
  `.github/workflows/ci.yml`. El filtro se mantiene solo en `push`, donde su única
  consecuencia es no ejecutar de más. Toda pull request dispara la CI, sin importar la
  rama destino.

- **Prevención:** aplica la jerarquía de la sección 6.5 de la metodología. La garantía
  pasa de **nivel 3** — *"acordarse de sincronizar el nombre de la rama con el filtro"*,
  que se cumple mientras alguien lo recuerde — a **nivel 1**: sin filtro, **ningún
  nombre de rama puede silenciar el check**. El modo de falla queda imposible en vez de
  quedar recordado.

- **Aprendizaje:** una configuración que filtra eventos falla en silencio por diseño, así
  que no basta con que sea válida: hay que verificar que **coincide con el estado real
  del sistema**. Y de forma más general, cuando el único síntoma de un fallo es la
  ausencia de algo, el fallo no se detecta mirando lo que hay; se detecta preguntando
  explícitamente por lo que debería estar.

- **Nota — lección de proceso:** el nombre de la rama llegó afirmado como hecho en el
  bloque `CONTEXT` del prompt, y era falso. **Una precondición no verificada no se
  afirma como hecho en el CONTEXT de un prompt; se verifica dentro del turno.** El
  Ejecutor tiene acceso al repositorio y el Arquitecto no, así que comprobar el estado
  real es responsabilidad del Ejecutor incluso — sobre todo — cuando el prompt ya lo da
  por resuelto.

---

### 002 — Un notebook sintácticamente roto sigue siendo un JSON perfectamente válido

- **Fecha:** 2026-08-25
- **Fase:** 01-data-and-eda

- **Síntoma:** una celda de código con un error de sintaxis **no produce ningún fallo**
  al escribir el archivo. El `.ipynb` se guarda sin quejas, se abre sin quejas y se
  versiona sin quejas. `ruff` lo lintea y pasa. El error aparece recién cuando un kernel
  intenta ejecutar esa celda, y aparece como `SyntaxError: unterminated string literal`
  sobre un fragmento de código que en el archivo se ve perfectamente normal.

- **Causa raíz:** el formato `.ipynb` **guarda el código fuente como cadenas de texto
  dentro de un JSON**. Para todo validador de archivos, un notebook cuyo código no compila
  sigue siendo un documento JSON bien formado: las llaves cierran, las comillas balancean,
  el esquema de nbformat se cumple. **La sintaxis de Python nunca se verifica al escribir,
  porque en ese momento el código no es código: es el contenido de un campo de texto.**
  No hay ninguna etapa entre "escribir el archivo" y "arrancar un kernel" donde alguien
  compile ese contenido.

  En el caso concreto, el notebook se generó por programa y un escape `\n` destinado a
  formar parte de una cadena de Python dentro de una celda se resolvió una vez de más al
  escribir el archivo, convirtiéndose en un salto de línea real. La celda quedó con una
  cadena abierta. El archivo quedó impecable.

- **Diagnóstico:** no se detectó leyendo el archivo, que se veía bien, ni con las
  herramientas de calidad, que pasaron. Se detectó **ejecutando el notebook completo con
  `nbconvert --execute`**, que abortó nombrando la celda y la línea. Por el camino se
  descartó que fuera un problema de codificación del archivo y de escapado del shell: la
  inspección con `repr()` de las líneas del generador mostró que el archivo contenía un
  solo carácter de barra invertida donde hacían falta dos.

- **Solución:** eliminar el escape en vez de arreglarlo — un `print()` vacío seguido de un
  `print("…")`, que no necesita ningún `\n` embebido y por tanto no puede volver a
  romperse por la misma vía. Además, mientras el generador existió, se le agregó un
  `compile()` de cada celda de código **antes** de escribir el archivo, para que el fallo
  apareciera al construir y no al ejecutar.

- **Prevención:** la defensa que queda en el repositorio no es el generador —era
  herramienta descartable y no se conservó— sino **la verificación del notebook**, en dos
  partes:

  1. **Ejecutarlo entero con un kernel limpio**, con
     `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_preprocessing.ipynb`.
     La bandera de ejecución lanza un kernel nuevo y corre las celdas en orden; aborta si
     alguna falla. Es un *Restart & Run All* sin intervención humana.
  2. **Comprobar que los contadores de ejecución guardados forman una secuencia
     consecutiva desde uno.** Un notebook cuyos contadores no son consecutivos **no es el
     resultado de una corrida limpia**, y entonces cualquier cifra que muestre puede venir
     de un estado que nadie puede reconstruir — que es exactamente el modo de falla de la
     sección 7.4 de `docs/METHODOLOGY.md`.

  En la jerarquía de la sección 6.5, esto es hoy una garantía de **nivel 2**: hay una
  comprobación que lo detecta, pero alguien tiene que correrla. Subirla a nivel 1
  requeriría que la corriera la CI en cada pull request, y **esa decisión no está tomada**.

- **Aprendizaje:** **un archivo válido no es un archivo correcto, y cuanto más envuelto
  está el contenido, más tarde se entera uno.** Un `.py` roto lo detecta el import; un
  `.ipynb` roto no lo detecta nada hasta que hay un kernel. La regla generalizable: cuando
  un formato *contiene* código en vez de *ser* código —notebooks, YAML con expresiones,
  plantillas, SQL en una cadena—, la validación del contenedor no dice nada sobre el
  contenido, y hace falta una verificación que lo ejecute de verdad.

  Se conecta con la entrada 001 por el mismo lado: allí el síntoma era **la ausencia de un
  check**, acá es **la ausencia de una validación**. Los dos fallos son invisibles mirando
  lo que hay, y los dos se detectan preguntando explícitamente por lo que debería estar.

---

### 003 — `id()` no es una prueba de identidad: el intérprete recicla direcciones

- **Fecha:** 2026-08-26
- **Fase:** 02-modeling

- **Síntoma:** un test que afirmaba que **cada fold recibe una instancia nueva del
  preprocesador** pasaba en verde sin verificar nada. Estuvo tres turnos en verde. Cuando
  por fin falló, lo hizo con un mensaje que a primera vista parece delatar un bug del
  código bajo prueba:

      assert 3 == 4
       +  where 3 = len({1556906293712, 1556906294480, 1556906512784})
       +  and   4 = len([1556906294480, 1556906512784, 1556906293712, 1556906294480])

  Cuatro folds, cuatro instancias, **tres direcciones distintas**: una aparece dos veces.
  Leído sin más, dice "el preprocesador se está reutilizando entre folds", que es
  exactamente el modo de fuga que el test existe para impedir. No era eso.

- **Causa raíz:** en CPython, `id()` devuelve **la dirección de memoria del objeto**, y esa
  dirección **se reutiliza en cuanto el objeto anterior es recolectado**. El contrato de
  `id()` es que dos objetos *vivos a la vez* tienen identificadores distintos; no dice nada
  sobre objetos que no coexisten.

  Los preprocesadores de este test viven exactamente lo que dura su fold y mueren al
  terminarlo. Nunca coexisten. De modo que el test comparaba direcciones que el intérprete
  había reciclado, y su resultado dependía del momento de asignación de memoria y no de
  ninguna propiedad del código.

  Lo que hace el fallo especialmente traicionero es la dirección del error: **el test es
  laxo, no estricto**. Un test frágil que falla de más molesta y se arregla. Éste **pasaba**
  cuando la asignación no reciclaba, que es la mayoría de las veces, y por eso pareció
  correcto durante tres turnos.

- **Diagnóstico:** no se dedujo leyendo el código del test sino **midiendo el
  comportamiento de `id()`** con seis objetos de vida corta:

      class T: pass
      ids = []
      for _ in range(6):
          t = T(); ids.append(id(t)); del t
      # ids -> las seis direcciones son la misma

  **Seis objetos, una sola dirección distinta.** Con esa medición el mensaje del fallo deja
  de leerse como un bug del preprocesador y pasa a leerse como lo que era.

  Se descartó por el camino la hipótesis de que la fábrica estuviera devolviendo un objeto
  reutilizado: la fábrica construye una instancia nueva en cada llamada, y el contador de
  llamadas —que es una variable ordinaria y no una dirección— seguía marcando cuatro.

- **Solución:** identificar cada instancia por un **número de serie asignado en la
  construcción**, entregado por el diario de la prueba, en vez de por su dirección. El
  número de serie sigue siendo único mientras el diario viva, aunque el objeto ya no exista.
  El test comprueba además que se construyeron tantas instancias como ajustes se
  registraron. Verificado estable en tres corridas seguidas.

- **Prevención:** el arreglo sube la garantía de **nivel 3** —*"acordarse de que `id()` es
  frágil"*— a **nivel 1** dentro de este test: un contador asignado en construcción no puede
  reciclarse, así que el modo de falla desaparece en vez de quedar recordado.

  Lo que **no** hay es nada que impida escribir el mismo error en otro test. Un `grep` de
  `id(` en `tests/` es la comprobación barata, y no está automatizada; subirla a nivel 1
  requeriría una regla de lint propia, y **esa decisión no está tomada**.

- **Aprendizaje:** **cualquier prueba de identidad basada en `id()` sobre objetos de vida
  corta es una prueba que no prueba nada.** La regla generalizable: `id()` responde
  "¿son el mismo objeto?" solo entre objetos que están vivos simultáneamente. En cuanto la
  vida de los objetos no se solapa —un bucle que crea, usa y descarta, que es la forma
  exacta de una validación cruzada— la respuesta se vuelve una función del asignador de
  memoria. Para identidad a lo largo del tiempo hace falta algo que el objeto **lleve
  consigo**: un número de serie, un UUID, un contador.

  Se conecta con las entradas 001 y 002 por el mismo lado, y esta vez desde el otro extremo.
  Allí el problema era una **ausencia** que se parecía a un éxito; aquí es una **afirmación
  vacía** que se parece a una verificación. Los tres fallos comparten que **el verde no
  significaba lo que parecía significar**, y los tres se detectaron preguntando
  explícitamente qué estaba comprobando el verde.

---

### 004 — El paquete instala sin error y falla al importar: la DLL está, el runtime no

- **Fecha:** 2026-08-30
- **Fase:** 03-genai *(la primera ocurrencia fue en 02-modeling, 2026-08-26)*

- **Síntoma:** `uv add sentence-transformers` termina en verde. El primer `import` revienta:

      OSError: [WinError 126] No se puede encontrar el módulo especificado.
      Error loading "...\.venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.

  El mensaje señala un archivo **que existe**. `c10.dll` está donde dice, con el tamaño que le
  corresponde, y aun así "no se puede encontrar el módulo".

- **Causa raíz:** el wheel trae la biblioteca dinámica pero **no el runtime del que ella
  depende**. `c10.dll` está compilada contra el runtime de Microsoft Visual C++ y lo enlaza
  dinámicamente, así que cargarla exige `vcruntime140.dll`, `vcruntime140_1.dll`,
  `msvcp140.dll` y `msvcp140_1.dll` en el sistema. Este equipo tenía **únicamente las
  variantes `*_clr0400`** —`msvcp140_clr0400.dll`, `vcruntime140_clr0400.dll`—, que son las
  que empaqueta .NET y **no sirven** como el runtime de C++ que busca el cargador.

  El mensaje de Windows dice "no se puede encontrar el módulo" refiriéndose a **una
  dependencia** de la DLL nombrada, no a la DLL nombrada. Esa ambigüedad es la mitad del
  problema: manda a verificar el archivo que sí está.

  Y la razón de que el gestor de paquetes reporte éxito es que **instalar y cargar son
  operaciones distintas**. `uv` descarga, verifica el hash y coloca los archivos; nunca ejecuta
  el cargador dinámico del sistema operativo. Un paquete puede quedar perfectamente instalado
  y ser imposible de importar, y ninguna herramienta de empaquetado está en posición de
  advertirlo.

- **Diagnóstico:** no se dedujo del mensaje sino **comprobando qué había en `System32`**, que
  es la afirmación negativa que el mensaje sugería y no demostraba:

      vcruntime140.dll        MISSING
      vcruntime140_1.dll      MISSING
      msvcp140.dll            MISSING
      msvcp140_1.dll          MISSING
      msvcp140_clr0400.dll    PRESENT
      vcruntime140_clr0400.dll PRESENT

  Cuatro ausentes y las variantes de .NET presentes. Con eso el mensaje deja de leerse como
  "falta `c10.dll`" y pasa a leerse como lo que es.

  Se descartó por el camino la hipótesis de un wheel corrupto —el hash estaba verificado— y la
  de una versión de Python incompatible: el wheel era el correcto para 3.11 en Windows x64.

- **Solución:** `winget install --id Microsoft.VCRedist.2015+.x64` (14.51.36247.0). Las cuatro
  DLL pasaron a estar presentes y `torch 2.13.0+cpu` importó. **La instalación de un runtime
  del sistema se consultó antes de ejecutarla**: es un cambio fuera del repositorio y la
  decisión no le corresponde al Ejecutor.

- **Prevención:** ninguna herramienta del proyecto puede impedir esto, y decirlo es más útil
  que inventar una defensa. Lo que sí queda es **un diagnóstico de dos minutos escrito**: ante
  un `import` que falla nombrando una DLL que existe, la primera comprobación es el runtime en
  `System32`, no el paquete. `scripts/build_rag_index.py` y `scripts/evaluate_retrieval.py`
  atrapan el fallo de carga del modelo y salen con código 1 nombrándolo, en vez de arrastrar
  una traza de importación.

- **Aprendizaje:** **es una clase de error, no un incidente.** Ocurrió **dos veces con cuatro
  días de diferencia** y el mismo mecanismo exacto: el 2026-08-26 con **LightGBM**
  (`lib_lightgbm.dll`, ver la nota del stack en `docs/ROADMAP.md`) y el 2026-08-30 con
  **torch**. La primera vez se resolvió sustituyendo la dependencia, que era barato porque
  `HistGradientBoostingClassifier` hacía lo mismo; la segunda no había sustituto, porque
  `sentence-transformers` era un requisito explícito.

  La regla generalizable: **en Windows, "instala" y "carga" son dos garantías distintas, y el
  verde del gestor de paquetes solo cubre la primera.** Cualquier dependencia con extensiones
  nativas —torch, LightGBM, XGBoost, ONNX Runtime— debe verificarse con un `import` real antes
  de darse por instalada, y ese `import` pertenece a la CI tanto como los tests.

  Efecto colateral que conviene registrar: instalar el runtime **volvió utilizable LightGBM**,
  de modo que la afirmación negativa del ROADMAP caducó. Se corrigió allí en vez de borrarse,
  y la sustitución del ADR-0007 se mantiene porque reejecutarla movería un número dentro del
  ruido.

---

### 005 — Verificación contaminada: tres consultas escritas después de leer los fragmentos

- **Fecha:** 2026-08-30
- **Fase:** 03-genai

- **Síntoma:** al cerrar la construcción del índice vectorial se ejecutaron **tres consultas de
  prueba** contra el corpus. Las tres devolvieron el fragmento correcto **en el primer puesto**,
  con scores entre 0,8690 y 0,8811, y el resultado se reportó como verificación de que el
  sistema recuperaba bien.

  Un turno después, un set de **26 preguntas con respuesta** anotadas a mano midió sobre el
  mismo índice y la misma estrategia: **hit@1 = 0,346**. Una de cada tres, no tres de tres.

- **Causa raíz:** las tres consultas **se redactaron después de leer los fragmentos**, en la
  misma sesión en que se había transcrito el corpus. Compartían vocabulario con su fragmento
  objetivo porque el vocabulario del fragmento estaba a la vista al escribirlas.

  El mecanismo, dicho sin rodeos: **quien escribe la consulta mirando el chunk mide su propia
  memoria y no el sistema.** Un recuperador denso empareja superficies; una consulta construida
  con las palabras del documento le entrega precisamente la superficie que necesita. El
  experimento estaba resuelto antes de ejecutarse.

  Lo que lo hace especialmente traicionero es que **el fallo es en la dirección optimista y no
  produce ningún síntoma**. Un test frágil que falla de más molesta y se arregla. Aquí los tres
  resultados eran correctos, los scores altos, las citas exactas, y todo el conjunto se leía
  como una demostración de que el componente funcionaba.

- **Diagnóstico:** no se detectó revisando las consultas —seguían pareciendo razonables— sino
  al **construir un set con un procedimiento distinto** y ver la diferencia de magnitud. La
  confirmación vino de medir el **solapamiento léxico** entre pregunta y fragmento anotado:
  0,100 en el documento en inglés, 0,169 en la Ley 1266, 0,254 en la política interna y 0,276
  en la Circular Básica. Las tres consultas originales estaban muy por encima de ese rango,
  porque estaban hechas de las mismas palabras.

- **Solución:** el set de `data/eval/retrieval_questions.yaml` sustituye a las tres consultas
  como evidencia. Las cifras del turno anterior no se borran: quedan como lo que eran, una
  demostración de funcionamiento y no una medición.

- **Prevención:** cuatro reglas, y la última es la que convierte a las otras tres en algo
  verificable en vez de una promesa.

  1. **Enumerar las tareas del usuario antes que los contenidos del corpus.** La pregunta sale
     de lo que un analista necesita hacer, no de lo que un documento dice.
  2. **Redactar en el registro del usuario y no en el del documento.** "Puntaje" y no *score*;
     "reporte negativo" y no *dato negativo*; "que le puede afectar la plata" y no *que pueda
     afectar su capacidad de pago*.
  3. **Anotar el fragmento correcto antes de ejecutar ninguna búsqueda.** Anotar después es
     anotar lo que el sistema encuentra.
  4. **Medir el solapamiento léxico entre pregunta y fragmento, y reportarlo.**
     `scripts/evaluate_retrieval.py` lo calcula en cada corrida. Un set contaminado puntúa alto
     ahí, y la regla 3 deja de depender de la palabra de quien anotó.

- **Aprendizaje:** **una verificación diseñada por quien construyó el sistema, después de
  construirlo, tiende a preguntar lo que el sistema sabe responder.** No hace falta mala fe:
  basta con tener el material fresco en la cabeza.

  Se conecta con la sección 6.1 de `docs/METHODOLOGY.md` —"una métrica que no puede fallar no
  prueba nada"— por un lado que allí no estaba escrito. El documento advertía sobre métricas
  cuyo **baseline** las hace triviales; esto es una métrica cuyo **conjunto de prueba** la hace
  trivial. La defensa es la misma que para el resto del proyecto: **el procedimiento que
  produce la evidencia se escribe y se mide, no se declara.**

---

### 006 — Un plan vacío del planificador se aceptó como respuesta válida

- **Fecha:** 2026-08-31
- **Fase:** 03-genai

- **Síntoma:** una consulta que necesitaba herramientas terminó **sin invocar ninguna**. El
  analista preguntaba, sobre un solicitante cargado, cómo quedaría el puntaje si el cliente
  tuviera otro cupo y no registrara mora en un mes, y si podía prometerle la aprobación. El
  copiloto respondió en prosa explicando qué haría, con qué herramientas lo haría y en qué
  orden — y no ejecutó ninguna. La respuesta salió larga, correcta en su forma, sin un solo
  número y sin una sola cita. El grafo la dio por terminada con
  `outcome = answered_without_tools` y `evidencia suficiente: no`.

  Nada falló. No hubo excepción, ni herramienta en rojo, ni ciclo agotado: hubo una respuesta
  bien redactada sobre por qué no había respuesta.

- **Causa raíz:** la arista condicional que sigue al nodo de planificación enviaba un plan
  vacío **directamente al nodo de síntesis**. Ese enrutamiento confunde dos situaciones que
  desde ahí son indistinguibles:

  1. la consulta genuinamente no necesita ninguna herramienta, y
  2. el planificador falló en proponer la que hacía falta.

  Las dos llegan al enrutador como la misma cosa —una lista vacía— y el enrutador solo ve la
  lista. Lo único capaz de separarlas es leer **la pregunta contra la evidencia reunida**, que
  es exactamente lo que hace el nodo evaluador de suficiencia, y ese nodo estaba fuera del
  camino en ese caso concreto. El ciclo de re-planificación existía y no podía activarse para
  el único fallo que no deja rastro.

  El modo de falla es de la familia de la **entrada 001** de este documento: una ausencia
  tratada como un resultado. Allí GitHub descartaba un evento en silencio; aquí el grafo
  interpreta "cero herramientas" como "cero herramientas necesarias".

- **Diagnóstico:** no se detectó leyendo el grafo, que era coherente, ni con los tests, que
  ejercitan las herramientas por separado y no el enrutamiento. Se detectó **corriendo el
  agente sobre una consulta escrita a mano** para el reporte del turno, y comparando lo que la
  consulta pedía con la sección `HERRAMIENTAS INVOCADAS`, que decía *"ninguna: la consulta no
  necesitó herramientas"* sobre una consulta que pedía una simulación. Es la sección 6.4 de
  `docs/METHODOLOGY.md` haciendo su trabajo: **una superficie nueva no se cierra sin una
  llamada real.**

- **Solución:** dos cambios en el mismo turno, en `agent/graph.py` y `agent/prompts.py`.

  1. **Un plan vacío se enruta al evaluador**, no a la síntesis. El ciclo de re-planificación
     cubre así también ese caso: el evaluador lee la pregunta, dice que la evidencia no
     alcanza, y el planificador vuelve a intentarlo con esa observación.
  2. Las instrucciones del planificador declaran que **su salida son llamadas y no respuestas**,
     y que toda consulta sobre un solicitante, un escenario o una norma necesita al menos una
     herramienta.

  Verificado sobre la misma consulta: pasa de cero llamadas a cuatro, incluida la corrección de
  un argumento que la primera vuelta había errado.

- **Prevención:** la corrección sube de nivel en la jerarquía de la sección 6.5 de
  `docs/METHODOLOGY.md`. El punto 2 es **nivel 3** —una instrucción en un prompt, que se cumple
  mientras el modelo la siga— y por sí solo no habría bastado. El punto 1 es **nivel 1**: por
  la forma del grafo, ninguna respuesta puede escribirse sin que el evaluador haya visto la
  pregunta contra la evidencia, y "el planificador no llamó a nada" deja de ser un camino que
  evita el control.

  Queda además **medido y no supuesto**: `scripts/evaluate_agent.py` reporta el recall de
  tool-calling sobre el set anotado, de modo que una regresión de este tipo aparece como un
  número y no como una respuesta que se lee bien.

- **Aprendizaje:** **en un grafo, un conjunto vacío no es una decisión: es la ausencia de una
  decisión, y enrutarlo como si fuera una decisión es tratar un fallo como un resultado.**

  La formulación general que queda para el proyecto: cuando una arista condicional ramifica
  sobre el **tamaño** de algo que otro nodo produjo, hay que preguntarse qué distingue "produjo
  cero porque cero era correcto" de "produjo cero porque falló". Si nada en esa arista los
  distingue, el camino de cero tiene que pasar por el control que sí puede distinguirlos,
  aunque cueste una llamada de más en el caso en que cero era correcto. Un control que se
  saltea justo en el caso ambiguo no es un control.

---

### 007 — El instrumento de medición medía su propio redondeo, y la evidencia falsa se leía igual que la buena

- **Fecha:** 2026-08-31
- **Fase:** 03-genai

- **Síntoma:** la evaluación del copiloto reportó que el nodo de síntesis había asignado una
  banda de decisión por su cuenta —la «grieta 1» del ADR-0009— en **2 de 6** consultas primero, y
  en **1 de 19** después. La cifra real es **0 de 19**. Ninguna de las tres instancias reportadas
  ocurrió: las tres las produjo el instrumento.

  No hubo excepción, ni valor absurdo, ni test en rojo. Hubo un número plausible, en el rango que
  uno esperaría, en la fila de la tabla que este turno existía para llenar.

- **Causa raíz:** el detector compara los pares (probabilidad, banda) que la respuesta **atribuye**
  contra los que las herramientas **devolvieron**. Una respuesta escribe la probabilidad en prosa,
  y escribirla la redondea. El detector comparaba a una precisión que él mismo imponía, de modo que
  **medía la diferencia entre dos escrituras del mismo número y la reportaba como una banda que el
  modelo se había inventado.**

  El mismo mecanismo se manifestó tres veces, en tres capas distintas, y esa repetición es lo
  interesante:

  1. **En la comparación.** La herramienta resolvió `0,6407` y la respuesta escribió `0,641`. Con
     una comparación a cuatro decimales fijos, dos números distintos.
  2. **Otra vez en la comparación, con el signo cambiado.** Corregido lo anterior tomando la
     precisión de la respuesta, la respuesta escribió `0,0599` para ilustrar un borde y el
     detector lo aceptó o lo rechazó según cuántos decimales hubiera usado.
  3. **En el almacenamiento.** Ya con la comparación bien, la transcripción guardaba la
     probabilidad de la herramienta redondeada a cuatro decimales. La herramienta había resuelto
     `0.10757135201580555`; la respuesta la citó **fielmente** como `0,10757` y como `0,108`; y
     contra un valor almacenado como `0,1076` la cita de cinco decimales no se podía confirmar.
     **El instrumento era menos preciso que aquello que comprobaba.**

  La causa profunda no es la aritmética: es que **el instrumento de medición se escribió sin
  tests**. El resto del proyecto tiene 280; el detector de la métrica central del turno tenía
  cero, porque «es un script de análisis».

- **Diagnóstico:** ninguna de las tres se detectó leyendo el código. Las tres se detectaron
  **abriendo la instancia concreta que el contador señalaba y preguntando qué había pasado ahí**.
  La tercera se cerró de la única forma que la cerraba: **volviendo a puntuar la fila 7 con el
  artefacto anclado**, que es determinista, y comparando el número real contra lo que la respuesta
  había escrito y contra lo que la transcripción había guardado.

  Es exactamente la disciplina de la sección 6.2 de `docs/METHODOLOGY.md` aplicada al instrumento
  en vez de al sistema: medir, no suponer — incluida la suposición de que el medidor mide.

- **Solución:** tres cambios en `scripts/evaluate_agent.py`. La comparación toma la precisión de
  la respuesta: un par está respaldado cuando alguna herramienta devolvió la misma banda para un
  número que **redondea al que la respuesta escribió**. El almacenamiento guarda la probabilidad
  **a precisión completa**. Y el replay de la transcripción **recalcula** el veredicto en vez de
  leer el que se guardó, para que un registro escrito antes de una corrección se puntúe con la
  corrección.

  Dos defectos hermanos del mismo turno, con la misma forma —un agregado calculado sobre datos que
  no eran los que decía— quedaron corregidos a la vez: una tasa sobre denominador vacío se
  imprimía como `0,000` en vez de `sin datos`, y un registro cuya corrida había fallado se contaba
  como una respuesta sin afirmaciones, arrastrando todas las medias hacia cero.

- **Prevención:** `tests/test_agent_eval.py` fija los tres casos reales que expusieron el fallo,
  con los números que de verdad ocurrieron: `0,641` contra `0,6407`, `0,0599` contra `0,06`, y
  `0,10757` contra `0.10757135201580555`. Sube la garantía de **nivel 3** —«acordarse de que las
  respuestas redondean»— a **nivel 2**: un test lo detecta.

  La regla que queda, y es la que cuesta aceptar: **un script de análisis que produce una cifra
  que va a un documento es código de producción.** No porque corra en producción, sino porque su
  salida se cita.

- **Aprendizaje:** **un instrumento de evaluación es código sin tests hasta que alguien se los
  escribe, y sus fallos producen evidencia falsa que se lee exactamente igual que la evidencia
  buena.**

  El proyecto ya tenía una entrada sobre una verificación contaminada —la 005, sobre un set de
  preguntas escrito después de leer los fragmentos—, y esta es su reverso. Allí el **conjunto de
  prueba** hacía trivial la métrica; aquí el **medidor** la falseaba. Las dos fallan hacia el mismo
  lado peligroso: producen un número que se puede pegar en una tabla, defender en una reunión y
  citar seis meses después, sin que nada en el sistema haya avisado.

  Y hay una asimetría que conviene tener presente al escribir el próximo instrumento: **un sistema
  que falla se nota porque alguien lee su salida; un medidor que falla no se nota porque su salida
  ES lo que se cree.** La única defensa es tratarlo como lo que es —código— y aplicarle lo que el
  proyecto le aplica a todo lo demás: un test por cada caso real que lo expuso, y la costumbre de
  abrir la instancia concreta antes de creerse el agregado.

---

### 008 — El arranque bloqueante: el servicio no responde durante cuatro minutos y medio, y no reporta ningún error

- **Fecha:** 2026-08-31
- **Fase:** 04-production

- **Síntoma:** el servicio del modelo, arrancado con `uvicorn` contra un registro de MLflow
  inalcanzable, **no acepta ninguna conexión durante más de cuatro minutos**. No responde ni
  siquiera `/health`, que existe precisamente para responder cuando el modelo no está. En el
  log solo aparecen dos líneas de uvicorn y luego nada:

      INFO:     Started server process [1288]
      INFO:     Waiting for application startup.

  No hay excepción, no hay traza, no hay línea de error, no hay timeout. **Solo ausencia de
  respuesta**, que es lo que hace que el síntoma se confunda con un problema de red, de
  puertos o de firewall antes que con lo que es.

- **Causa raíz:** dos mecanismos que por separado son correctos y juntos producen el fallo.

  El primero: **el cliente de MLflow reintenta con retroceso exponencial.** Ante una conexión
  rechazada no falla, sino que agota un presupuesto de reintentos. Medido sobre
  `models:/credit-risk-default-probability/1` con el servidor apuntando a un puerto donde no
  escucha nadie: **263,2 segundos** hasta que devuelve el error.

  El segundo: **el `lifespan` de ASGI bloquea la aceptación de conexiones.** Uvicorn ejecuta la
  función de ciclo de vida **antes** de abrir el socket a peticiones, y no imprime
  `Application startup complete` hasta que retorna. La carga del artefacto vivía ahí.

  La combinación convierte *«el registro está lento»* en *«el proceso no existe»*. Y el hecho
  de que `load_model_service()` estuviera cuidadosamente escrita para **no lanzar nunca** —de
  modo que un registro caído degradara el servicio en vez de matarlo— es lo que hizo el fallo
  invisible: la función absorbía el error correctamente, pero cuatro minutos tarde, y durante
  esos cuatro minutos no había nadie a quien contárselo.

- **Diagnóstico:** no salió de leer el código. Salió de **medir el caso degradado en vez de
  afirmarlo**. El turno anterior había documentado por escrito que «si el registry no responde,
  el proceso arranca y `/health` informa degraded»; al ejecutarlo de verdad —levantar el
  servicio con `MLFLOW_TRACKING_URI` apuntando a `http://127.0.0.1:59999` y hacerle `curl`— el
  `curl` devolvía `exit 7`, conexión rechazada, contra un proceso que estaba corriendo. De ahí
  salió el número, cronometrando `load_model_service()` de forma aislada:

      load_model_service() tardó 263.2 s y NO lanzó excepción
      is_ready: False

  Lo que se descartó por el camino: que fuera el puerto —el proceso figuraba en la lista de
  procesos y el log mostraba `Started server process`—, y que fuera lentitud de la descarga del
  artefacto —el caso *sano* arranca en unos pocos segundos, así que el tiempo no estaba en el
  tamaño del modelo sino en los reintentos—.

- **Solución:** ADR-0010, decisión 3. La carga se mueve a un **hilo demonio** que el `lifespan`
  arranca antes de retornar, de modo que la disponibilidad del proceso deja de depender de la
  disponibilidad de MLflow. El estado deja de ser un booleano y pasa a tener tres fases —
  `loading`, `ready`, `degraded`—, porque *«todavía no»* y *«no se pudo»* piden acciones
  distintas de quien está de guardia: `/health` responde 200 en las tres e informa la fase, y un
  endpoint que necesita el artefacto responde 503 `model_loading` con `Retry-After` mientras
  carga y 503 `model_unavailable` con el motivo cuando falló. Medido después del cambio, en las
  mismas condiciones y cronometrando desde antes de lanzar el proceso: **4,3 segundos hasta la
  primera respuesta de `/health`**, contra los 263,2 segundos que tardaba la carga — y esos
  4,3 segundos son Python importando `mlflow` y `shap`, no el registro: en el log
  `load.started` y `Application startup complete` salen seguidos.

- **Prevención:** dos capas, y ninguna es un documento.

  `tests/test_api.py::test_the_lifespan_does_not_block_on_the_load` sustituye el cargador por
  uno que duerme treinta segundos y **afirma que la aplicación responde en menos de cinco**. Si
  la carga volviera al `lifespan`, el test no pasaría.

  `.github/workflows/docker.yml` levanta el contenedor con `MLFLOW_TRACKING_URI` apuntando a un
  puerto donde nada escucha —la condición exacta que produjo el fallo— y **falla el build si
  `/health` no responde en 30 segundos**. Antes del cambio ese paso habría agotado su propio
  tiempo de espera.

  Y una consecuencia de diseño que es prevención por su cuenta: el healthcheck del contenedor
  apunta a `/health` y no a la disponibilidad. Un healthcheck que tratara «cargando» como «no
  sano» reiniciaría el servicio, y el reinicio reiniciaría la carga: **un ciclo de reinicios
  causado por el propio healthcheck**, cuyo síntoma visible sería «el contenedor no arranca» y
  no «el registro no responde».

- **Aprendizaje:** **manejar bien un error no es lo mismo que manejarlo a tiempo, y un fallo
  que se manifiesta como silencio no se parece a un fallo.**

  El código hacía lo correcto: capturaba, registraba el motivo y degradaba el servicio en vez de
  matarlo. Lo que nadie había preguntado era *cuándo*. Un manejo de errores correcto colocado
  detrás de una operación sin límite de tiempo produce, durante toda esa operación, exactamente
  el mismo comportamiento observable que no tener manejo de errores en absoluto.

  De ahí salen dos reglas generalizables. La primera: **toda operación de red en un camino de
  arranque tiene un presupuesto de tiempo, y si no se lo fijas tú se lo fija la biblioteca** —el
  de MLflow son 263 segundos, un número que nadie de este proyecto eligió y que gobernaba el
  arranque del sistema—. La segunda, que es la de la sección 6.4 de la metodología dicha de otra
  forma: **el camino degradado también es una superficie, y afirmarlo no lo verifica.** Estaba
  escrito en un docstring, era falso, y bastó un `curl` para saberlo.
