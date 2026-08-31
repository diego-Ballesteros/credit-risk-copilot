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
