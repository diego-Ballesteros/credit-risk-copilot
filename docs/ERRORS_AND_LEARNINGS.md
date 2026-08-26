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
