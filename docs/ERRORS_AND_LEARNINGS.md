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
