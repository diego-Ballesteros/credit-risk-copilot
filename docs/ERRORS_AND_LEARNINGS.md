# Errores y aprendizajes

Una entrada por error real que costo tiempo, documentando el **mecanismo** que lo
produjo y no solo el sintoma: un registro que dice "fallaba el import" no sirve a
nadie dentro de tres semanas; uno que dice por que fallaba, si.

## Formato de entrada

    ### NNN — Titulo corto del error

    - **Fecha:** AAAA-MM-DD
    - **Fase:** NN-nombre-de-la-fase
    - **Sintoma:** que se observo, con el mensaje de error literal si lo hubo.
    - **Causa raiz:** el mecanismo subyacente. Por que el sistema se comporto asi.
    - **Diagnostico:** como se llego a la causa. Que se descarto por el camino.
    - **Solucion:** el cambio concreto que lo resolvio, con el commit o PR.
    - **Prevencion:** que test, hook, tipo o regla impide que vuelva a ocurrir.
    - **Aprendizaje:** la regla generalizable que queda para el resto del proyecto.

Las entradas se numeran de forma consecutiva y no se reescriben ni se borran: una
entrada corregida se corrige con una entrada nueva que referencia a la anterior.

---

<!-- Las entradas van debajo de esta linea, de la mas antigua a la mas reciente. -->
