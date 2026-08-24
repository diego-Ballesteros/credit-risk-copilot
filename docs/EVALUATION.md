# Registro de evaluacion

Una entrada por medicion realizada, dejando explicito **que se midio, con que
protocolo, contra que baseline y con que resultado**: una metrica sin el baseline al
lado y sin el procedimiento que la produjo no es evidencia de nada, porque no se
puede saber si el numero es bueno ni se puede reproducir.

## Formato de entrada

    ### NNN — Que se evaluo

    - **Fecha:** AAAA-MM-DD
    - **Fase:** NN-nombre-de-la-fase
    - **Objeto evaluado:** modelo, pipeline o componente, con su version o commit.
    - **Datos:** dataset, particion (train/valid/test), numero de filas y periodo.
    - **Protocolo:** esquema de validacion, semilla, y como se evito la fuga de
      informacion entre particiones.
    - **Metricas:** las metricas primarias y secundarias, con su valor y, cuando
      aplique, su intervalo de confianza.
    - **Baseline:** contra que se compara y cual es su valor en las mismas metricas
      y sobre los mismos datos.
    - **Resultado:** la diferencia respecto al baseline y si supera el umbral de
      decision fijado de antemano.
    - **Reproduccion:** el comando exacto y el run de MLflow que regeneran el numero.
    - **Interpretacion:** que significa el resultado para el problema de negocio y
      que limitaciones tiene.

Las entradas se numeran de forma consecutiva. Un resultado nunca se edita para
mejorarlo: si una medicion resulta invalida, se anade una entrada nueva que la
invalida y explica por que.

---

<!-- Las entradas van debajo de esta linea, de la mas antigua a la mas reciente. -->
