# Registro de evaluación

Una entrada por medición realizada, dejando explícito **qué se midió, con qué
protocolo, contra qué baseline y con qué resultado**: una métrica sin el baseline al
lado y sin el procedimiento que la produjo no es evidencia de nada, porque no se
puede saber si el número es bueno ni se puede reproducir.

## Métricas fijadas

Las fija el **ADR-0002** y no se eligen por entrada. Toda evaluación reporta todas.

| Rol | Métrica |
| --- | --- |
| **Métrica de decisión** | **PR-AUC** (área bajo la curva precisión-recall) |
| Contexto | ROC-AUC |
| Contexto | KS |
| Contexto | Gini |
| Contexto | Brier score (calibración) |
| Contexto | **precision@top-10%** (principal) |
| Contexto | **precision@top-5%** (secundaria) |

Las comparaciones entre modelos se deciden **solo** por PR-AUC. Las métricas de
contexto describen el resultado; no lo eligen.

## Formato de entrada

    ### NNN — Qué se evaluó

    - **Fecha:** AAAA-MM-DD
    - **Fase:** NN-nombre-de-la-fase
    - **Objeto evaluado:** modelo, pipeline o componente, con su versión o commit.
    - **Datos:** dataset, partición (train/valid/test), número de filas y periodo.
    - **Protocolo:** esquema de validación, semilla, y cómo se evitó la fuga de
      información entre particiones.
    - **Métricas:** tabla con las siete métricas fijadas arriba. Cada fila lleva el
      valor obtenido **y su baseline al lado, en la misma fila**, más la desviación
      estándar entre folds cuando aplique.
    - **Baseline — campo obligatorio:** contra qué se compara, por qué es el baseline
      correcto para esta medición, y su valor en las mismas métricas y sobre los mismos
      datos. **Una entrada sin este campo está incompleta y no cuenta como evidencia.**
      Cuando no exista baseline previo, se declara el baseline trivial (clase mayoritaria
      o azar estratificado) y se reporta su valor, nunca "no aplica".
    - **Resultado:** la diferencia en PR-AUC respecto al baseline y si supera el umbral
      de decisión fijado de antemano.
    - **Reproducción:** el comando exacto y el run de MLflow que regeneran el número.
    - **Interpretación:** qué significa el resultado para el problema de negocio y qué
      limitaciones tiene.

Las entradas se numeran de forma consecutiva. Un resultado nunca se edita para
mejorarlo: si una medición resulta inválida, se añade una entrada nueva que la
invalida y explica por qué.

---

<!-- Las entradas van debajo de esta línea, de la más antigua a la más reciente. -->
