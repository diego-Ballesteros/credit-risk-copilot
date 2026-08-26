# ADR 0005 — Diseño de las features de comportamiento de pago

- **Status:** Accepted
- **Date:** 2026-08-25

---

## Contexto

Las features de comportamiento de pago son las que ponen a prueba la hipótesis principal
del proyecto. Cinco decisiones de diseño tuvieron alternativas reales y ninguna es
evidente desde el código.

---

## Decisión

### 1 · La tendencia de utilización es la pendiente de una regresión lineal

`UTILIZATION_TREND_M2_M6` es la **pendiente de una regresión lineal** sobre los cinco
meses del bloque homogéneo. Con meses equiespaciados tiene forma cerrada y su denominador
es constante, por lo que **no introduce ningún caso nuevo de valor no calculable**.

Alternativas descartadas:

- **La diferencia entre el primer y el último mes.** Usa dos observaciones y descarta
  tres, y las dos que conserva son los bordes, donde una compra atípica distorsiona la
  feature completa.
- **La media de diferencias consecutivas.** Se telescopia a la diferencia de extremos y
  por tanto hereda el mismo defecto.
- **El cociente entre el último y el primer mes.** Reintroduce el problema del denominador
  cero, frecuente porque la utilización es exactamente cero bajo el código `-2`.

### 2 · El denominador de todo ratio debe superar un piso positivo

El denominador de todo ratio debe ser **mayor que un piso positivo**, no simplemente
distinto de cero. Dos razones medidas.

**Primera: un saldo anterior negativo es legítimo.** Aparece **3.932 veces** en el
archivo. Dividir por él invierte el signo, de modo que un abono de 5.000 sobre un saldo de
-2.000 produciría **-2,5** en una escala donde el resto de los valores son fracciones de
cobertura, y el modelo lo leería como comportamiento excelente.

**Segunda: un denominador positivo pero trivial produce ratios sin significado.** El caso
extremo medido es un abono de **10.002** contra un saldo anterior de **2**, que da un
ratio de **5.001**.

Se fija el piso en **100 NT\$**, una fracción trivial del cupo mínimo del dataset, que es
**10.000**, de modo que el umbral no puede descartar comportamiento significativo.

**Regla exacta: `denominador > 100`, estricta.** Un denominador de exactamente 100 NT\$
**no** es utilizable. El piso es el mayor valor excluido, que es la misma forma que tenía
la regla anterior `> 0` —el cero también estaba excluido— con el valor excluido movido.

#### Efecto medido del piso frente a la regla anterior

Filas que pasan a **no calculable** al aplicar el piso de 100 frente a la regla anterior
de mayor que cero, sobre las 30.000 filas del archivo descargado:

| Feature | Denominador | No calculables con `> 0` | Con piso de 100 | **Filas adicionales** | **% adicional** |
| --- | --- | ---: | ---: | ---: | ---: |
| `PAYMENT_RATIO_M2` | `BILL_AMT3` | 3.525 (11,75%) | 3.595 (11,98%) | **70** | **0,23%** |
| `PAYMENT_RATIO_M3` | `BILL_AMT4` | 3.870 (12,90%) | 3.933 (13,11%) | **63** | **0,21%** |
| `PAYMENT_RATIO_M4` | `BILL_AMT5` | 4.161 (13,87%) | 4.246 (14,15%) | **85** | **0,28%** |
| `PAYMENT_RATIO_M5` | `BILL_AMT6` | 4.708 (15,69%) | 4.781 (15,94%) | **73** | **0,24%** |
| `UTILIZATION_*` | `LIMIT_BAL` | 0 (0,00%) | 0 (0,00%) | **0** | **0,00%** |

Agregado por fila: las filas con **al menos un** ratio de pago no calculable pasan de
6.862 (22,87%) a **6.975 (23,25%)**, y las que tienen **los cuatro** de 1.893 (6,31%) a
**1.922 (6,41%)**.

El piso no toca la utilización en este archivo porque `LIMIT_BAL` tiene mínimo 10.000, que
es cien veces el umbral. La columna indicadora sigue existiendo por la decisión 5.

### 3 · La mora máxima aplasta a cero los códigos `-2`, `-1` y `0`

`MAX_DELINQUENCY_M2_M6` **aplasta a cero los códigos `-2`, `-1` y `0` antes de tomar el
máximo**, en vez de ordenarlos entre sí.

El [ADR-0004](0004-codigos-no-documentados-de-pay-status.md) estableció que la zona baja de
la escala **no es ordinal**, mientras que la zona igual o mayor a `3` sí es homogénea entre
meses y sí representa un conteo de meses de atraso. **El máximo se toma solo donde existe
un orden.**

### 4 · El ratio de pago existe para los meses 2 a 5 y no para el 6

El ratio de pago existe para los meses **2 a 5** del bloque y **no para el 6**.

Su denominador es el saldo del mes **cronológicamente anterior**, que con el índice
canónico lleva número **mayor**; el del mes 6 sería el mes 7, que no existe en el panel.

### 5 · Se conserva el indicador de utilización no calculable

Se conserva `UTILIZATION_NOT_COMPUTABLE` **aunque su varianza sea cero en este dataset**.

Existe para que el pipeline **declare** el supuesto en vez de apoyarse en él; una
extracción futura con el cupo corrupto aparecería como una columna en vez de propagarse
como valor faltante sin explicación.

---

## Consecuencias

### Positiva

**Ninguna feature esconde una imputación silenciosa.** Todo valor no calculable es un
faltante explícito con su columna indicadora al lado.

### Negativa

El pipeline **debe decidir explícitamente qué hacer con los faltantes**, porque varios
modelos no los aceptan. Esa decisión corresponde al turno del pipeline y no a este.

### Riesgo

El piso de **100 NT\$** es un umbral **elegido con criterio y no derivado de los datos**, y
la medición muestra que **se queda corto respecto de su propio objetivo**, no que se pase.

El piso quita el caso extremo y **no quita la cola**. Medido sobre las 30.000 filas: el
máximo de `PAYMENT_RATIO_M4` **no se mueve de 129,71**, porque su denominador es de 780
NT\$ y supera el umbral; el de `PAYMENT_RATIO_M5` sigue en 447,74 sobre un denominador de
291 NT\$. Un denominador puede superar los 100 NT\$ y seguir siendo lo bastante chico como
para que el ratio mida el tamaño del resumen y no la disciplina del cliente. El p99 apenas
baja: de 1,3491 a 1,3235 en el mes 2.

**La acotación de la cola no se resuelve subiendo el piso: se resuelve en el pipeline.** El
piso es una regla de *calculabilidad* —por debajo de él el cociente no significa nada— y la
cola es un problema de *escala*, que se trata con un recorte por percentil aprendido en
`fit` sobre los datos de entrenamiento. Esa decisión pertenece al ADR del turno del
pipeline, no a este.

Si un análisis posterior mostrara que el piso descarta comportamiento relevante —la
evidencia disponible apunta en dirección contraria: agrega entre 0,21% y 0,28% de filas no
calculables por mes— se revisa mediante un ADR que **supersede** a este.
