# ADR 0004 — Códigos no documentados de `PAY_STATUS`, `EDUCATION` y `MARRIAGE`

- **Status:** Accepted
- **Date:** 2026-08-25

---

## Contexto

El **86,57%** de las 30.000 filas contiene al menos un código que la documentación
oficial de UCI no declara. En las seis columnas `PAY_STATUS_*` el código `0` es el valor
**modal**: entre el **49,12%** y el **56,49%** de la tabla según el mes.

Esto no es un problema de datos sucios. Es más incómodo que eso: **la documentación de la
fuente describe correctamente una minoría de los datos**. Un contrato que solo admita lo
declarado rechaza el dataset entero, y el validador —que existe para dar señal— se
convierte en un rojo permanente que nadie vuelve a leer.

Las decisiones de abajo se apoyan en las mediciones de
**`docs/analysis/undocumented-codes-evidence.md`**, producidas por
`scripts/analyze_undocumented_codes.py` sobre el archivo descargado. Toda cifra citada
acá es reproducible corriendo ese script.

---

## Decisión

### 1 · Significado aceptado de los códigos `-2` y `0` de `PAY_STATUS`

Sobre la evidencia medida se acepta que:

- **`-2`** corresponde a **ausencia de consumo en el mes**.
- **`0`** corresponde a **crédito revolvente**: un cliente que usó la tarjeta, pagó parte
  del saldo y lo arrastra **sin estar en mora**.

La evidencia principal es el **ratio de cobertura de pago** — el pago de un mes dividido
por el saldo del mes anterior, sobre las filas con saldo anterior positivo:

| Código | Mediana del ratio de cobertura |
| --- | --- |
| `-1` | **1,000** en los cinco meses calculables |
| `-2` | **1,000** en los cinco meses calculables |
| `0` | **0,057 · 0,055 · 0,047 · 0,042 · 0,044** |

Los códigos `-1` y `-2` saldan el resumen del mes anterior; el `0` paga una fracción
positiva y pequeña de él. Es la separación más nítida que produjo la medición.

Tres mediciones adicionales sostienen la lectura del código `0`:

- **No hay saldo cero.** En el mes 1, ninguna de las 14.737 filas con código `0` tiene
  `BILL_AMT1 = 0` — el mínimo observado es 260 NT\$. En los meses 2 a 6 la proporción con
  saldo cero se mantiene entre **1,38%** y **3,19%**, contra **44,63%** a **61,08%** bajo
  el código `-2`.
- **El saldo es alto.** La mediana bajo el código `0` es **49.605 NT\$** contra **22.382**
  de la columna completa en el mes 1 — algo más del doble — y queda por encima de la
  mediana general en los seis meses.
- **El riesgo es bajo.** En el mes 1 la tasa de default del código `0` es **12,81%**, la
  más baja de la columna; en los meses 2 a 6 queda entre **15,91%** y **18,85%**, siempre
  por debajo del baseline de **22,12%**.

> **Sobre la literatura secundaria.** Esta lectura coincide con la interpretación
> difundida en trabajos publicados sobre este dataset. **Se adopta por la medición propia
> y no por la literatura.** La distinción no es de orgullo: la literatura secundaria no
> cita una fuente primaria para esa interpretación, así que apoyarse en ella sería
> apoyarse en una cadena de citas sin origen. La medición sí tiene origen y es
> reproducible.

### 2 · `PAY_STATUS` es categórica, no ordinal

**El orden numérico de estas columnas no es un orden de severidad.** Medido en el mes 1:

| Código | Tasa de default |
| --- | --- |
| `-2` | 13,23% |
| `-1` | 16,78% |
| `0` | **12,81%** |
| `1` | 33,95% |
| `2` | 69,14% |

El código `0` es **menos riesgoso** que el `-1` y que el `-2`, que lo preceden en el orden
numérico. Un modelo que trate la columna como numérica aprende una monotonía que **no
existe en el dato**.

**Consecuencia directa y no negociable:** estas seis columnas van a **one-hot encoding** y
**nunca** a escalado numérico.

### 3 · La escala del mes 1 difiere de la de los meses 2 a 6 en la zona baja

Tres evidencias independientes:

1. **Frecuencia.** El código `1` aparece **3.688** veces en `PAY_STATUS_1` y **28, 4, 2,
   0 y 0** veces en los meses 2 a 6. Si las seis columnas compartieran escala, esa
   distribución no sería posible.
2. **Correlación de Spearman entre columnas contiguas.** Las cuatro parejas que no
   involucran al mes 1 miden **0,799 · 0,801 · 0,822 · 0,821** — todas dentro de 0,023
   entre sí. La pareja del mes 1 con el mes 2 mide **0,627**, o sea 0,172 por debajo de la
   más baja de las otras cuatro.
3. **Matriz de transición.** En la matriz del mes 1 al mes 2, la fila del código `0` envía
   **0,00%** de sus **14.737** filas al código `2`. Las mismas filas envían entre **3,48%**
   y **5,32%** en las otras cuatro matrices. **Un cero sobre catorce mil observaciones es
   una imposibilidad estructural, no una muestra insuficiente.**

**Matiz medido que acota el alcance de la decisión.** La anomalía **se concentra en los
códigos `1`, `0` y `-1`**. Los códigos iguales o mayores a `3` transitan de `k` a `k-1`
con probabilidad dominante en las cinco matrices por igual — en la matriz del mes 1 al 2:
84,47% · 76,32% · 96,15% · 100% · 100% · 100% para los códigos 3 a 8. La zona alta de la
escala **sí** es homogénea entre meses.

**Consecuencia:** ninguna feature derivada puede tratar las seis columnas como un panel
homogéneo. **Las features de trayectoria se construyen sobre los meses 2 a 6**, y el
**mes 1 se trata como una variable aparte**.

### 4 · `EDUCATION`: los códigos `0`, `5` y `6` se colapsan al nivel `4`

El nivel `4` es el que la fuente declara como **"otros"**.

| Subconjunto | Filas | Tasa de default |
| --- | ---: | ---: |
| Códigos `0`, `5` y `6` agrupados | 345 | **7,54%** |
| Nivel documentado `4` ("otros") | 123 | **5,69%** |
| Niveles documentados `1`, `2` y `3` | 29.532 | **19,23% a 25,16%** |

Los tres códigos sin documentar se comportan como el nivel residual y no como los niveles
educativos ordinarios. Colapsarlos hacia el `4` los pone donde la medición los ubica.

### 5 · `MARRIAGE`: el código `0` se conserva como nivel propio y **NO** se colapsa

La evidencia apunta en **dirección contraria** a la de `EDUCATION`:

| Subconjunto | Filas | Tasa de default |
| --- | ---: | ---: |
| Código `0` | 54 | **9,26%** |
| Nivel documentado `3` ("otros") | 323 | **26,01%** |
| Nivel `1` | 13.659 | 23,47% |
| Nivel `2` | 15.964 | 20,93% |

El código `0` está por debajo de **los tres** niveles documentados, incluido el "otros" de
su propia columna, que es **2,8 veces más riesgoso**.

**Con 54 filas la estimación es imprecisa, y así se declara.** No se calculó intervalo de
confianza. Pero la imprecisión no vuelve indiferente la elección: colapsar hacia un nivel
2,8 veces más riesgoso introduce un sesgo de dirección conocida, mientras que conservar el
código separado deja la incertidumbre donde está.

> **`EDUCATION` y `MARRIAGE` se resuelven distinto porque la medición dio distinto, no por
> inconsistencia.** En `EDUCATION` el código sin documentar se alinea con el nivel
> residual; en `MARRIAGE` no se alinea con nada. Aplicar la misma regla a las dos columnas
> sería consistencia de forma a cambio de exactitud.

### 6 · La columna `ID` se elimina en la carga

Hoy `ID` sobrevive al DataFrame de trabajo y `docs/DATA_DICTIONARY.md` afirma que **nunca
debe usarse como feature**. Según la jerarquía de la sección 6.5 de
`docs/METHODOLOGY.md`, eso es una garantía de **nivel 3** — *"está escrito en un
documento"* — y **el nivel 3 falla**.

Eliminar la columna en `loader.load_dataset` sube la garantía a **nivel 1**: *"la
herramienta lo impide"*. No puede usarse como feature porque **no existe** en la tabla que
el proyecto consume.

El CSV crudo la conserva intacta; la eliminación ocurre en la carga. `load_raw_dataframe`
sigue devolviendo el archivo completo, así que la trazabilidad hacia la fuente no se
pierde.

---

## Alternativas consideradas

**Tratar los códigos no documentados como valores faltantes e imputarlos.**
Descartado. La evidencia muestra que tienen **perfil propio y consistente** —el ratio de
cobertura de 0,057 del código `0` frente al 1,000 del `-2` es una diferencia de
comportamiento, no una ausencia de dato—. Imputarlos destruiría información real y
convertiría un hecho medido en un valor inventado, que es exactamente el modo de falla
descrito en la sección 7.1 de `docs/METHODOLOGY.md`.

**Tratar `PAY_STATUS` como ordinal por simplicidad.**
Descartado por la decisión 2. La monotonía que asumiría no existe: el código `0` tiene
menor tasa de default que el `-1` y que el `-2`.

**Descartar las filas afectadas.**
Descartado. Son el **86,57%** del dataset.

---

## Consecuencias

### Positiva

El contrato de datos pasa a describir **el dataset real**. El validador vuelve a ser una
señal útil —un hallazgo bloqueante significa otra vez que algo está mal— en vez de un rojo
permanente que se aprende a ignorar.

### Negativa

One-hot sobre seis columnas con once niveles cada una **expande considerablemente el
espacio de features**. El costo se acepta a cambio de no codificar un orden falso: un
espacio más grande es un problema de eficiencia, y una monotonía inventada es un problema
de corrección.

### Riesgo aceptado

**La interpretación de `-2` y `0` es una inferencia sobre evidencia, no un hecho
documentado por la fuente.** Si apareciera documentación oficial que la contradiga, este
ADR se marca `superseded` con la razón y **las features derivadas se revisan**. Por eso
los códigos aceptados viven en un mapa **separado** del que declara lo que la fuente dice:
la diferencia entre *lo declarado* y *lo aceptado por nosotros* tiene que seguir siendo
legible el día que llegue esa documentación.
