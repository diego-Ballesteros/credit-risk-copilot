# Diccionario de datos — Default of Credit Card Clients

Contrato de datos del proyecto. Describe qué columnas existen, qué tipo tienen, qué dice
la fuente que contienen y qué contienen realmente.

**La distinción entre las dos últimas cosas es el contenido principal de este documento.**
La columna *"Valores según la documentación oficial"* transcribe lo que UCI declara; la
columna *"Rango observado"* es una medición sobre el archivo descargado. Donde no
coinciden, la diferencia está en la sección
[Discrepancias con la documentación oficial](#discrepancias-con-la-documentación-oficial).

| | |
| --- | --- |
| **Fuente** | UCI Machine Learning Repository, dataset ID 350 |
| **URL** | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| **Registros** | 30.000 |
| **Columnas del archivo crudo** | 25 (identificador + 23 predictoras + target) |
| **Columnas de la tabla de trabajo** | **24** — `ID` se elimina en la carga ([ADR-0004](adr/0004-codigos-no-documentados-de-pay-status.md) §6) |
| **Tamaño del CSV crudo** | 2,76 MiB |
| **Target** | `DEFAULT_PAYMENT_NEXT_MONTH` — 6.636 positivos, 22,12% |
| **Valores nulos** | 0 en las 25 columnas del archivo crudo (medido, no supuesto) |
| **Filas duplicadas exactas** | 0 sobre el archivo crudo, `ID` incluido |
| **Filas idénticas salvo por `ID`** | 35 — informativas, no bloqueantes |
| **Período cubierto** | abril a septiembre de 2005 |
| **Moneda** | dólar taiwanés (NT$) |
| **Fecha de medición** | 2026-08-25 |

> **Sobre el conteo de columnas.** El ADR-0001 y `docs/ROADMAP.md` hablan de *24
> columnas*: cuentan las 23 predictoras más el target y dejan afuera `ID`, que sin
> embargo es una columna del archivo entregado. Este documento describe el archivo, así
> que tiene **25 filas**. No es una corrección al ADR: los ADRs no se reescriben, y la
> cifra de 23 predictoras que el ADR usa para justificar la elección del dataset sigue
> siendo exacta.
>
> Desde el [ADR-0004](adr/0004-codigos-no-documentados-de-pay-status.md) §6 la aritmética
> cierra por los dos lados: el **archivo** tiene 25 columnas y la **tabla de trabajo** que
> devuelve `load_dataset` tiene **24**, porque `ID` se elimina en la carga. La tabla de
> abajo describe el archivo, así que sigue teniendo 25 filas y marca `ID` como eliminada.

---

## Cómo se leen estos datos

El archivo crudo vive en `data/raw/` y está gitignoreado: **nunca se commitea y nunca se
edita**. Se obtiene con:

```
uv run python scripts/download_dataset.py
```

La descarga es idempotente. Si el archivo ya existe no se vuelve a pedir a UCI, salvo que
se pase `--force`.

El CSV crudo conserva los **nombres originales de la fuente** (`PAY_0`,
`default payment next month`, …). El renombrado a nombres canónicos ocurre en código, una
sola vez, en el mapa declarativo `RAW_TO_CANONICAL` del módulo `schema`, y lo aplica
`loader.load_dataset`, que es **la única puerta de entrada a los datos en todo el
proyecto**.

> **Nota sobre lo que entrega `ucimlrepo`.** La librería no devuelve los nombres
> documentados: rotula las columnas como `X1`…`X23`/`Y` y publica la leyenda en una tabla
> aparte. El archivo crudo se escribe con los nombres documentados, que son los que usa la
> fuente en su propia página y los que usa cualquier ejemplo publicado del dataset. La
> leyenda está congelada en el mapa `UCI_CODE_TO_RAW` y se compara contra la que entrega
> la API en cada descarga: si UCI cambia un nombre, la descarga falla ruidosamente en vez
> de cambiar de significado en silencio.

Solo dos bloques cambian de nombre al pasar a canónico — siete columnas en total. Las
otras dieciocho se conservan textuales, porque toda la literatura publicada sobre este
dataset usa ese vocabulario:

| Original | Canónico | Por qué cambia |
| --- | --- | --- |
| `PAY_0`, `PAY_2`…`PAY_6` | `PAY_STATUS_1`…`PAY_STATUS_6` | La fuente numera el bloque salteando el 1, así que ningún bucle puede recorrerlo sin un caso especial. Además `PAY_*` y `PAY_AMT*` son variables distintas que comparten prefijo. El sufijo canónico es el índice de mes que ya usan `BILL_AMT1..6` y `PAY_AMT1..6`, de modo que los tres bloques quedan alineados: `PAY_STATUS_1`, `BILL_AMT1` y `PAY_AMT1` describen el mismo mes. |
| `default payment next month` | `DEFAULT_PAYMENT_NEXT_MONTH` | Los espacios impiden usarlo como identificador. La transformación es la mínima que lo arregla: espacio a guion bajo, mayúsculas como el resto. No se agrega ni se quita ninguna palabra. |

**Índice de mes:** 1 = septiembre 2005 · 2 = agosto · 3 = julio · 4 = junio · 5 = mayo ·
6 = abril. El índice 1 es siempre el mes **más reciente**.

---

## Las 25 columnas

| Nombre canónico | Nombre original | Tipo | Descripción | Valores según la documentación oficial | Rango observado |
| --- | --- | --- | --- | --- | --- |
| `ID` | `ID` | `int64` | Identificador de fila asignado por la fuente. **`load_dataset` la elimina**, así que no está en la tabla que el proyecto consume; `load_raw_dataframe` sí la devuelve. Ver [ADR-0004](adr/0004-codigos-no-documentados-de-pay-status.md) §6. | **No documentada.** La fuente describe 23 variables explicativas y el target; no menciona esta columna. | 1 … 30.000, únicos y contiguos |
| `LIMIT_BAL` | `LIMIT_BAL` | `int64` | Monto del crédito otorgado en NT$. Incluye el crédito individual y el familiar (suplementario). | Entero en NT$. Sin rango declarado. | 10.000 … 1.000.000 |
| `SEX` | `SEX` | `int64` | Género del titular. | 1 = hombre; 2 = mujer | {1, 2} |
| `EDUCATION` | `EDUCATION` | `int64` | Nivel educativo del titular. | 1 = posgrado; 2 = universidad; 3 = secundaria; 4 = otros | {0, 1, 2, 3, 4, 5, 6} ⚠️ |
| `MARRIAGE` | `MARRIAGE` | `int64` | Estado civil del titular. | 1 = casado; 2 = soltero; 3 = otros | {0, 1, 2, 3} ⚠️ |
| `AGE` | `AGE` | `int64` | Edad en años. | Años. Sin rango declarado. | 21 … 79 |
| `PAY_STATUS_1` | `PAY_0` | `int64` | Estado de pago en septiembre de 2005. | -1 = pagó puntualmente; 1 … 8 = atraso de ese número de meses; 9 = atraso de nueve meses o más | {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8} ⚠️ |
| `PAY_STATUS_2` | `PAY_2` | `int64` | Estado de pago en agosto de 2005. | Ídem `PAY_STATUS_1` | {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8} ⚠️ |
| `PAY_STATUS_3` | `PAY_3` | `int64` | Estado de pago en julio de 2005. | Ídem `PAY_STATUS_1` | {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8} ⚠️ |
| `PAY_STATUS_4` | `PAY_4` | `int64` | Estado de pago en junio de 2005. | Ídem `PAY_STATUS_1` | {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8} ⚠️ |
| `PAY_STATUS_5` | `PAY_5` | `int64` | Estado de pago en mayo de 2005. | Ídem `PAY_STATUS_1` | {-2, -1, 0, 2, 3, 4, 5, 6, 7, 8} ⚠️ |
| `PAY_STATUS_6` | `PAY_6` | `int64` | Estado de pago en abril de 2005. | Ídem `PAY_STATUS_1` | {-2, -1, 0, 2, 3, 4, 5, 6, 7, 8} ⚠️ |
| `BILL_AMT1` | `BILL_AMT1` | `int64` | Monto del estado de cuenta de septiembre de 2005, en NT$. | Entero en NT$. Sin rango declarado. | -165.580 … 964.511 |
| `BILL_AMT2` | `BILL_AMT2` | `int64` | Monto del estado de cuenta de agosto de 2005, en NT$. | Entero en NT$. Sin rango declarado. | -69.777 … 983.931 |
| `BILL_AMT3` | `BILL_AMT3` | `int64` | Monto del estado de cuenta de julio de 2005, en NT$. | Entero en NT$. Sin rango declarado. | -157.264 … 1.664.089 |
| `BILL_AMT4` | `BILL_AMT4` | `int64` | Monto del estado de cuenta de junio de 2005, en NT$. | Entero en NT$. Sin rango declarado. | -170.000 … 891.586 |
| `BILL_AMT5` | `BILL_AMT5` | `int64` | Monto del estado de cuenta de mayo de 2005, en NT$. | Entero en NT$. Sin rango declarado. | -81.334 … 927.171 |
| `BILL_AMT6` | `BILL_AMT6` | `int64` | Monto del estado de cuenta de abril de 2005, en NT$. | Entero en NT$. Sin rango declarado. | -339.603 … 961.664 |
| `PAY_AMT1` | `PAY_AMT1` | `int64` | Monto pagado en septiembre de 2005, en NT$. | Entero en NT$. Sin rango declarado. | 0 … 873.552 |
| `PAY_AMT2` | `PAY_AMT2` | `int64` | Monto pagado en agosto de 2005, en NT$. | Entero en NT$. Sin rango declarado. | 0 … 1.684.259 |
| `PAY_AMT3` | `PAY_AMT3` | `int64` | Monto pagado en julio de 2005, en NT$. | Entero en NT$. Sin rango declarado. | 0 … 896.040 |
| `PAY_AMT4` | `PAY_AMT4` | `int64` | Monto pagado en junio de 2005, en NT$. | Entero en NT$. Sin rango declarado. | 0 … 621.000 |
| `PAY_AMT5` | `PAY_AMT5` | `int64` | Monto pagado en mayo de 2005, en NT$. | Entero en NT$. Sin rango declarado. | 0 … 426.529 |
| `PAY_AMT6` | `PAY_AMT6` | `int64` | Monto pagado en abril de 2005, en NT$. | Entero en NT$. Sin rango declarado. | 0 … 528.666 |
| `DEFAULT_PAYMENT_NEXT_MONTH` | `default payment next month` | `int64` | **Target.** Indica si el cliente incumple el pago del mes siguiente (octubre de 2005). | 1 = incumple; 0 = no incumple | {0, 1} — 22,12% positivos |

⚠️ = la columna contiene valores que la documentación oficial no declara. Detalle abajo.

---

## Discrepancias con la documentación oficial

Todas las frecuencias de esta sección están **medidas** sobre las 30.000 filas del archivo
descargado, no estimadas.

**Esta sección describe lo que hay y registra lo que se decidió.** Las mediciones que
sostienen cada decisión están en
[`docs/analysis/undocumented-codes-evidence.md`](analysis/undocumented-codes-evidence.md)
y son reproducibles con `uv run python scripts/analyze_undocumented_codes.py`. El
razonamiento y las alternativas descartadas están en el
[**ADR-0004**](adr/0004-codigos-no-documentados-de-pay-status.md). Este documento no
justifica: apunta.

### 1 · Valores presentes en los datos que la fuente no documenta

Todos los códigos de esta tabla están **aceptados** por el ADR-0004. Ninguno se agregó a
la lista de niveles declarados por la fuente: viven en un mapa aparte, `OBSERVED_CODES_ACCEPTED`
del módulo `schema`, y el validador los reporta como **informativos**, nunca como
bloqueantes. La separación existe para que el día que UCI publique documentación se pueda
seguir distinguiendo *lo que la fuente declara* de *lo que este proyecto aceptó midiendo*.

| Columna | Código | Filas | % del total | Significado aceptado | Tratamiento | Decisión |
| --- | --- | ---: | ---: | --- | --- | --- |
| `EDUCATION` | `0` | 14 | 0,05% | Código de educación no documentado | Se colapsa al nivel `4` ("otros") | ADR-0004 §4 |
| `EDUCATION` | `5` | 280 | 0,93% | Ídem | Se colapsa al nivel `4` | ADR-0004 §4 |
| `EDUCATION` | `6` | 51 | 0,17% | Ídem | Se colapsa al nivel `4` | ADR-0004 §4 |
| | **subtotal** | **345** | **1,15%** | Default agrupado **7,54%** contra **5,69%** del nivel `4` | | |
| `MARRIAGE` | `0` | 54 | 0,18% | Código de estado civil no documentado | **Se conserva como nivel propio.** No se colapsa | ADR-0004 §5 |
| | **subtotal** | **54** | **0,18%** | Default **9,26%** contra **26,01%** del nivel `3` ("otros") | | |
| `PAY_STATUS_1` | `-2` | 2.759 | 9,20% | Sin consumo en el mes | Nivel propio en one-hot | ADR-0004 §1 |
| `PAY_STATUS_1` | `0` | 14.737 | 49,12% | Crédito revolvente | Nivel propio en one-hot | ADR-0004 §1 |
| | **subtotal** | **17.496** | **58,32%** | | | |
| `PAY_STATUS_2` | `-2` | 3.782 | 12,61% | Sin consumo en el mes | Nivel propio en one-hot | ADR-0004 §1 |
| `PAY_STATUS_2` | `0` | 15.730 | 52,43% | Crédito revolvente | Nivel propio en one-hot | ADR-0004 §1 |
| | **subtotal** | **19.512** | **65,04%** | | | |
| `PAY_STATUS_3` | `-2` | 4.085 | 13,62% | Sin consumo en el mes | Nivel propio en one-hot | ADR-0004 §1 |
| `PAY_STATUS_3` | `0` | 15.764 | 52,55% | Crédito revolvente | Nivel propio en one-hot | ADR-0004 §1 |
| | **subtotal** | **19.849** | **66,16%** | | | |
| `PAY_STATUS_4` | `-2` | 4.348 | 14,49% | Sin consumo en el mes | Nivel propio en one-hot | ADR-0004 §1 |
| `PAY_STATUS_4` | `0` | 16.455 | 54,85% | Crédito revolvente | Nivel propio en one-hot | ADR-0004 §1 |
| | **subtotal** | **20.803** | **69,34%** | | | |
| `PAY_STATUS_5` | `-2` | 4.546 | 15,15% | Sin consumo en el mes | Nivel propio en one-hot | ADR-0004 §1 |
| `PAY_STATUS_5` | `0` | 16.947 | 56,49% | Crédito revolvente | Nivel propio en one-hot | ADR-0004 §1 |
| | **subtotal** | **21.493** | **71,64%** | | | |
| `PAY_STATUS_6` | `-2` | 4.895 | 16,32% | Sin consumo en el mes | Nivel propio en one-hot | ADR-0004 §1 |
| `PAY_STATUS_6` | `0` | 16.286 | 54,29% | Crédito revolvente | Nivel propio en one-hot | ADR-0004 §1 |
| | **subtotal** | **21.181** | **70,60%** | | | |

**Alcance conjunto:** 25.970 de las 30.000 filas — el **86,57%** del dataset — tienen al
menos un código no documentado en alguna de estas columnas.

> **La lectura de `-2` y `0` es una inferencia, no un hecho de la fuente.** La evidencia
> que la sostiene es la mediana del ratio de cobertura de pago: **1,000** para `-1` y `-2`
> en los cinco meses calculables, contra **0,042 a 0,057** para el código `0`. Si UCI
> llegara a publicar documentación que la contradiga, el ADR-0004 se marca `superseded` y
> las features derivadas se revisan.

### 1.b · Consecuencias sobre el tratamiento de `PAY_STATUS_*`

Dos restricciones que el ADR-0004 fija y que el paso de features no puede ignorar:

- **Estas seis columnas son categóricas, no ordinales.** El orden numérico no es orden de
  severidad: en el mes 1 el código `0` tiene **12,81%** de default, *menor* que el `-1`
  (16,78%) y que el `-2` (13,23%). Van a **one-hot**, nunca a escalado numérico.
- **El mes 1 no comparte escala con los meses 2 a 6 en la zona baja.** Las features de
  trayectoria se construyen sobre `PAY_STATUS_2..6` y el mes 1 se trata como variable
  aparte. El contrato lo declara en `PAY_STATUS_HOMOGENEOUS_COLUMNS` y
  `PAY_STATUS_ISOLATED_COLUMN` para que un bucle sobre las seis columnas tenga que
  escribirse a propósito.

`SEX` y el target son las únicas columnas categóricas cuyos valores coinciden exactamente
con lo que la fuente declara.

### 2 · Valores que la fuente documenta y que no aparecen en los datos

La discrepancia también corre en la otra dirección. La documentación declara la escala de
atraso hasta `9` ("nueve meses o más") y ese código no existe en ninguna de las seis
columnas. En dos de ellas tampoco aparece el `1`.

| Columna | Códigos declarados y ausentes |
| --- | --- |
| `PAY_STATUS_1` | `9` |
| `PAY_STATUS_2` | `9` |
| `PAY_STATUS_3` | `9` |
| `PAY_STATUS_4` | `9` |
| `PAY_STATUS_5` | `1`, `9` |
| `PAY_STATUS_6` | `1`, `9` |

Medición adicional sobre el código `1` ("atraso de un mes"), que no es una discrepancia
formal pero sí un patrón que no se explica con la documentación: aparece 3.688 veces en
`PAY_STATUS_1`, 28 veces en `PAY_STATUS_2`, 4 en `PAY_STATUS_3`, 2 en `PAY_STATUS_4` y
ninguna en los dos meses restantes.

### 3 · Columnas sin ninguna documentación

`ID` no figura en la descripción de variables de la fuente. Su significado —
identificador de fila, único y contiguo de 1 a 30.000 — está **inferido de la medición**,
no transcrito de la fuente.

**`ID` ya no forma parte del DataFrame de trabajo.** `loader.load_dataset` la elimina; el
CSV crudo la conserva intacta y `loader.load_raw_dataframe` sigue devolviéndola, así que
la trazabilidad hacia la fuente no se pierde. La razón está en el
[ADR-0004](adr/0004-codigos-no-documentados-de-pay-status.md) §6 y es de jerarquía de
garantías: mientras la regla *"no debe usarse como feature"* vivía solo en este documento
era una garantía de nivel 3 según la sección 6.5 de `docs/METHODOLOGY.md`, y el nivel 3
falla. Una columna que no existe no se puede usar.

Consecuencia medida sobre el chequeo de duplicados: sin `ID` en la tabla, la detección de
**duplicados exactos** —el chequeo que distingue una extracción rota de dos clientes
iguales— **no se puede correr**, porque su condición es que un identificador se repita. El
validador lo reporta como omitido en vez de dejarlo pasar en silencio, y las 35 filas
idénticas siguen siendo un hallazgo **informativo**, exactamente como cuando `ID` estaba
presente. Para correr el chequeo bloqueante hay que validar sobre `load_raw_dataframe`.

---

## Rangos plausibles y su criterio

El validador compara las columnas numéricas contra rangos declarados en el módulo
`schema`. **Un límite no es el mínimo ni el máximo observado**: ajustar el intervalo a los
datos produce un chequeo que no puede fallar nunca. Un límite responde otra pregunta —
pasado este punto el valor ya no es un hecho de negocio sino un error de datos.

| Columna | Rango declarado | Criterio |
| --- | --- | --- |
| `ID` | ≥ 1 | Identificador 1-based. Sin techo: no tiene significado de negocio, así que cualquier techo sería arbitrario y una extracción mayor lo superaría legítimamente. |
| `LIMIT_BAL` | 1 … 5.000.000 | Cero o negativo no es un límite de crédito sino un registro ausente o corrupto. El techo es unas cinco veces el máximo observado: suficientemente ancho para no dispararse con un límite legítimo, suficientemente ajustado para detectar un cambio de unidad monetaria. |
| `AGE` | 18 … 120 | Un titular es mayor de edad, y 18 es el umbral de mayoría de edad más permisivo que existe, así que el piso no puede rechazar un registro legítimo; 120 excede cualquier longevidad registrada. **No se verificó** cuál era la edad legal mínima en Taiwán en 2005, así que no se afirma. Mínimo observado: 21. |
| `BILL_AMT1..6` | -5.000.000 … 5.000.000 | **El piso es negativo a propósito.** Un saldo negativo es legítimo y frecuente: un pago en exceso o una devolución dejan la cuenta a favor del cliente. Medido, no supuesto: 3.932 saldos negativos repartidos en 1.930 filas (6,43% de las filas), con mínimo -339.603. Un piso de cero los rechazaría a todos. |
| `PAY_AMT1..6` | 0 … 5.000.000 | **El piso es cero, y la asimetría con `BILL_AMT` es el punto.** El dinero que vuelve al cliente aparece como saldo a favor en el estado de cuenta, nunca como un pago negativo. Verificado contra los datos, no asumido: cero valores negativos en los seis meses. |

Con la configuración actual, ninguna columna numérica tiene valores fuera de rango.

---

## Criterio de severidad del validador

El validador acumula todos los problemas antes de fallar y los clasifica en dos niveles.
El criterio no es cuán alarmante se ve el hallazgo, sino **quién puede absorberlo**.

| Severidad | Criterio | Chequeos |
| --- | --- | --- |
| **Bloqueante** | Seguir adelante exige una decisión que ningún valor por defecto puede tomar bien. | Columna faltante · columna sobrante · tipo distinto al esperado · valores nulos · **categoría que ni la fuente declara ni un ADR acepta** · valor fuera de rango · fila duplicada exacta |
| **Informativo** | Un hecho medido que conviene conocer y que no invalida el contrato. | **Código no documentado aceptado por un ADR** · filas que difieren solo en `ID` · chequeo de rango omitido por tipo no numérico · **chequeo de duplicados exactos omitido por ausencia de `ID`** |

Tres elecciones que no son obvias:

- **Una columna sobrante es bloqueante.** No rompe ningún cálculo, pero una columna que el
  contrato no conoce tiene exactamente la forma de un vector de fuga de información, y la
  política del proyecto es hacer la fuga imposible, no acordarse de no provocarla.
- **Filas que difieren solo en `ID` son informativas.** Dos clientes distintos con
  atributos idénticos son esperables en una muestra de 30.000 filas comparada sobre las 24
  variables restantes, en su mayoría gruesas. Se miden 35 casos. El número igual se reporta, porque acota cuánta
  información realmente independiente hay.
- **Un código aceptado por un ADR es informativo, no silencioso.** El ADR-0004 le quitó el
  carácter bloqueante al hallazgo, no el hallazgo. La lectura de esos códigos es una
  inferencia sobre evidencia propia, no un hecho documentado por la fuente, y quien lea la
  salida de una corrida tiene derecho a verlo declarado cada vez en lugar de tener que
  saber que en algún momento se decidió algo.

Estado actual del dataset: **0 hallazgos bloqueantes** y **10 informativos** — ocho
códigos aceptados por el ADR-0004 (`EDUCATION`, `MARRIAGE` y las seis columnas
`PAY_STATUS_*`), las 35 filas idénticas salvo por `ID`, y el chequeo de duplicados exactos
omitido por ausencia de `ID`. `uv run python scripts/download_dataset.py` sale con
**código 0**.

---

*Toda afirmación negativa de este documento — "0 nulos", "0 duplicados exactos", "cero
pagos negativos" — es cierta a la fecha de medición y debe reverificarse cuando cambie la
fuente. Correr `uv run python scripts/download_dataset.py --force` las reverifica todas.*
