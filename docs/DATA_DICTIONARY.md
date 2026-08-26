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

## Features derivadas del comportamiento de pago

**Estas columnas no vienen en la fuente: las construye el proyecto.** No forman parte de
la tabla que devuelve `load_dataset` —que sigue teniendo 24 columnas— sino que las produce
el transformador `PaymentBehaviourFeatures` del módulo `features.builder`, que es un paso
de `Pipeline` de sklearn y no una función suelta. La razón de que sea un estimador y no una
función está en la sección 6.3 de `docs/METHODOLOGY.md`: el notebook, el script de
entrenamiento y la API tienen que consumir **el mismo objeto**, porque dos copias de la
misma aritmética divergen.

Son **21 columnas**. Ponen a prueba la hipótesis principal de `docs/ROADMAP.md` — que el
comportamiento de pago reciente predice mejor que la demografía estática — y su poder
predictivo **todavía no está medido**: este documento describe qué son y qué contienen, no
cuánto aportan.

**Fecha de medición de los rangos observados:** 2026-08-25, sobre las 30.000 filas del
archivo descargado.

### El corte que impone el ADR-0004

La [decisión 3 del ADR-0004](adr/0004-codigos-no-documentados-de-pay-status.md) gobierna la
forma de esta tabla: el mes 1 no comparte la escala de los meses 2 a 6 en la zona baja de
los códigos, así que **ninguna feature agrega las seis columnas como si fueran un panel
homogéneo**. Cada feature de abajo pertenece a exactamente uno de tres grupos:

| Grupo | Qué lee | Cuántas features |
| --- | --- | ---: |
| **Bloque homogéneo** | Meses 2 a 6 (agosto a abril) + `LIMIT_BAL` | 17 |
| **Mes 1 aislado** | Mes 1 (septiembre) + `LIMIT_BAL` | 2 |
| **Sin mes** | Solo `LIMIT_BAL` | 1 |

`LIMIT_BAL` aparece en los tres porque **no tiene mes**: es un atributo de la cuenta, no
del panel, así que usarlo a los dos lados del corte no cruza ninguna frontera.

> **`PAY_AMT1` no se usa en ninguna feature de este módulo, y esa ausencia es una
> consecuencia medible de la regla de aislamiento.** Un ratio de pago del mes 1 necesitaría
> el saldo del mes 2 como denominador, y eso sí cruzaría el corte. La feature no se
> construye; la decisión de si vale la pena hacer una excepción para los montos —donde el
> ADR-0004 no midió ningún problema de escala, porque el problema es de los códigos— no
> está tomada.

### Las 21 columnas

| Nombre | Grupo | Definición | Columnas de origen | Rango esperado | Rango observado | Qué mide en términos de negocio |
| --- | --- | --- | --- | --- | --- | --- |
| `UTILIZATION_M2` … `UTILIZATION_M6` | Bloque | `BILL_AMTm / LIMIT_BAL` para cada mes del bloque | `BILL_AMT2..6`, `LIMIT_BAL` | Real. Típicamente 0 … 1, **sin cota por arriba ni por abajo** | -1,5095 … 10,6886 | Qué fracción del cupo otorgado ocupa el saldo. Un cliente pegado al techo del cupo se quedó sin margen, que es la forma habitual del estrés financiero antes de un impago |
| `UTILIZATION_TREND_M2_M6` | Bloque | Pendiente de la recta de mínimos cuadrados de las cinco utilizaciones contra el tiempo, en puntos de utilización por mes. **Positivo = la utilización sube con el paso del tiempo** | `BILL_AMT2..6`, `LIMIT_BAL` | Real, centrado en 0 | -0,5407 … 1,0846 | Si el cliente se está llenando o vaciando. El nivel dice *qué tan hondo*; la tendencia dice *hacia dónde*, y un cliente al 40% subiendo es otro riesgo que uno al 40% bajando |
| `PAYMENT_RATIO_M2` … `PAYMENT_RATIO_M5` | Bloque | `PAY_AMTm / BILL_AMT(m+1)` — lo abonado sobre el saldo del mes **cronológicamente anterior**, que es el de índice **mayor** | `PAY_AMT2..5`, `BILL_AMT3..6` | ≥ 0, típicamente 0 … 1. `NaN` si el denominador no es positivo | 0 … 5.001,0 (p99 ≈ 1,35) | Qué fracción del resumen anterior pagó realmente. Es el separador de comportamiento más nítido que midió el ADR-0004: mediana 1,000 para los códigos que saldan contra 0,042–0,057 para el revolvente |
| `PAYMENT_RATIO_NOT_COMPUTABLE_M2` … `_M5` | Bloque | 1 si el saldo del mes anterior es cero o negativo | `BILL_AMT3..6` | {0, 1} | {0, 1} — 11,75% a 15,69% en 1 | Que el ratio de ese mes **no existe**, no que sea bajo. La ausencia suele ser informativa por sí sola |
| `DELINQUENCY_STREAK_M2_M6` | Bloque | Meses consecutivos con código ≥ 1 contando hacia atrás desde el mes **más reciente del bloque** (mes 2). Se corta en el primer mes sin mora | `PAY_STATUS_2..6` | Entero 0 … 5 | 0 … 5 — 85,21% en 0, 4,56% en 5 | Persistencia, no severidad: un mes tarde es un accidente, cuatro seguidos son una trayectoria. Un cliente que se atrasó y se recuperó saca 0 por más grave que haya sido el tramo anterior |
| `MAX_DELINQUENCY_M2_M6` | Bloque | Máximo de `max(código, 0)` sobre el bloque | `PAY_STATUS_2..6` | Entero 0 … 9 (la fuente declara hasta 9) | 0 … 8 | El peor atraso alcanzado en cualquier momento del bloque, en meses. **No supone que los códigos sean ordinales**: los códigos por debajo del umbral se aplastan a 0 primero, así que `-2`, `-1` y `0` valen todos "sin mora" en vez de ordenarse entre sí ([ADR-0004](adr/0004-codigos-no-documentados-de-pay-status.md) §2) |
| `BILL_VOLATILITY_M2_M6` | Bloque | Desviación estándar **poblacional** (`ddof=0`) de los cinco saldos | `BILL_AMT2..6` | ≥ 0, en NT$ | 0 … 621.397,56 (mediana 3.320,59) | Qué tan errático es el saldo. Un resumen que casi no se mueve es un hábito estable; uno que oscila es un usuario irregular o una cuenta bajo tensión. El divisor poblacional es deliberado: los cinco meses **son** la ventana descrita, no una muestra de algo más grande |
| `MONTHS_WITHOUT_PAYMENT_M2_M6` | Bloque | Cuántos de los cinco meses registran `PAY_AMTm == 0` | `PAY_AMT2..6` | Entero 0 … 5 | 0 … 5 — 54,52% en 0, 6,21% en 5 | Un evento concreto, distinto del código de mora, que describe un *estado*. Un pago de cero es el evento que precede al estado |
| `UTILIZATION_MOST_RECENT_M1` | Mes 1 | `BILL_AMT1 / LIMIT_BAL` | `BILL_AMT1`, `LIMIT_BAL` | Real. Típicamente 0 … 1, sin cota | -0,6199 … 6,4553 | La misma cantidad que las utilizaciones del bloque, deliberadamente **no promediada con ellas**: es la observación más cercana al mes que el target describe y la primera que miraría un analista de riesgo |
| `IS_DELINQUENT_MOST_RECENT_M1` | Mes 1 | 1 si `PAY_STATUS_1 ≥ 1` | `PAY_STATUS_1` | {0, 1} | {0, 1} — 22,73% en 1 | Si el cliente estaba en mora en el último mes observado. Binaria y no el código crudo, porque el código es categórico: entregar el entero le enseñaría al modelo una monotonía que el dato niega. El código no se pierde, va a one-hot en otra rama del pipeline |
| `UTILIZATION_NOT_COMPUTABLE` | Sin mes | 1 si `LIMIT_BAL ≤ 0` | `LIMIT_BAL` | {0, 1} | **Constante en 0** sobre las 30.000 filas | Que la utilización **de todas** las features que la usan —las cinco del bloque y la del mes 1— no se puede expresar, porque comparten un único denominador. Una cuenta sin cupo positivo no es una cuenta sin uso: es una cuenta cuyo uso no es una fracción de nada |

### Política de denominador no positivo

**Una sola política, escrita en un solo lugar y aplicada a las dos divisiones:** si el
denominador no es **estrictamente positivo**, el resultado es `NaN` y el hecho de que no se
pudo calcular sale como **columna indicadora propia**.

Dos elecciones que no son obvias:

- **El criterio es `> 0`, no `!= 0`.** Un saldo anterior de cero significa que no había
  nada que cubrir, así que "qué fracción se cubrió" no tiene respuesta. Un saldo anterior
  **negativo** significa que la cuenta estaba a favor del cliente —legítimo y frecuente:
  3.932 saldos negativos en el archivo— y dividir por él invierte el signo. Un cliente que
  abona 5.000 sobre un saldo de -2.000 daría -2,5 en una escala donde todo lo demás es una
  fracción de cobertura. Eso no es un ratio chico: es otra cantidad disfrazada con el mismo
  nombre, y un modelo la leería como comportamiento excelente.
- **El faltante es `NaN`, nunca `0`.** Cero es una medición —"no pagó nada"— y esto es la
  ausencia de una. Convertirlo en 0 es exactamente el modo de falla de la sección 7.1 de
  `docs/METHODOLOGY.md`: un "no sé" que pasa a ser un hecho de negocio falso. **Este módulo
  no imputa nada**; qué hacer con esos `NaN` es una decisión del paso siguiente del
  pipeline y debe quedar declarada cuando se tome.

Alcance medido de la política:

| Columna | Filas no calculables | % |
| --- | ---: | ---: |
| `PAYMENT_RATIO_NOT_COMPUTABLE_M2` | 3.525 | 11,75% |
| `PAYMENT_RATIO_NOT_COMPUTABLE_M3` | 3.870 | 12,90% |
| `PAYMENT_RATIO_NOT_COMPUTABLE_M4` | 4.161 | 13,87% |
| `PAYMENT_RATIO_NOT_COMPUTABLE_M5` | 4.708 | 15,69% |
| `UTILIZATION_NOT_COMPUTABLE` | 0 | 0,00% |

**6.862 filas (22,87%) tienen al menos un ratio de pago no calculable** y 1.893 (6,31%)
tienen los cuatro. De estas últimas, el 60,17% trae el código `-2` en los cinco meses del
bloque: son cuentas sin consumo, no cuentas rotas. El indicador está midiendo un
comportamiento real, que es la razón por la que se conserva como columna en vez de tirarse.

`UTILIZATION_NOT_COMPUTABLE` sale constante porque `LIMIT_BAL` tiene mínimo 10.000 en este
archivo y `schema.NUMERIC_RANGES` ya declara `LIMIT_BAL ≥ 1`. **Con varianza cero no aporta
información a ningún modelo sobre este dataset.** Se conserva igual: existe para que el
pipeline *declare* el supuesto en vez de apoyarse en él, y para que una extracción futura
con un cupo corrupto se vea en una columna en lugar de propagarse como `NaN` sin
explicación.

### Rangos que conviene mirar antes de modelar

Tres hechos medidos que el rango "típico" de la tabla no transmite:

- **La utilización no está acotada a [0, 1] y se sale por los dos lados.** Por arriba: 2.115
  filas (7,05%) en el mes 1 y entre 798 y 1.940 filas por mes en el bloque. El máximo es
  10,69 — un saldo de 855.086 NT$ contra un cupo de 80.000. Por abajo: alrededor del 2% de
  las filas por mes, porque un saldo negativo es legítimo. **Ninguno de los dos casos es un
  error de cálculo**; son el dato.
- **El ratio de pago tiene una cola larguísima.** El p99 está en ≈1,35 y el máximo en
  5.001,0, que corresponde a un abono de 10.002 NT$ contra un saldo anterior de **2 NT$**.
  La política de denominador excluye el cero y los negativos, pero un denominador de 2 NT$
  es positivo y produce un número enorme que es aritméticamente correcto y como señal de
  negocio no dice nada. Entre 116 y 138 filas por mes (≈0,4%) superan 2,0.
- **`MAX_DELINQUENCY_M2_M6` tiene un agujero en el valor 1:** solo 23 filas (0,08%), contra
  22.300 (74,33%) en 0 y 6.675 (22,25%) en 2. Es la decisión 3 del ADR-0004 reapareciendo
  desde otro ángulo — el código `1` aparece 28, 4, 2, 0 y 0 veces en los meses 2 a 6 — y
  significa que la escala de esta feature **salta de 0 a 2** en la práctica.

### Correspondencia con las 7 features del `docs/ROADMAP.md` §4.4

El roadmap enumera siete features con nombres en español. Los identificadores del código
van en inglés por la convención de idioma del proyecto, así que la correspondencia se
registra acá:

| `docs/ROADMAP.md` §4.4 | Implementado como | Nota |
| --- | --- | --- |
| `utilizacion_cupo_m` | `UTILIZATION_M2..M6` + `UTILIZATION_MOST_RECENT_M1` | Se parte en dos por el ADR-0004 §3 |
| `tendencia_utilizacion` | `UTILIZATION_TREND_M2_M6` | El roadmap dice "en 6 meses"; el ADR-0004 §3 la restringe a **5**, los meses 2 a 6 |
| `ratio_pago_m` | `PAYMENT_RATIO_M2..M5` | El roadmap la escribe `PAY_AMT_m / BILL_AMT_(m-1)`. Con el índice canónico —1 es el mes más reciente— el mes cronológicamente anterior es **`m+1`**, y así está implementada |
| `racha_mora` | `DELINQUENCY_STREAK_M2_M6` | |
| `mora_maxima` | `MAX_DELINQUENCY_M2_M6` | |
| `volatilidad_saldo` | `BILL_VOLATILITY_M2_M6` | |
| `meses_sin_pago` | `MONTHS_WITHOUT_PAYMENT_M2_M6` | |

Las cinco columnas restantes —los cuatro `PAYMENT_RATIO_NOT_COMPUTABLE_M*` y
`UTILIZATION_NOT_COMPUTABLE`— no están en el roadmap: son la política de denominador hecha
visible, y existen porque un "no calculable" enterrado dentro de la feature sería una
imputación silenciosa.

---

*Toda afirmación negativa de este documento — "0 nulos", "0 duplicados exactos", "cero
pagos negativos" — es cierta a la fecha de medición y debe reverificarse cuando cambie la
fuente. Correr `uv run python scripts/download_dataset.py --force` las reverifica todas.*
