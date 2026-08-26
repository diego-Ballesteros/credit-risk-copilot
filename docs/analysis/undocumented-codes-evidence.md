# Evidencia sobre los códigos no documentados

> **Estado al 2026-08-25, posterior a esta medición.** Las decisiones que estas
> mediciones habilitaron están tomadas y registradas en el
> [**ADR-0004**](../adr/0004-codigos-no-documentados-de-pay-status.md). Los ocho hallazgos
> bloqueantes que el texto de abajo describe **ya no son bloqueantes**: el validador los
> reporta como informativos citando ese ADR, y `scripts/download_dataset.py` sale con
> código 0.
>
> **El cuerpo de este documento no se reescribe.** Es el registro de una medición con
> fecha, y era exacto el día que se hizo; corregirlo hacia atrás borraría el estado sobre
> el que se decidió. Las cifras siguen siendo reproducibles corriendo el script.

El validador de datos reporta ocho hallazgos bloqueantes de tipo `unknown_category`: el
bloque `PAY_STATUS_1..6`, `EDUCATION` y `MARRIAGE` contienen códigos que la documentación
oficial de UCI nunca declara. Este documento **mide** esos códigos. No decide qué
significan, no propone qué hacer con ellos y no modifica nada.

> **Qué no hay acá.** Ninguna afirmación sobre el significado de un código. Lo que sí hay:
> mediciones, y una lectura que dice si cada medición es *compatible* o *incompatible* con
> una hipótesis, siempre con la cifra concreta que sostiene esa lectura. La diferencia no
> es de estilo: decir "el −2 significa sin consumo" cierra el hallazgo con una conjetura;
> decir "el 61,08% de esas filas tiene saldo cero en el mes 6 y el 25,15% en el mes 1"
> deja la decisión donde corresponde.

## Cómo se reproduce

```
uv run python scripts/analyze_undocumented_codes.py
```

Todas las tablas de este documento son la salida literal de ese script. Ninguna cifra
está escrita a mano. El script solo lee: no toca los datos, ni `schema.py`, ni el
validador.

| | |
| --- | --- |
| **Filas** | 30.000 |
| **Columnas** | 24 — eran 25 al hacer esta medición; `ID` se elimina en la carga desde el [ADR-0004](../adr/0004-codigos-no-documentados-de-pay-status.md) §6. Ninguna medición de este documento usaba esa columna, y las 284 filas de tabla siguen siendo idénticas |
| **Target** | `DEFAULT_PAYMENT_NEXT_MONTH` — 6.636 positivos, **22,12%** |
| **Fecha de medición** | 2026-08-25 |

El **22,12%** es el baseline contra el que se lee toda tasa de default de este documento.
Un subconjunto con 18% no es "bajo" en abstracto: es bajo *contra ese número*.

## Índice de mes y dirección del panel

El índice 1 es el mes **más reciente** (septiembre de 2005) y el 6 el más viejo (abril de
2005). Por lo tanto el mes `m+1` es **anterior en el tiempo** al mes `m`. Esto tiene dos
consecuencias que afectan a los números y no a la redacción:

- El **ratio de cobertura de pago** se define como `PAY_AMT{m} / BILL_AMT{m+1}`, es decir,
  el pago de un mes dividido por el saldo del mes anterior. Un ciclo de facturación se
  cierra y recién después se paga: la plata registrada en `PAY_AMT{m}` salda el resumen
  que el cliente recibió al cierre del ciclo anterior. Dividir por `BILL_AMT{m}` metería
  el consumo nuevo del mes `m` en el denominador y haría que un cliente que pagó todo
  parezca pagador parcial cada vez que volvió a usar la tarjeta. El mes 6 no tiene mes
  anterior dentro del dataset, así que su ratio **no existe** y se reporta `n/a`.
- Las **matrices de transición** condicionan en el mes `m` y distribuyen sobre el `m+1`,
  o sea que se leen **hacia atrás en el tiempo**. Las cinco leen en la misma dirección,
  que es lo que la comparación entre ellas requiere.

## Cómo se leen las tablas por código

Cada mes tiene dos tablas. Las filas son los códigos **presentes en el dato**,
documentados o no, más una fila `all` con el mismo estadístico sobre la columna entera.

| Columna | Qué es |
| --- | --- |
| `rows`, `share` | Filas del subconjunto y su porcentaje sobre las 30.000 |
| `default` | Porcentaje del subconjunto con target 1. Baseline: 22,12% |
| `min`, `p25`, `p50`, `p75`, `max` | Estadísticos del monto, en NT$ |
| `= 0`, `<= 0` | Porcentaje **del subconjunto**, no de la tabla |
| `coverage p50` | Mediana de `PAY_AMT{m} / BILL_AMT{m+1}` sobre las filas con denominador positivo |
| `excluded` | Porcentaje del subconjunto excluido de esa mediana por denominador no positivo |

`excluded` no es un detalle de implementación: una mediana calculada sobre el 40% de un
subconjunto es una afirmación distinta de una calculada sobre el 98%, y sin ese número no
se puede saber cuál de las dos se está leyendo.

---

# Las hipótesis

| | Enunciado | De dónde sale |
| --- | --- | --- |
| **H1** | El código `-2` significa "sin consumo en el mes". Si es cierto, esas filas deben tener `BILL_AMT` cercano a cero. | Lectura financiera del producto |
| **H2** | El código `0` significa crédito revolvente: se usó la tarjeta, se pagó algo, y se arrastra saldo sin estar en mora. Si es cierto, esas filas deben tener `BILL_AMT` alto y `PAY_AMT` positivo pero menor que el saldo. | Lectura financiera del producto |
| **H3** | La escala de `PAY_STATUS_1` difiere de la de `PAY_STATUS_2..6`. | El código `1` aparece 3.688 veces en el mes 1 y 28, 4, 2, 0 y 0 veces en los meses 2 a 6 |

Ninguna sale de la documentación de UCI. Esa es la razón de medirlas.

---

# Mediciones — H1 y H2

Las dos hipótesis se miden sobre las mismas tablas, porque ambas son afirmaciones sobre
qué montos acompañan a un código.

## Mes 1 (septiembre de 2005)

| PAY_STATUS_1 |   rows |   share | default | BILL_AMT1 min |    p25 |     p50 |     p75 |     max |    = 0 |   <= 0 |
| -----------: | -----: | ------: | ------: | ------------: | -----: | ------: | ------: | ------: | -----: | -----: |
|           -2 |  2,759 |   9.20% |  13.23% |       -15,308 |      0 |   1,179 |   5,819 | 478,030 | 25.15% | 32.73% |
|           -1 |  5,686 |  18.95% |  16.78% |        -1,855 |    830 |   2,740 |   8,725 | 386,405 |  0.04% |  0.11% |
|            0 | 14,737 |  49.12% |  12.81% |           260 | 22,233 |  49,605 | 104,691 | 964,511 |  0.00% |  0.00% |
|            1 |  3,688 |  12.29% |  33.95% |      -165,580 |      0 |   4,606 |  29,206 | 523,618 | 35.57% | 45.80% |
|            2 |  2,667 |   8.89% |  69.14% |            37 | 17,182 |  41,632 |  85,810 | 613,860 |  0.00% |  0.00% |
|            3 |    322 |   1.07% |  75.78% |            99 |  2,500 |  18,354 |  49,716 | 415,735 |  0.00% |  0.00% |
|            4 |     76 |   0.25% |  68.42% |         2,646 | 19,133 |  31,633 |  57,531 | 581,775 |  0.00% |  0.00% |
|            5 |     26 |   0.09% |  50.00% |         1,800 | 20,464 |  39,752 |  72,843 | 589,654 |  0.00% |  0.00% |
|            6 |     11 |   0.04% |  54.55% |        32,875 | 36,810 |  71,310 | 111,194 | 254,951 |  0.00% |  0.00% |
|            7 |      9 |   0.03% |  77.78% |        22,858 | 33,816 | 126,220 | 243,234 | 405,366 |  0.00% |  0.00% |
|            8 |     19 |   0.06% |  57.89% |        16,942 | 24,329 |  43,340 | 144,178 | 477,094 |  0.00% |  0.00% |
|          all | 30,000 | 100.00% |  22.12% |      -165,580 |  3,559 |  22,382 |  67,091 | 964,511 |  6.69% |  8.66% |

| PAY_STATUS_1 | PAY_AMT1 min |   p25 |   p50 |   p75 |     max |     = 0 | coverage p50 | excluded |
| -----------: | -----------: | ----: | ----: | ----: | ------: | ------: | -----------: | -------: |
|           -2 |            0 |     0 | 1,131 | 5,000 | 368,199 |  33.06% |        1.000 |   34.51% |
|           -1 |            0 |   331 | 1,732 | 6,035 | 873,552 |  17.97% |        1.000 |   12.36% |
|            0 |            0 | 1,850 | 3,000 | 6,000 | 323,014 |   1.77% |        0.057 |    1.91% |
|            1 |            0 |     0 |     0 | 1,398 | 505,000 |  59.38% |        0.037 |   31.86% |
|            2 |            0 | 1,000 | 2,000 | 4,000 | 150,000 |  20.43% |        0.049 |    2.36% |
|            3 |            0 |     0 |     0 | 1,500 |  17,944 |  61.80% |        0.000 |    0.00% |
|            4 |            0 |     0 |     0 |     0 |  10,000 |  76.32% |        0.000 |    0.00% |
|            5 |            0 |     0 |     0 |     0 |   8,355 |  88.46% |        0.000 |    0.00% |
|            6 |            0 |     0 |     0 |     0 |       0 | 100.00% |        0.000 |    0.00% |
|            7 |            0 |     0 |     0 |     0 |       0 | 100.00% |        0.000 |    0.00% |
|            8 |            0 |     0 |     0 |     0 |       0 | 100.00% |        0.000 |    0.00% |
|          all |            0 | 1,000 | 2,100 | 5,006 | 873,552 |  17.50% |        0.078 |   10.58% |

## Mes 2 (agosto de 2005)

| PAY_STATUS_2 |   rows |   share | default | BILL_AMT2 min |    p25 |    p50 |     p75 |     max |    = 0 |   <= 0 |
| -----------: | -----: | ------: | ------: | ------------: | -----: | -----: | ------: | ------: | -----: | -----: |
|           -2 |  3,782 |  12.61% |  18.27% |       -69,777 |      0 |      0 |   2,564 | 419,644 | 44.63% | 55.68% |
|           -1 |  6,050 |  20.17% |  15.97% |       -26,214 |    416 |  2,035 |   7,280 | 481,382 |  8.66% | 11.59% |
|            0 | 15,730 |  52.43% |  15.91% |       -17,710 | 20,450 | 48,428 |  99,620 | 983,931 |  1.50% |  1.81% |
|            1 |     28 |   0.09% |  17.86% |       -67,526 | -1,992 |   -338 |   2,499 | 385,726 | 21.43% | 75.00% |
|            2 |  3,927 |  13.09% |  55.61% |        -4,577 |  9,439 | 27,689 |  65,970 | 552,234 |  1.32% |  1.60% |
|            3 |    326 |   1.09% |  61.66% |            78 | 10,966 | 24,481 |  51,037 | 572,677 |  0.00% |  0.00% |
|            4 |     99 |   0.33% |  50.51% |         1,113 | 16,897 | 27,371 |  59,898 | 581,775 |  0.00% |  0.00% |
|            5 |     25 |   0.08% |  60.00% |         4,003 | 21,087 | 37,936 | 102,832 | 321,476 |  0.00% |  0.00% |
|            6 |     12 |   0.04% |  75.00% |         8,001 | 30,218 | 77,765 | 224,523 | 397,754 |  0.00% |  0.00% |
|            7 |     20 |   0.07% |  60.00% |        16,721 | 24,016 | 39,860 | 133,526 | 469,882 |  0.00% |  0.00% |
|            8 |      1 |   0.00% |   0.00% |        25,589 | 25,589 | 25,589 |  25,589 |  25,589 |  0.00% |  0.00% |
|          all | 30,000 | 100.00% |  22.12% |       -69,777 |  2,985 | 21,200 |  64,006 | 983,931 |  8.35% | 10.58% |

| PAY_STATUS_2 | PAY_AMT2 min |   p25 |   p50 |    p75 |       max |     = 0 | coverage p50 | excluded |
| -----------: | -----------: | ----: | ----: | -----: | --------: | ------: | -----------: | -------: |
|           -2 |            0 |     0 |   190 |  3,132 | 1,684,259 |  48.18% |        1.000 |   49.52% |
|           -1 |            0 |   316 | 1,562 |  6,458 | 1,227,082 |  19.75% |        1.000 |   15.21% |
|            0 |            0 | 1,545 | 2,772 |  5,105 | 1,024,516 |   6.36% |        0.055 |    3.45% |
|            1 |            0 |     0 |   537 | 12,503 |   361,560 |  42.86% |        1.026 |   42.86% |
|            2 |            0 |     0 | 1,646 |  3,302 |   232,702 |  27.22% |        0.048 |    4.53% |
|            3 |            0 |     0 |     0 |  1,452 |    18,112 |  51.84% |        0.000 |    0.00% |
|            4 |            0 |     0 |     0 |      0 |     8,000 |  81.82% |        0.000 |    0.00% |
|            5 |            0 |     0 |     0 |      0 |    10,000 |  76.00% |        0.000 |    0.00% |
|            6 |            0 |     0 |     0 |    127 |     1,000 |  66.67% |        0.000 |    0.00% |
|            7 |            0 |     0 |     0 |      0 |         0 | 100.00% |        0.000 |    0.00% |
|            8 |            0 |     0 |     0 |      0 |         0 | 100.00% |        0.000 |    0.00% |
|          all |            0 |   833 | 2,009 |  5,000 | 1,684,259 |  17.99% |        0.078 |   11.75% |

## Mes 3 (julio de 2005)

| PAY_STATUS_3 |   rows |   share | default | BILL_AMT3 min |    p25 |    p50 |     p75 |       max |    = 0 |   <= 0 |
| -----------: | -----: | ------: | ------: | ------------: | -----: | -----: | ------: | --------: | -----: | -----: |
|           -2 |  4,085 |  13.62% |  18.53% |       -46,127 |      0 |      0 |   1,772 |   855,086 | 50.99% | 61.49% |
|           -1 |  5,938 |  19.79% |  15.59% |       -61,506 |    499 |  2,252 |   8,350 | 1,664,089 |  7.17% |  9.51% |
|            0 | 15,764 |  52.55% |  17.45% |      -157,264 | 19,795 | 46,894 |  94,546 |   693,131 |  1.71% |  2.05% |
|            1 |      4 |   0.01% |  25.00% |        15,882 | 51,228 | 68,485 | 135,778 |   321,232 |  0.00% |  0.00% |
|            2 |  3,819 |  12.73% |  51.56% |       -25,443 |  9,548 | 26,305 |  59,152 |   565,550 |  2.38% |  3.27% |
|            3 |    240 |   0.80% |  57.50% |           204 | 10,177 | 20,441 |  51,976 |   471,175 |  0.00% |  0.00% |
|            4 |     76 |   0.25% |  57.89% |            37 | 16,772 | 31,448 |  70,421 |   572,677 |  0.00% |  0.00% |
|            5 |     21 |   0.07% |  57.14% |         1,200 | 10,630 | 32,308 |  87,486 |   389,903 |  0.00% |  0.00% |
|            6 |     23 |   0.08% |  60.87% |           142 | 23,184 | 33,756 | 114,878 |   461,402 |  0.00% |  0.00% |
|            7 |     27 |   0.09% |  81.48% |           150 |  2,400 |  2,400 |   2,400 |    89,011 |  0.00% |  0.00% |
|            8 |      3 |   0.01% |  66.67% |         2,400 | 12,168 | 21,936 |  29,481 |    37,026 |  0.00% |  0.00% |
|          all | 30,000 | 100.00% |  22.12% |      -157,264 |  2,666 | 20,088 |  60,165 | 1,664,089 |  9.57% | 11.75% |

| PAY_STATUS_3 | PAY_AMT3 min |   p25 |   p50 |   p75 |     max |    = 0 | coverage p50 | excluded |
| -----------: | -----------: | ----: | ----: | ----: | ------: | -----: | -----------: | -------: |
|           -2 |            0 |     0 |     0 | 2,311 | 344,261 | 54.22% |        1.000 |   55.64% |
|           -1 |            0 |   192 | 1,390 | 6,000 | 889,043 | 21.57% |        1.000 |   14.95% |
|            0 |            0 | 1,200 | 2,200 | 5,000 | 896,040 |  6.90% |        0.047 |    3.41% |
|            1 |            3 |   578 | 4,435 | 8,584 |  10,036 |  0.00% |        0.091 |    0.00% |
|            2 |            0 |     0 | 1,231 | 3,000 | 349,395 | 28.88% |        0.041 |    4.48% |
|            3 |            0 |     0 |     0 |   943 |  25,065 | 65.83% |        0.000 |    0.00% |
|            4 |            0 |     0 |     0 |     0 |  15,405 | 76.32% |        0.000 |    0.00% |
|            5 |            0 |     0 |     0 |     0 |   5,000 | 85.71% |        0.000 |    0.00% |
|            6 |            0 |     0 |     0 |     0 |   1,170 | 82.61% |        0.000 |    0.00% |
|            7 |            0 |     0 |     0 |     0 |  10,000 | 96.30% |        0.000 |    0.00% |
|            8 |            0 |     0 |     0 | 1,250 |   2,500 | 66.67% |        0.000 |    0.00% |
|          all |            0 |   390 | 1,800 | 4,505 | 896,040 | 19.89% |        0.062 |   12.90% |

## Mes 4 (junio de 2005)

| PAY_STATUS_4 |   rows |   share | default | BILL_AMT4 min |     p25 |     p50 |     p75 |     max |    = 0 |   <= 0 |
| -----------: | -----: | ------: | ------: | ------------: | ------: | ------: | ------: | ------: | -----: | -----: |
|           -2 |  4,348 |  14.49% |  19.25% |       -65,167 |       0 |       0 |     999 | 339,176 | 56.42% | 66.74% |
|           -1 |  5,687 |  18.96% |  15.90% |       -14,795 |     538 |   2,156 |   8,121 | 891,586 |  7.81% | 10.18% |
|            0 | 16,455 |  54.85% |  18.33% |      -170,000 |  18,071 |  38,465 |  85,052 | 706,864 |  1.38% |  1.78% |
|            1 |      2 |   0.01% |  50.00% |        64,178 | 124,972 | 185,766 | 246,560 | 307,354 |  0.00% |  0.00% |
|            2 |  3,159 |  10.53% |  52.33% |        -7,511 |  12,757 |  28,380 |  60,275 | 572,805 |  2.25% |  3.04% |
|            3 |    180 |   0.60% |  61.11% |           116 |  12,876 |  27,332 |  53,817 | 486,776 |  0.00% |  0.00% |
|            4 |     69 |   0.23% |  66.67% |            37 |   1,650 |  20,779 |  63,359 | 384,981 |  0.00% |  0.00% |
|            5 |     35 |   0.12% |  51.43% |         1,200 |  14,907 |  22,757 |  68,060 | 452,405 |  0.00% |  0.00% |
|            6 |      5 |   0.02% |  40.00% |           142 |   1,800 |  24,579 |  57,663 |  96,593 |  0.00% |  0.00% |
|            7 |     58 |   0.19% |  82.76% |           150 |   2,400 |   2,400 |   2,400 | 377,145 |  0.00% |  0.00% |
|            8 |      2 |   0.01% |  50.00% |         2,400 |  15,754 |  29,108 |  42,461 |  55,815 |  0.00% |  0.00% |
|          all | 30,000 | 100.00% |  22.12% |      -170,000 |   2,327 |  19,052 |  54,506 | 891,586 | 10.65% | 12.90% |

| PAY_STATUS_4 | PAY_AMT4 min |   p25 |   p50 |    p75 |     max |     = 0 | coverage p50 | excluded |
| -----------: | -----------: | ----: | ----: | -----: | ------: | ------: | -----------: | -------: |
|           -2 |            0 |     0 |     0 |  1,769 | 330,982 |  57.01% |        1.000 |   58.33% |
|           -1 |            0 |    19 | 1,132 |  5,072 | 621,000 |  23.90% |        1.000 |   15.58% |
|            0 |            0 | 1,000 | 2,000 |  4,830 | 528,897 |   8.46% |        0.042 |    3.83% |
|            1 |          100 | 3,825 | 7,550 | 11,275 |  15,000 |   0.00% |        0.116 |    0.00% |
|            2 |            0 |     0 | 1,100 |  3,000 | 171,716 |  28.46% |        0.039 |    3.45% |
|            3 |            0 |     0 |     0 |  1,000 |  25,703 |  69.44% |        0.000 |    0.00% |
|            4 |            0 |     0 |     0 |      0 |  11,078 |  79.71% |        0.000 |    0.00% |
|            5 |            0 |     0 |     0 |      0 |       0 | 100.00% |        0.000 |    0.00% |
|            6 |            0 |     0 |     0 |      0 |  10,000 |  80.00% |        0.000 |    0.00% |
|            7 |            0 |     0 |     0 |      0 |       0 | 100.00% |        0.000 |    0.00% |
|            8 |            0 |     0 |     0 |      0 |       0 | 100.00% |        0.000 |    0.00% |
|          all |            0 |   296 | 1,500 |  4,013 | 621,000 |  21.36% |        0.052 |   13.87% |

## Mes 5 (mayo de 2005)

| PAY_STATUS_5 |   rows |   share | default | BILL_AMT5 min |    p25 |    p50 |     p75 |     max |    = 0 |   <= 0 |
| -----------: | -----: | ------: | ------: | ------------: | -----: | -----: | ------: | ------: | -----: | -----: |
|           -2 |  4,546 |  15.15% |  19.69% |       -53,007 |      0 |      0 |     697 | 265,852 | 58.86% | 68.19% |
|           -1 |  5,539 |  18.46% |  16.19% |        -9,584 |    475 |  1,954 |   7,334 | 514,114 |  8.68% | 10.81% |
|            0 | 16,947 |  56.49% |  18.85% |       -81,334 | 16,142 | 31,032 |  78,592 | 927,171 |  1.84% |  2.42% |
|            2 |  2,626 |   8.75% |  54.19% |       -15,306 | 16,048 | 30,478 |  65,778 | 823,540 |  1.45% |  1.98% |
|            3 |    178 |   0.59% |  63.48% |           300 |  7,964 | 20,031 |  50,750 | 381,863 |  0.00% |  0.00% |
|            4 |     84 |   0.28% |  60.71% |            37 |  7,922 | 21,629 |  59,807 | 503,914 |  0.00% |  0.00% |
|            5 |     17 |   0.06% |  58.82% |         1,200 |  1,250 | 22,192 |  56,225 | 145,533 |  0.00% |  0.00% |
|            6 |      4 |   0.01% |  75.00% |           142 |  1,386 | 11,413 | 108,482 | 370,850 |  0.00% |  0.00% |
|            7 |     58 |   0.19% |  82.76% |           150 |  2,400 |  2,400 |   2,400 | 105,083 |  0.00% |  0.00% |
|            8 |      1 |   0.00% | 100.00% |         2,400 |  2,400 |  2,400 |   2,400 |   2,400 |  0.00% |  0.00% |
|          all | 30,000 | 100.00% |  22.12% |       -81,334 |  1,763 | 18,104 |  50,190 | 927,171 | 11.69% | 13.87% |

| PAY_STATUS_5 | PAY_AMT5 min | p25 |   p50 |   p75 |     max |     = 0 | coverage p50 | excluded |
| -----------: | -----------: | --: | ----: | ----: | ------: | ------: | -----------: | -------: |
|           -2 |            0 |   0 |     0 | 1,442 | 379,267 |  59.88% |        1.000 |   60.93% |
|           -1 |            0 |  33 | 1,170 | 5,000 | 426,529 |  23.67% |        1.000 |   16.90% |
|            0 |            0 | 993 | 2,000 | 4,850 | 417,990 |  10.18% |        0.044 |    5.55% |
|            2 |            0 |   0 | 1,458 | 3,000 | 142,000 |  26.20% |        0.040 |    2.32% |
|            3 |            0 |   0 |     0 | 1,362 |  27,150 |  61.80% |        0.000 |    0.00% |
|            4 |            0 |   0 |     0 |     0 |  33,000 |  85.71% |        0.000 |    0.00% |
|            5 |            0 |   0 |     0 |     0 |  11,678 |  76.47% |        0.000 |    0.00% |
|            6 |            0 |   0 |     0 |     0 |       0 | 100.00% |        0.000 |    0.00% |
|            7 |            0 |   0 |     0 |     0 |   1,000 |  98.28% |        0.000 |    0.00% |
|            8 |            0 |   0 |     0 |     0 |       0 | 100.00% |        0.000 |    0.00% |
|          all |            0 | 252 | 1,500 | 4,032 | 426,529 |  22.34% |        0.056 |   15.69% |

## Mes 6 (abril de 2005)

| PAY_STATUS_6 |   rows |   share | default | BILL_AMT6 min |    p25 |    p50 |    p75 |     max |    = 0 |   <= 0 |
| -----------: | -----: | ------: | ------: | ------------: | -----: | -----: | -----: | ------: | -----: | -----: |
|           -2 |  4,895 |  16.32% |  20.04% |      -150,953 |      0 |      0 |    444 | 227,835 | 61.08% | 70.83% |
|           -1 |  5,740 |  19.13% |  16.99% |      -339,603 |    469 |  1,882 |  6,947 | 426,518 |  8.62% | 10.42% |
|            0 | 16,286 |  54.29% |  18.84% |      -209,051 | 16,137 | 31,268 | 78,771 | 961,664 |  3.19% |  3.81% |
|            2 |  2,766 |   9.22% |  50.65% |        -3,148 | 15,724 | 30,492 | 66,568 | 527,566 |  0.58% |  0.80% |
|            3 |    184 |   0.61% |  64.13% |           300 |  3,768 | 19,367 | 38,884 | 437,305 |  0.00% |  0.00% |
|            4 |     49 |   0.16% |  63.27% |            37 |  1,650 | 16,780 | 36,265 | 278,436 |  0.00% |  0.00% |
|            5 |     13 |   0.04% |  53.85% |         1,200 | 12,189 | 18,646 | 34,420 | 364,365 |  0.00% |  0.00% |
|            6 |     19 |   0.06% |  73.68% |           142 |  1,800 |  1,950 | 42,830 | 527,711 |  0.00% |  0.00% |
|            7 |     46 |   0.15% |  82.61% |           150 |  2,396 |  2,400 |  2,400 |   2,646 |  0.00% |  0.00% |
|            8 |      2 |   0.01% | 100.00% |         2,400 | 27,680 | 52,961 | 78,242 | 103,522 |  0.00% |  0.00% |
|          all | 30,000 | 100.00% |  22.12% |      -339,603 |  1,256 | 17,071 | 49,198 | 961,664 | 13.40% | 15.69% |

| PAY_STATUS_6 | PAY_AMT6 min |   p25 |   p50 |   p75 |     max |     = 0 | coverage p50 | excluded |
| -----------: | -----------: | ----: | ----: | ----: | ------: | ------: | -----------: | -------: |
|           -2 |            0 |     0 |     0 | 1,165 | 228,300 |  63.74% |          n/a |      n/a |
|           -1 |            0 |   130 | 1,140 | 5,094 | 528,666 |  23.41% |          n/a |      n/a |
|            0 |            0 | 1,000 | 2,000 | 4,870 | 443,001 |  10.35% |          n/a |      n/a |
|            2 |            0 |     0 | 1,300 | 3,000 | 254,000 |  27.91% |          n/a |      n/a |
|            3 |            0 |     0 |     0 |     0 |  14,000 |  77.72% |          n/a |      n/a |
|            4 |            0 |     0 |     0 |     0 |   6,608 |  77.55% |          n/a |      n/a |
|            5 |            0 |     0 |     0 | 1,500 |   5,000 |  61.54% |          n/a |      n/a |
|            6 |            0 |     0 |     0 |   108 |  10,478 |  73.68% |          n/a |      n/a |
|            7 |            0 |     0 |     0 |     0 |       0 | 100.00% |          n/a |      n/a |
|            8 |            0 |     0 |     0 |     0 |       0 | 100.00% |          n/a |      n/a |
|          all |            0 |   118 | 1,500 | 4,000 | 528,666 |  23.91% |          n/a |      n/a |

---

## Lectura de H1 — el código `-2`

**La evidencia es parcialmente consistente, y la parte que falla es la del mes 1.**

La hipótesis pide que las filas con `-2` tengan `BILL_AMT` cercano a cero. La proporción
con saldo exactamente cero crece de forma monótona con la antigüedad del mes:

| Mes | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BILL_AMT = 0` en las filas con `-2` | 25,15% | 44,63% | 50,99% | 56,42% | 58,86% | 61,08% |
| `p50` de `BILL_AMT` | 1.179 | 0 | 0 | 0 | 0 | 0 |
| `p75` de `BILL_AMT` | 5.819 | 2.564 | 1.772 | 999 | 697 | 444 |

**Consistente:** en los meses 2 a 6 la mediana es exactamente 0 y el p75 cae hasta 444
NT$ en el mes 6. Sobre una columna cuyo p75 general es 49.198 NT$, eso es un subconjunto
concentrado cerca de cero.

**Inconsistente:** en el mes 1 solo el 25,15% de las 2.759 filas tiene saldo cero, la
mediana es 1.179 y el p75 es 5.819. Tres de cada cuatro filas con `-2` en septiembre
tienen saldo distinto de cero. La hipótesis, tal como está enunciada, no se sostiene en
ese mes.

**Insuficiente para una lectura fuerte en cualquier mes:** ni siquiera en el mes 6 el
saldo cero es universal — el 38,92% restante tiene saldo distinto de cero, y el máximo
observado bajo `-2` llega a 855.086 NT$ en el mes 3. Una lectura de "sin consumo" que
admite un saldo de 855.086 NT$ necesita explicar esas filas, y esta medición no lo hace.

Dos mediciones adicionales sobre el mismo subconjunto, que no contradicen ni confirman la
hipótesis pero acotan qué más es compatible con ella:

- El `coverage p50` de `-2` es exactamente **1,000** en los cinco meses donde se puede
  calcular — con entre 34,51% y 60,93% de las filas excluidas por denominador no
  positivo. Entre las filas con `-2` que **sí** tenían saldo el mes anterior, la mediana
  pagó ese saldo completo. El código `-1` muestra el mismo 1,000, con muchas menos
  exclusiones (12,36% a 16,90%).
- La tasa de default bajo `-2` va de 13,23% (mes 1) a 20,04% (mes 6), siempre por debajo
  del baseline de 22,12%. El código separa la clase, pero débilmente y en la dirección de
  menor riesgo.

## Lectura de H2 — el código `0`

**La evidencia es consistente con las tres partes de la hipótesis.**

| Lo que pide la hipótesis | Lo que mide el dato (mes 1) | Otros meses |
| --- | --- | --- |
| `BILL_AMT` alto | `p50` = 49.605 contra 22.382 de la columna entera | `p50` entre 31.032 y 48.428 |
| Saldo presente, no cero | `BILL_AMT = 0` en el **0,00%** de las 14.737 filas | entre 1,38% y 3,19% |
| `PAY_AMT` positivo | `PAY_AMT = 0` en solo el 1,77%, contra 17,50% de la columna entera; `p50` = 3.000 | `= 0` entre 6,36% y 10,35% |
| Pago **menor** que el saldo | `coverage p50` = **0,057**, con solo 1,91% de filas excluidas | 0,055 · 0,047 · 0,042 · 0,044 |

Sobre el `p50` de `BILL_AMT` conviene una precisión: entre los códigos con más de mil
filas, el `0` tiene el `p50` más alto en los seis meses, y en el mes 5 lo hace por poco
(31.032 contra 30.478 del código `2`). Hay códigos con `p50` mayor —el `7` mide 126.220 en
el mes 1 y el `6` mide 77.765 en el mes 2— pero sobre 9 y 12 filas respectivamente, así
que no son una comparación.

El `coverage p50` es la medición que más carga: 0,057 significa que la mediana de las
filas con código `0` pagó alrededor del 5,7% del saldo del mes anterior — positivo, y muy
lejos de saldarlo. Y el `excluded` de 1,91% dice que esa mediana se calculó sobre el
98,09% del subconjunto, no sobre un resto seleccionado. El contraste con los códigos `-1`
y `-2`, cuyo `coverage p50` es 1,000, es la comparación más directa que produce este
documento.

La tasa de default bajo `0` es **12,81%** en el mes 1, la más baja de todos los códigos de
esa columna, y queda entre 15,91% y 18,85% en el resto. Bajo el baseline de 22,12% en los
seis meses.

**No se afirma qué significa el código.** Se afirma que el perfil medido —saldo alto, sin
ceros, pago positivo, cobertura mediana del 4% al 6%— es compatible con la hipótesis y no
se parece al de ningún otro código de la columna.

---

# Mediciones — H3

## H3.1 · Frecuencia de cada código por columna

Frecuencia absoluta:

| code | PAY_STATUS_1 | PAY_STATUS_2 | PAY_STATUS_3 | PAY_STATUS_4 | PAY_STATUS_5 | PAY_STATUS_6 |
| ---: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
|   -2 |        2,759 |        3,782 |        4,085 |        4,348 |        4,546 |        4,895 |
|   -1 |        5,686 |        6,050 |        5,938 |        5,687 |        5,539 |        5,740 |
|    0 |       14,737 |       15,730 |       15,764 |       16,455 |       16,947 |       16,286 |
|    1 |        3,688 |           28 |            4 |            2 |            0 |            0 |
|    2 |        2,667 |        3,927 |        3,819 |        3,159 |        2,626 |        2,766 |
|    3 |          322 |          326 |          240 |          180 |          178 |          184 |
|    4 |           76 |           99 |           76 |           69 |           84 |           49 |
|    5 |           26 |           25 |           21 |           35 |           17 |           13 |
|    6 |           11 |           12 |           23 |            5 |            4 |           19 |
|    7 |            9 |           20 |           27 |           58 |           58 |           46 |
|    8 |           19 |            1 |            3 |            2 |            1 |            2 |

Proporción sobre las 30.000 filas:

| code | PAY_STATUS_1 | PAY_STATUS_2 | PAY_STATUS_3 | PAY_STATUS_4 | PAY_STATUS_5 | PAY_STATUS_6 |
| ---: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
|   -2 |        9.20% |       12.61% |       13.62% |       14.49% |       15.15% |       16.32% |
|   -1 |       18.95% |       20.17% |       19.79% |       18.96% |       18.46% |       19.13% |
|    0 |       49.12% |       52.43% |       52.55% |       54.85% |       56.49% |       54.29% |
|    1 |       12.29% |        0.09% |        0.01% |        0.01% |        0.00% |        0.00% |
|    2 |        8.89% |       13.09% |       12.73% |       10.53% |        8.75% |        9.22% |
|    3 |        1.07% |        1.09% |        0.80% |        0.60% |        0.59% |        0.61% |
|    4 |        0.25% |        0.33% |        0.25% |        0.23% |        0.28% |        0.16% |
|    5 |        0.09% |        0.08% |        0.07% |        0.12% |        0.06% |        0.04% |
|    6 |        0.04% |        0.04% |        0.08% |        0.02% |        0.01% |        0.06% |
|    7 |        0.03% |        0.07% |        0.09% |        0.19% |        0.19% |        0.15% |
|    8 |        0.06% |        0.00% |        0.01% |        0.01% |        0.00% |        0.01% |

## H3.2 · Matrices de transición

Filas: código en el mes `m`. Columnas: distribución del código en el mes `m+1`, que es el
mes **anterior en el tiempo**. Cada fila suma 100%.

**De `PAY_STATUS_1` a `PAY_STATUS_2`:**

| m=1 |   rows |     -2 |     -1 |      0 |     1 |      2 |      3 |      4 |       5 |       6 |       7 |     8 |
| --: | -----: | -----: | -----: | -----: | ----: | -----: | -----: | -----: | ------: | ------: | ------: | ----: |
|  -2 |  2,759 | 92.82% |  7.00% |  0.00% | 0.00% |  0.18% |  0.00% |  0.00% |   0.00% |   0.00% |   0.00% | 0.00% |
|  -1 |  5,686 |  0.00% | 81.62% | 10.60% | 0.00% |  6.81% |  0.83% |  0.09% |   0.05% |   0.00% |   0.00% | 0.00% |
|   0 | 14,737 |  0.00% |  3.24% | 96.76% | 0.00% |  0.00% |  0.00% |  0.00% |   0.00% |   0.00% |   0.00% | 0.00% |
|   1 |  3,688 | 33.11% | 16.59% |  0.08% | 0.76% | 45.34% |  2.96% |  0.87% |   0.19% |   0.05% |   0.03% | 0.03% |
|   2 |  2,667 |  0.00% |  4.72% | 32.43% | 0.00% | 59.66% |  2.66% |  0.52% |   0.00% |   0.00% |   0.00% | 0.00% |
|   3 |    322 |  0.00% |  0.00% |  0.00% | 0.00% | 84.47% | 12.73% |  2.48% |   0.31% |   0.00% |   0.00% | 0.00% |
|   4 |     76 |  0.00% |  0.00% |  0.00% | 0.00% |  0.00% | 76.32% | 19.74% |   3.95% |   0.00% |   0.00% | 0.00% |
|   5 |     26 |  0.00% |  0.00% |  0.00% | 0.00% |  0.00% |  0.00% | 96.15% |   0.00% |   3.85% |   0.00% | 0.00% |
|   6 |     11 |  0.00% |  0.00% |  0.00% | 0.00% |  0.00% |  0.00% |  0.00% | 100.00% |   0.00% |   0.00% | 0.00% |
|   7 |      9 |  0.00% |  0.00% |  0.00% | 0.00% |  0.00% |  0.00% |  0.00% |   0.00% | 100.00% |   0.00% | 0.00% |
|   8 |     19 |  0.00% |  0.00% |  0.00% | 0.00% |  0.00% |  0.00% |  0.00% |   0.00% |   0.00% | 100.00% | 0.00% |

**De `PAY_STATUS_2` a `PAY_STATUS_3`:**

| m=2 |   rows |     -2 |     -1 |      0 |      1 |      2 |      3 |       4 |       5 |       6 |       7 |     8 |
| --: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ------: | ------: | ------: | ------: | ----: |
|  -2 |  3,782 | 89.29% | 10.60% |  0.03% |  0.00% |  0.08% |  0.00% |   0.00% |   0.00% |   0.00% |   0.00% | 0.00% |
|  -1 |  6,050 |  6.64% | 77.24% |  9.44% |  0.00% |  6.36% |  0.31% |   0.00% |   0.00% |   0.00% |   0.00% | 0.00% |
|   0 | 15,730 |  1.52% |  3.48% | 89.47% |  0.00% |  5.32% |  0.17% |   0.03% |   0.01% |   0.00% |   0.00% | 0.00% |
|   1 |     28 | 42.86% | 32.14% |  3.57% | 14.29% |  7.14% |  0.00% |   0.00% |   0.00% |   0.00% |   0.00% | 0.00% |
|   2 |  3,927 |  1.40% |  7.82% | 28.44% |  0.00% | 58.62% |  2.24% |   0.56% |   0.15% |   0.05% |   0.64% | 0.08% |
|   3 |    326 |  0.00% |  0.00% |  0.00% |  0.00% | 88.96% |  7.36% |   2.76% |   0.31% |   0.31% |   0.31% | 0.00% |
|   4 |     99 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 82.83% |  16.16% |   1.01% |   0.00% |   0.00% | 0.00% |
|   5 |     25 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 100.00% |   0.00% |   0.00% |   0.00% | 0.00% |
|   6 |     12 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |   0.00% | 100.00% |   0.00% |   0.00% | 0.00% |
|   7 |     20 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |   0.00% |   0.00% | 100.00% |   0.00% | 0.00% |
|   8 |      1 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |   0.00% |   0.00% |   0.00% | 100.00% | 0.00% |

**De `PAY_STATUS_3` a `PAY_STATUS_4`:**

| m=3 |   rows |     -2 |     -1 |      0 |      1 |      2 |      3 |      4 |      5 |     6 |      7 |      8 |
| --: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ----: | -----: | -----: |
|  -2 |  4,085 | 89.40% | 10.55% |  0.00% |  0.00% |  0.05% |  0.00% |  0.00% |  0.00% | 0.00% |  0.00% |  0.00% |
|  -1 |  5,938 |  5.94% | 73.75% | 16.52% |  0.00% |  3.70% |  0.05% |  0.03% |  0.00% | 0.00% |  0.00% |  0.00% |
|   0 | 15,764 |  1.66% |  3.51% | 90.43% |  0.00% |  4.26% |  0.12% |  0.01% |  0.01% | 0.00% |  0.00% |  0.00% |
|   1 |      4 |  0.00% |  0.00% | 25.00% | 50.00% | 25.00% |  0.00% |  0.00% |  0.00% | 0.00% |  0.00% |  0.00% |
|   2 |  3,819 |  2.15% |  8.46% | 31.89% |  0.00% | 53.94% |  2.04% |  0.50% |  0.18% | 0.03% |  0.81% |  0.00% |
|   3 |    240 |  0.00% |  0.00% |  0.00% |  0.00% | 85.00% | 13.75% |  1.25% |  0.00% | 0.00% |  0.00% |  0.00% |
|   4 |     76 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 61.84% | 34.21% |  2.63% | 0.00% |  0.00% |  1.32% |
|   5 |     21 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 80.95% | 19.05% | 0.00% |  0.00% |  0.00% |
|   6 |     23 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 91.30% | 8.70% |  0.00% |  0.00% |
|   7 |     27 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 7.41% | 92.59% |  0.00% |
|   8 |      3 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 0.00% | 66.67% | 33.33% |

**De `PAY_STATUS_4` a `PAY_STATUS_5`:**

| m=4 |   rows |     -2 |     -1 |      0 |      2 |      3 |      4 |      5 |      6 |      7 |      8 |
| --: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: |
|  -2 |  4,348 | 89.93% | 10.00% |  0.07% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |
|  -1 |  5,687 |  5.91% | 72.31% | 19.64% |  2.04% |  0.09% |  0.02% |  0.00% |  0.00% |  0.00% |  0.00% |
|   0 | 16,455 |  1.44% |  4.87% | 90.09% |  3.48% |  0.09% |  0.01% |  0.01% |  0.00% |  0.00% |  0.00% |
|   1 |      2 |  0.00% |  0.00% | 50.00% | 50.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |
|   2 |  3,159 |  1.99% |  6.01% | 31.69% | 57.14% |  2.47% |  0.66% |  0.03% |  0.00% |  0.00% |  0.00% |
|   3 |    180 |  0.00% |  0.00% |  0.00% | 72.78% | 22.78% |  2.22% |  1.67% |  0.00% |  0.56% |  0.00% |
|   4 |     69 |  0.00% |  0.00% |  0.00% |  0.00% | 56.52% | 37.68% |  5.80% |  0.00% |  0.00% |  0.00% |
|   5 |     35 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 85.71% | 14.29% |  0.00% |  0.00% |  0.00% |
|   6 |      5 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 60.00% | 20.00% | 20.00% |  0.00% |
|   7 |     58 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  5.17% | 94.83% |  0.00% |
|   8 |      2 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 50.00% | 50.00% |

**De `PAY_STATUS_5` a `PAY_STATUS_6`:**

| m=5 |   rows |     -2 |     -1 |      0 |      2 |      3 |      4 |      5 |      6 |      7 |       8 |
| --: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ------: |
|  -2 |  4,546 | 90.23% |  9.77% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |   0.00% |
|  -1 |  5,539 |  6.97% | 74.20% | 15.06% |  3.57% |  0.18% |  0.00% |  0.00% |  0.02% |  0.00% |   0.00% |
|   0 | 16,947 |  2.09% |  6.46% | 86.95% |  4.32% |  0.18% |  0.00% |  0.00% |  0.01% |  0.00% |   0.00% |
|   2 |  2,626 |  2.02% |  3.50% | 27.30% | 64.81% |  1.79% |  0.34% |  0.15% |  0.08% |  0.00% |   0.00% |
|   3 |    178 |  0.00% |  0.00% |  0.00% | 75.28% | 20.79% |  2.81% |  1.12% |  0.00% |  0.00% |   0.00% |
|   4 |     84 |  0.00% |  0.00% |  0.00% |  0.00% | 70.24% | 27.38% |  1.19% |  1.19% |  0.00% |   0.00% |
|   5 |     17 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 70.59% | 17.65% | 11.76% |  0.00% |   0.00% |
|   6 |      4 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 75.00% | 25.00% |  0.00% |   0.00% |
|   7 |     58 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 18.97% | 79.31% |   1.72% |
|   8 |      1 |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% |  0.00% | 100.00% |

## H3.3 · Correlación de Spearman

Con el target:

|       column | spearman |
| -----------: | -------: |
| PAY_STATUS_1 |    0.292 |
| PAY_STATUS_2 |    0.217 |
| PAY_STATUS_3 |    0.195 |
| PAY_STATUS_4 |    0.174 |
| PAY_STATUS_5 |    0.159 |
| PAY_STATUS_6 |    0.143 |

Entre columnas contiguas:

|                        pair | spearman |
| --------------------------: | -------: |
| PAY_STATUS_1 ~ PAY_STATUS_2 |    0.627 |
| PAY_STATUS_2 ~ PAY_STATUS_3 |    0.799 |
| PAY_STATUS_3 ~ PAY_STATUS_4 |    0.801 |
| PAY_STATUS_4 ~ PAY_STATUS_5 |    0.822 |
| PAY_STATUS_5 ~ PAY_STATUS_6 |    0.821 |

## Lectura de H3 — la escala del mes 1

**La evidencia es consistente, y lo es por tres caminos independientes.**

**1 · La frecuencia.** El código `1` aparece 3.688 veces en el mes 1 (12,29% de la tabla)
y 28, 4, 2, 0 y 0 veces en los meses 2 a 6. Es un factor de 132 contra el segundo mes más
poblado, y el código está **completamente ausente** de `PAY_STATUS_5` y `PAY_STATUS_6`. El
código `8` muestra la misma asimetría en menor escala: 19 filas en el mes 1 contra 1, 3, 2,
1 y 2 en el resto. Si las seis columnas se generaran con el mismo procedimiento, una
diferencia de dos órdenes de magnitud en un código no sería un accidente de muestreo.

**2 · La correlación entre columnas contiguas.** Las cuatro parejas que no involucran al
mes 1 están dentro de 0,023 entre sí (0,799 · 0,801 · 0,822 · 0,821). La pareja (1, 2)
mide **0,627**, o sea 0,172 por debajo de la más baja de las otras cuatro. La ruptura está
exactamente donde la hipótesis la predice y no aparece en ningún otro punto de la serie.

**3 · Las matrices de transición.** Dos observaciones concretas.

- La fila del código `1` en la matriz 1→2 reparte **33,11% a `-2`**, 16,59% a `-1`, 0,08%
  a `0` y 45,34% a `2`, y solo el 0,76% permanece en `1`. Ninguna otra fila de ninguna
  otra matriz reparte así. Bajo una escala homogénea, esta fila diría que un tercio de los
  clientes que están en el estado `1` en septiembre estaban en el estado `-2` en agosto.
- La fila del código `0` en la matriz 1→2 manda **0,00%** al código `2`. Las mismas filas
  en las otras cuatro matrices mandan 5,32%, 4,26%, 3,48% y 4,32%. Es un cero estructural,
  no un cero por muestra chica: son 14.737 filas.

En cambio, los códigos altos se comportan igual en todas las matrices: `k` en el mes `m`
va a `k-1` en el mes `m+1` con probabilidad dominante (en la matriz 1→2: 84,47%, 76,32%,
96,15%, 100%, 100% y 100% para los códigos 3 a 8), y esa estructura se repite en las
cinco. **La diferencia de escala, si existe, no afecta a toda la columna por igual: los
códigos ≥ 2 se comportan igual entre meses y la ruptura se concentra en `1`, `0` y `-1`.**

**Insuficiente por sí sola:** la correlación con el target. La caída 0,292 → 0,217 entre
el mes 1 y el 2 es 3,4 veces más grande que la siguiente (0,217 → 0,195), lo cual es
sugerente, pero una serie temporal en la que el mes más reciente predice mejor que los
anteriores produce exactamente esa forma sin necesidad de ninguna diferencia de escala. Se
reporta porque se pidió; no sostiene la hipótesis por sí misma.

---

# Fuera de las hipótesis — `EDUCATION` y `MARRIAGE`

Los códigos no documentados no están escritos a mano en el script: se derivan restando los
niveles que declara `schema.CATEGORICAL_LEVELS` de los valores que contiene el dato.

`EDUCATION` — niveles documentados: `[1, 2, 3, 4]` — códigos no documentados presentes: `[0, 5, 6]`

|          EDUCATION | documented |   rows |   share | default |
| -----------------: | ---------: | -----: | ------: | ------: |
|                  0 |         NO |     14 |   0.05% |   0.00% |
|                  1 |        yes | 10,585 |  35.28% |  19.23% |
|                  2 |        yes | 14,030 |  46.77% |  23.73% |
|                  3 |        yes |  4,917 |  16.39% |  25.16% |
|                  4 |        yes |    123 |   0.41% |   5.69% |
|                  5 |         NO |    280 |   0.93% |   6.43% |
|                  6 |         NO |     51 |   0.17% |  15.69% |
|  documented levels |        yes | 29,655 |  98.85% |  22.29% |
| undocumented codes |         NO |    345 |   1.15% |   7.54% |
|                all |          - | 30,000 | 100.00% |  22.12% |

`MARRIAGE` — niveles documentados: `[1, 2, 3]` — códigos no documentados presentes: `[0]`

|           MARRIAGE | documented |   rows |   share | default |
| -----------------: | ---------: | -----: | ------: | ------: |
|                  0 |         NO |     54 |   0.18% |   9.26% |
|                  1 |        yes | 13,659 |  45.53% |  23.47% |
|                  2 |        yes | 15,964 |  53.21% |  20.93% |
|                  3 |        yes |    323 |   1.08% |  26.01% |
|  documented levels |        yes | 29,946 |  99.82% |  22.14% |
| undocumented codes |         NO |     54 |   0.18% |   9.26% |
|                all |          - | 30,000 | 100.00% |  22.12% |

## Lectura

**`EDUCATION`: los tres códigos no documentados se parecen al nivel 4, no a los niveles
1, 2 y 3.** Agrupados son 345 filas (1,15%) con 7,54% de default, contra 22,29% de los
niveles documentados. El nivel documentado más cercano es el `4` ("others"), con 5,69%
sobre 123 filas; los niveles 1, 2 y 3 miden 19,23%, 23,73% y 25,16%. Que `0`, `5` y `6` se
comporten como el `4` y no como los otros tres es compatible con que pertenezcan a la
misma familia de casos residuales, y es incompatible con que sean niveles educativos
ordinarios mal codificados.

**`MARRIAGE`: el código `0` no se parece a ningún nivel documentado, tampoco al `3`.** Sus
54 filas tienen 9,26% de default, contra 23,47%, 20,93% y 26,01% de los niveles 1, 2 y 3.
Está por debajo de los tres, incluido el nivel `3` ("others"), que es 2,8 veces más alto.
Esto es lo contrario de lo que pasa en `EDUCATION`: allí el código sin documentar se
alinea con el nivel residual; acá no se alinea con nada.

**Advertencia de tamaño de muestra, que aplica a las dos lecturas.** Los subconjuntos son
de 14, 51 y 54 filas. Una tasa de default de 0,00% sobre 14 filas no distingue una
población de riesgo cero de una población de riesgo 15%: con 14 observaciones, ver cero
positivos es un resultado frecuente incluso bajo el baseline del 22,12%. **Este script no
calcula intervalos de confianza**, así que las lecturas de arriba se apoyan en el
subconjunto agrupado (345 filas para `EDUCATION`) y en el código `5` (280 filas), que son
los únicos dos con tamaño suficiente para sostener una comparación. Las cifras de los
códigos `0` y `6` de `EDUCATION` y del código `0` de `MARRIAGE` se reportan pero no
sostienen nada por sí solas.

---

# Preguntas abiertas — qué no se puede determinar con este dataset

1. **A qué resumen se refiere `PAY_STATUS_m`.** Todo el bloque H1/H2 mide `BILL_AMT{m}`
   contra `PAY_STATUS_{m}` porque comparten índice de mes. Pero si el estado de pago del
   mes `m` describiera el resumen del mes `m+1` —el ciclo que efectivamente se estaba
   pagando— la medición correcta para H1 sería `BILL_AMT{m+1}`, y el resultado del mes 1
   podría cambiar. Este documento no mide esa alineación alternativa y el dataset no
   contiene nada que la resuelva: haría falta la especificación del proveedor.

2. **Si `-1` y `-2` son dos cosas distintas.** Los dos tienen `coverage p50` de exactamente
   1,000 en los cinco meses calculables. Se diferencian en la proporción de saldo cero
   (`-1`: 0,04% a 8,68%; `-2`: 25,15% a 61,08%) y en cuántas filas quedan fuera del ratio
   (`-1`: 12,36% a 16,90%; `-2`: 34,51% a 60,93%). Que esa diferencia sea una distinción
   de negocio o un artefacto de codificación no se puede establecer midiendo.

3. **Por qué el mes 1 tiene una escala distinta, si la tiene.** La evidencia dice *que*
   difiere, no *por qué*. Un cambio de sistema de registro, una definición distinta para
   el mes más reciente y un error de extracción producen la misma huella en estas
   mediciones.

4. **Qué hacer con el código `1` en los meses 5 y 6, donde tiene cero filas.** Un modelo
   entrenado sobre estas columnas nunca verá ese nivel ahí. Si aparece en producción, no
   hay dato con qué estimarlo. Esto es una restricción del dataset, no una decisión.

5. **Si los códigos `0`, `5` y `6` de `EDUCATION` son tres cosas o una.** Con 14, 280 y 51
   filas y sin intervalos de confianza, la única afirmación sostenible es sobre el grupo.

6. **El significado de cualquiera de estos códigos.** Ninguna cantidad de medición sobre
   este archivo puede recuperar una definición que la fuente no publicó. Lo que sí se
   puede establecer —y es lo que hay acá— es cómo se comportan, contra qué se parecen y
   contra qué no.

7. **Si el comportamiento observado se sostiene fuera de este período.** El dataset cubre
   seis meses de 2005 y no tiene fecha de originación de cada crédito, así que no se puede
   determinar si estas distribuciones son estables en el tiempo ni construir una
   validación fuera de período.
