# Evidencia — Disparidad del modelo productivo entre grupos demográficos

Medición de registro. Describe **qué se midió y qué salió**, con el tamaño de cada grupo al
lado. **No saca conclusiones sobre qué hacer**: mitigar es una decisión con alternativas y no
corresponde a este documento. La lectura vive en la entrada 010 de `docs/EVALUATION.md` y en
la sección de equidad del `docs/MODEL_CARD.md`.

- **Fecha:** 2026-08-26
- **Reproducción:** `uv run python scripts/run_fairness_analysis.py`
- **Objeto medido:** el modelo productivo —random forest tuneado, `class_weight=None`, más
  calibración sigmoide— en el umbral operativo **0,160**.
- **Datos:** UCI 350, 30.000 filas, prevalencia 0,221200.
- **Protocolo:** probabilidades **fuera de fold**,
  `StratifiedKFold(5, shuffle=True, random_state=42)`. Ninguna fila fue puntuada por un
  modelo ajustado con ella.
- **Atributos protegidos:** `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`. `LIMIT_BAL` **no** se
  cuenta como protegido: el cupo otorgado es una propiedad de la cuenta, no de la persona.

## Definiciones usadas

`A` es el grupo, `Y = 1` significa que el cliente incumplió de verdad, e `Ŷ = 1` que el
modelo lo puntúa en o por encima del umbral, es decir **recomienda rechazarlo**. El rechazo
es el resultado adverso.

| Nombre | Fórmula | Qué mide |
| --- | --- | --- |
| **Paridad demográfica** | `P(Ŷ=1 \| A=a)` igual para todo `a` | Si todos los grupos se rechazan a la misma tasa. **Ignora** si incumplen a la misma tasa |
| Razón de impacto dispar | `min_a / max_a` de lo anterior | La misma idea como cociente. La guía estadounidense señala por debajo de 0,80 |
| **Equidad de oportunidad** | `P(Ŷ=1 \| Y=1, A=a)` igual para todo `a` | Tasa de verdaderos positivos: de los que sí incumplieron, cuántos se atraparon |
| Tasa de falsos positivos | `P(Ŷ=1 \| Y=0, A=a)` | **De los que habrían pagado, cuántos se rechazaron por error** |

## Piso de tamaño de grupo

**500 filas.** El error estándar de una proporción cercana a la prevalencia es
`sqrt(0,22 × 0,78 / n)`: **1,9 puntos porcentuales con n = 500**, 4,2 con n = 100 y 5,7 con
n = 54. Por debajo de 500 el ruido de una tasa es del mismo tamaño que las disparidades que
se buscan. Los grupos pequeños **se imprimen con su tamaño** y quedan **fuera** de los
estadísticos de máximo-menos-mínimo.

## Tramos de edad

`21-29`, `30-39`, `40-49`, `50+`. Bandas por década, que son la convención del dominio. La
última se deja abierta **por una medición**: hay 2.341 clientes de 50-59 y solo **339** de 60
o más, y un tramo de 339 caería por debajo del piso.

---

## Resultado — modelo completo

### SEX

| Grupo | n | % libro | Rechazo | Default real | FPR | TPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.888 | 39,6% | 0,4192 | 0,2417 | 0,3188 | 0,7344 | 0,2656 |
| 2 | 18.112 | 60,4% | 0,3672 | 0,2078 | 0,2782 | 0,7066 | 0,2934 |

### EDUCATION

| Grupo | n | % libro | Rechazo | Default real | FPR | TPR | FNR | Nota |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 14 | 0,0% | 0,0714 | 0,0000 | 0,0714 | — | — | n < 500 |
| 1 | 10.585 | 35,3% | 0,3347 | 0,1923 | 0,2503 | 0,6891 | 0,3109 | |
| 2 | 14.030 | 46,8% | 0,4084 | 0,2373 | 0,3095 | 0,7261 | 0,2739 | |
| 3 | 4.917 | 16,4% | 0,4545 | 0,2516 | 0,3533 | 0,7559 | 0,2441 | |
| 4 | 123 | 0,4% | 0,2033 | 0,0569 | 0,2069 | 0,1429 | 0,8571 | n < 500 |
| 5 | 280 | 0,9% | 0,2786 | 0,0643 | 0,2634 | 0,5000 | 0,5000 | n < 500 |
| 6 | 51 | 0,2% | 0,4510 | 0,1569 | 0,4651 | 0,3750 | 0,6250 | n < 500 |

### MARRIAGE

| Grupo | n | % libro | Rechazo | Default real | FPR | TPR | FNR | Nota |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 54 | 0,2% | 0,4259 | 0,0926 | 0,4082 | 0,6000 | 0,4000 | n < 500 |
| 1 | 13.659 | 45,5% | 0,3922 | 0,2347 | 0,2929 | 0,7158 | 0,2842 | |
| 2 | 15.964 | 53,2% | 0,3816 | 0,2093 | 0,2920 | 0,7201 | 0,2799 | |
| 3 | 323 | 1,1% | 0,5046 | 0,2601 | 0,4100 | 0,7738 | 0,2262 | n < 500 |

### AGE

| Grupo | n | % libro | Rechazo | Default real | FPR | TPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21-29 | 9.618 | 32,1% | 0,4135 | 0,2284 | 0,3165 | 0,7410 | 0,2590 |
| 30-39 | 11.238 | 37,5% | 0,3505 | 0,2025 | 0,2640 | 0,6911 | 0,3089 |
| 40-49 | 6.464 | 21,5% | 0,3889 | 0,2297 | 0,2908 | 0,7178 | 0,2822 |
| 50+ | 2.680 | 8,9% | 0,4496 | 0,2530 | 0,3511 | 0,7404 | 0,2596 |

## Brechas, sobre grupos de al menos 500 filas

| Atributo | Paridad (dif.) | Razón | Equidad de oportunidad | Brecha FPR | Brecha de tasa base |
| --- | ---: | ---: | ---: | ---: | ---: |
| SEX | 0,0520 | 0,8759 | 0,0278 | 0,0406 | 0,0339 |
| **EDUCATION** | **0,1198** | **0,7364** | 0,0668 | **0,1029** | 0,0592 |
| MARRIAGE | 0,0106 | 0,9730 | 0,0043 | 0,0009 | 0,0254 |
| **AGE** | **0,0991** | **0,7796** | 0,0499 | **0,0871** | 0,0505 |

Grupos excluidos de estos máximos y mínimos por tamaño: `EDUCATION` 0, 4, 5 y 6;
`MARRIAGE` 0 y 3.

## Brecha de FPR traducida a personas

Cuántos clientes que **habrían pagado** son rechazados de más, frente a la tasa del grupo de
referencia del mismo atributo.

| Grupo | Pagadores | Rechazados | Con el FPR de la referencia | Exceso | Referencia |
| --- | ---: | ---: | ---: | ---: | --- |
| EDUCATION 2 | 10.701 | 3.312 | 2.678 | **+633** | EDUCATION 1 |
| EDUCATION 3 | 3.680 | 1.300 | 921 | **+379** | EDUCATION 1 |
| AGE 21-29 | 7.421 | 2.349 | 1.959 | **+390** | AGE 30-39 |
| AGE 50+ | 2.002 | 703 | 529 | **+174** | AGE 30-39 |
| SEX 1 | 9.015 | 2.874 | 2.508 | **+366** | SEX 2 |

---

## Resultado — modelo ciego a los atributos protegidos

Mismo preprocesador, mismo estimador, mismos folds, misma semilla. El clasificador no recibe
ninguna columna derivada de `SEX`, `EDUCATION`, `MARRIAGE` ni `AGE`.

### Desempeño

| Modelo | PR-AUC | Columnas |
| --- | ---: | ---: |
| Completo | 0,5640 ± 0,0075 | 110 |
| Ciego | 0,5619 ± 0,0093 | 99 |
| **Diferencia** | **−0,0020** | −11 |

Umbral de significancia práctica del proyecto: 0,02. La diferencia queda **dentro del ruido**.

### Brechas, completo contra ciego

| Atributo | Paridad completo | Paridad ciego | FPR completo | FPR ciego | Equidad op. completo | Equidad op. ciego |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SEX | 0,0520 | 0,0484 | 0,0406 | 0,0369 | 0,0278 | 0,0250 |
| EDUCATION | 0,1198 | 0,1085 | 0,1029 | 0,0908 | 0,0668 | 0,0590 |
| MARRIAGE | 0,0106 | 0,0080 | 0,0009 | **0,0188** | 0,0043 | **0,0184** |
| AGE | 0,0991 | 0,0712 | 0,0871 | 0,0606 | 0,0499 | **0,0554** |

Reducción de la brecha de FPR al cegar el modelo, calculada sobre los valores completos y
no sobre los redondeados de la tabla: SEX **−9,18%**, EDUCATION **−11,75%**, AGE **−30,44%**,
MARRIAGE **+1.936,69%** (de 0,000924 a 0,018819; la brecha del modelo completo era
prácticamente nula, así que el cociente está dominado por un denominador diminuto).

---

## Nota de alcance

Las cifras se midieron sobre **esta** población —Taiwán, 2005— en **este** umbral (0,160).
Un umbral distinto produce otras tasas de rechazo y otras brechas; la sensibilidad del umbral
al supuesto de costos está en la entrada 008 de `docs/EVALUATION.md`.

Los códigos `0`, `5` y `6` de `EDUCATION` y el `0` de `MARRIAGE` **no están documentados por
la fuente** (ADR-0004). Se reportan por su valor crudo, sin colapsar, precisamente para que
su tamaño quede visible.

Esta medición cubre disparidad de **resultado y de error** entre grupos. **No** cubre
interseccionalidad —combinaciones como sexo × edad, que no se midieron—, ni calibración por
grupo, ni ninguna noción de equidad individual.
