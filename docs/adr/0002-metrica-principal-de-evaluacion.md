# ADR 0002 — Métrica principal de evaluación

- **Status:** Accepted
- **Date:** 2026-08-24

---

## Contexto

El dataset tiene aproximadamente **22% de clase positiva**. Un clasificador que
ignore todas las variables y prediga siempre "no incumple" obtiene cerca del **78% de
accuracy y cero valor de negocio**: no identifica ni un solo caso de los que importan.

Además, los dos tipos de error tienen **costos asimétricos**:

| Error | Costo de negocio |
| --- | --- |
| Falso negativo | Pérdida del principal |
| Falso positivo | Ingreso no percibido |

El falso negativo es típicamente el más caro.

Finalmente, el uso de negocio **requiere una probabilidad y no solo una etiqueta**:
sin probabilidad calibrada no se puede calcular pérdida esperada ni fijar tasa.

## Decisión

La métrica principal de decisión del proyecto es **PR-AUC** (área bajo la curva
precisión-recall).

Se reportan además, como **métricas de contexto obligatorias**: ROC-AUC, KS, Gini,
Brier score y precision@k.

**Toda métrica se reporta junto a su baseline.**

### Por qué se reportan KS y Gini

No son la métrica de decisión, pero son el **estándar de facto en la industria
financiera de riesgo de crédito**. Omitirlas haría el reporte ilegible para cualquier
lector del dominio.

### Por qué se reporta Brier score

Mide la **calidad de la calibración**, no la capacidad de ordenamiento. Un modelo
puede tener PR-AUC excelente y probabilidades mal calibradas: ordena bien, pero sus
números no son interpretables como probabilidades.

Sin calibración no hay pérdida esperada, y sin pérdida esperada el modelo no sirve
para el uso de negocio declarado.

## Alternativas consideradas

**Accuracy.** Descartada. Con este desbalance, un modelo trivial la maximiza sin
aportar nada. Es una métrica que no puede fallar, y una métrica que no puede fallar no
prueba nada.

**ROC-AUC como métrica principal.** Descartada como métrica de decisión, aunque se
reporta. El mecanismo: el eje X de la curva ROC es la tasa de falsos positivos, cuyo
denominador son **todos los negativos reales**. Cuando la clase negativa es
mayoritaria, ese denominador es muy grande y absorbe los falsos positivos sin que la
curva se mueva. La curva PR usa precisión, cuyo denominador son **solo las
predicciones positivas**, por lo que sí refleja el costo de equivocarse sobre la clase
minoritaria. En un problema donde lo que importa es la clase escasa, ROC-AUC produce
una lectura sistemáticamente optimista.

**F1-score.** Descartado como métrica principal porque **exige fijar un umbral antes
de medir**, y la elección de umbral es una decisión de negocio posterior que debe
tomarse con una matriz de costos explícita. PR-AUC evalúa el modelo en todos los
umbrales a la vez.

## Consecuencias

### Positivas

Todas las comparaciones de modelos de las fases siguientes tienen un **criterio único
y decidido de antemano**, lo que elimina la tentación de elegir a posteriori la
métrica que favorece al modelo preferido.

### Negativas

PR-AUC es **menos conocida que ROC-AUC** y exige explicación en el README y en la
defensa del proyecto.

### Riesgo

Si durante el modelado el **desbalance resultara mucho menor de lo esperado** tras el
preprocesamiento, la ventaja de PR-AUC sobre ROC-AUC se reduciría y habría que revisar
esta decisión mediante un ADR que **supersede** a este.
