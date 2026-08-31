---
document_id: politica-interna-credito
title: Política Interna de Otorgamiento de Crédito de Consumo Rotativo (DOCUMENTO SINTÉTICO)
issuer: Documento sintético del proyecto Credit Risk Copilot — no corresponde a ninguna entidad financiera real
citation_prefix: Política Interna de Crédito (documento sintético)
language: es
is_synthetic: true
synthetic_notice: DOCUMENTO SINTÉTICO. Redactado para el proyecto académico Credit Risk Copilot. No representa la política de ninguna entidad financiera real, no ha sido aprobado por ningún comité y no debe usarse como referencia para una decisión de crédito real.
retrieved_at: 2026-08-30
scope_note: Documento redactado íntegramente para este proyecto. Sus umbrales están anclados en las cifras medidas del modelo `credit-risk-default-probability` versión 1 y en el supuesto de costos 5:1 documentado en `docs/MODEL_CARD.md`. Las referencias que hace a la Circular Básica Contable y Financiera y a la Ley 1266 de 2008 remiten a los otros documentos del corpus, que sí son fuentes reales.
---

## 0. Aviso sobre la naturaleza de este documento

### 0.1. Este documento es sintético

**Este documento es sintético.** Fue redactado para el proyecto académico Credit Risk Copilot con el fin de dar al copiloto una política interna contra la cual contrastar un score. **No representa la política de crédito de ninguna entidad financiera real**, no fue aprobado por ningún órgano de gobierno y no debe usarse como referencia para una decisión de crédito real.

Se distingue de los otros tres documentos del corpus, que sí son textos reales y citables: la Circular Básica Contable y Financiera de la Superintendencia Financiera de Colombia, la Ley 1266 de 2008 y los Principios del Comité de Basilea. Cualquier fragmento recuperado de este documento debe presentarse al analista identificado como sintético.

### 0.2. Alcance

Esta política aplica al otorgamiento y a la revisión de cupos de **crédito de consumo rotativo a personas naturales**, en los términos del numeral 2.1.2 de la Circular Básica Contable y Financiera. No aplica a crédito comercial, de vivienda ni a microcrédito.

## 1. El insumo cuantitativo: el score de riesgo

### 1.1. Qué produce el modelo

El modelo interno `credit-risk-default-probability` devuelve una **probabilidad de incumplimiento (PD) en el intervalo [0, 1]** para el siguiente período de facturación. No devuelve una decisión ni una etiqueta: devuelve una probabilidad, y la decisión la fija esta política.

La probabilidad está calibrada, de modo que un grupo de solicitantes con PD estimada de 0,20 debe incumplir en una proporción cercana al 20%. La calibración es lo que permite convertir la PD en pérdida esperada; sin ella los rangos de esta sección no tendrían sentido económico.

### 1.2. El umbral operativo y el supuesto que lo sostiene

El **umbral operativo de la entidad es PD = 0,160**. Un solicitante con PD igual o superior a 0,160 se marca para rechazo.

Ese umbral **no es una propiedad del modelo**: se deriva de suponer que **un falso negativo cuesta cinco veces un falso positivo** (relación 5:1), es decir, que prestar a quien incumple cuesta cinco veces lo que cuesta rechazar a quien habría pagado. El supuesto no tiene respaldo empírico en los datos disponibles: fue declarado, no medido. Cambiarlo mueve el umbral, y con él una parte muy grande del libro:

| Relación de costos FN:FP | Umbral | Proporción del libro rechazada |
| --- | ---: | ---: |
| 3:1 | 0,220 | 25,9% |
| **5:1 (vigente)** | **0,160** | **38,8%** |
| 10:1 | 0,105 | 74,4% |

Cualquier propuesta de mover el umbral es una modificación de esta política y requiere aprobación del Comité de Crédito, no una decisión del analista.

### 1.3. Homologación a la escala regulatoria

Para el reporte a centrales de riesgo, la PD se homologa a las categorías de la tabla del numeral 2.2 literal a de la Circular Básica Contable y Financiera, columna de **Consumo**. El umbral de 0,160 (16%) cae dentro de la categoría **BB** de esa escala, cuyo rango para consumo es "> 5-28". La homologación es un requisito de reporte y **no sustituye** las bandas de decisión de la sección 2.

## 2. Bandas de decisión por score

### 2.1. Tabla de bandas

La decisión primaria se determina por la banda en la que cae la PD del solicitante.

| Banda | Rango de PD | Decisión primaria | Autoridad que decide |
| --- | --- | --- | --- |
| **A — Riesgo bajo** | PD < 0,060 | Aprobar con cupo y tasa estándar | Automática, con revisión por muestreo |
| **B — Riesgo moderado** | 0,060 ≤ PD < 0,120 | Aprobar con condiciones estándar | Analista de crédito |
| **C — Riesgo de vigilancia** | 0,120 ≤ PD < 0,160 | Aprobar con cupo reducido al 60% del solicitado y seguimiento mensual | Analista de crédito, con concepto escrito |
| **D — Rechazo con excepción posible** | 0,160 ≤ PD < 0,300 | Rechazar, salvo excepción documentada conforme a la sección 3 | Analista de crédito para el rechazo; Comité de Crédito para la excepción |
| **E — Rechazo firme** | PD ≥ 0,300 | Rechazar | Analista de crédito, sin facultad de excepción |

La banda D empieza exactamente en el umbral operativo de 0,160 de la sección 1.2. Las fronteras de las bandas A, B, C y E son decisiones de apetito de riesgo de la entidad y no se derivan del modelo.

### 2.2. La banda no es la decisión final

**Ninguna banda autoriza una decisión automática de rechazo.** El modelo ordena bastante mejor que el azar y no es un oráculo: en el umbral de 0,160, aproximadamente **6 de cada 10 solicitantes rechazados habrían pagado**. Toda decisión de rechazo en las bandas D y E requiere que un analista la revise y la firme antes de comunicarse al solicitante.

La aprobación automática de la banda A es la única excepción, y está sujeta a una revisión por muestreo de al menos el 5% de los casos cada mes.

### 2.3. Qué invalida el score

El score no se usa, y el caso pasa a evaluación manual completa, cuando ocurre cualquiera de estos supuestos:

- El solicitante no tiene historial de comportamiento de pago suficiente para las variables que el modelo usa.
- La solicitud llega con campos faltantes en las variables de comportamiento de pago. **Un dato faltante no se imputa con un cero**: un cero significa "no debe nada" y un faltante significa "no sabemos", y confundirlos convierte una ignorancia en un hecho de negocio falso.
- El solicitante figura en un proceso concursal, judicial o administrativo que pueda afectar su capacidad de pago, en los términos del numeral 2.2.3.3 de la Circular Básica Contable y Financiera.
- El solicitante fue objeto de una reestructuración en los últimos doce meses.

## 3. Excepciones

### 3.1. Dónde cabe una excepción

Solo cabe excepción en la **banda D** (0,160 ≤ PD < 0,300). Las bandas A, B y C no requieren excepción, y la banda E no la admite en ningún caso.

### 3.2. Criterios de excepción

Una excepción en banda D solo procede si se acredita **al menos uno** de los siguientes criterios, y ninguno de ellos puede sustituirse por la antigüedad de la relación comercial ni por la referencia de un funcionario:

1. **Capacidad de pago verificada y superior a la que refleja el score.** Ingresos formales verificados que dejan una relación cuota-ingreso menor al 25% después de atender la totalidad de las obligaciones vigentes reportadas en centrales de riesgo.
2. **Deterioro puntual con causa documentada.** La mora que empuja el score corresponde a un evento único y documentado, ya resuelto, y el comportamiento de los tres meses más recientes está al día.
3. **Garantía idónea.** Existe una garantía idónea en los términos del numeral 1.3.2.3.1 literal d de la Circular Básica Contable y Financiera, cuyo valor de realización cubre al menos el 120% del cupo solicitado. La garantía **no sustituye** la evaluación de la capacidad de pago: solo puede acompañarla.
4. **Reducción del cupo solicitado.** El solicitante acepta un cupo tal que la relación cuota-ingreso resultante lo lleva a las condiciones del criterio 1.

### 3.3. Límites de las excepciones

- Las excepciones aprobadas no pueden superar el **5% del número de solicitudes aprobadas en el trimestre**. Alcanzado ese tope, toda excepción adicional requiere aprobación del Comité de Riesgos, no del Comité de Crédito.
- Una excepción se documenta con el criterio invocado, la evidencia que lo acredita, la PD original y la identidad de quien aprueba. Una excepción sin criterio escrito es una aprobación irregular.
- Las excepciones se revisan trimestralmente contra el desempeño observado de la cohorte aprobada por excepción. Si esa cohorte incumple por encima de la banda D en su conjunto, los criterios de la sección 3.2 se revisan.

## 4. Requisitos de documentación

### 4.1. Documentación mínima del expediente

Todo expediente de solicitud, con independencia de la banda, contiene:

- Identificación del solicitante y verificación de identidad.
- Consulta vigente a centrales de riesgo, con fecha no anterior a 30 días calendario.
- Soporte de ingresos, según la naturaleza de la vinculación laboral o de la actividad económica del solicitante.
- Relación de obligaciones vigentes y del nivel de endeudamiento agregado.
- La información previa al otorgamiento entregada al solicitante conforme al numeral 1.3.2.3.1 literal a de la Circular Básica Contable y Financiera: tasa de interés y su equivalente efectivo anual, base de capital, tasa de mora, comisiones, plazo, condiciones de prepago y derechos de ambas partes.

### 4.2. Documentación específica de la decisión asistida por modelo

Cuando la decisión se apoya en el score, el expediente incorpora además:

- La **PD estimada**, con cuatro decimales.
- El **nombre y la versión del modelo** que la produjo, tal como figuran en el registro de modelos.
- El **umbral operativo vigente** al momento de la decisión.
- Las **cinco variables de mayor contribución** a la estimación, con su dirección y magnitud, tomadas de la explicación local del modelo.
- La **banda resultante** y la decisión primaria asociada.
- Si hubo excepción, el criterio invocado y su evidencia.

### 4.3. Lo que la explicación del modelo no autoriza a afirmar

La explicación local atribuye la **predicción del modelo**, no el efecto de cambiar una variable en el mundo. El expediente y la comunicación al solicitante pueden afirmar que *"si este cliente presentara una utilización de cupo del 30% en lugar del 85%, el modelo estimaría una probabilidad de 0,12 en lugar de 0,31"*. **No pueden afirmar** que reducir la utilización hará que el cliente deje de incumplir. La primera es una afirmación verificable sobre el modelo; la segunda es una afirmación causal que los datos observacionales no soportan.

## 5. Escalamiento a comité

### 5.1. Qué escala al Comité de Crédito

El Comité de Crédito conoce y decide:

- Toda **excepción en banda D**, conforme a la sección 3.
- Toda solicitud cuyo cupo exceda el límite de facultad individual del analista.
- Todo caso en el que el **concepto del analista contradiga la banda del modelo** en cualquier dirección: tanto una aprobación propuesta sobre un score de banda D o E, como un rechazo propuesto sobre un score de banda A o B. Una discrepancia entre el modelo y el analista es información, y se decide en comité, no en el escritorio.
- Todo caso en el que el score haya quedado invalidado por la sección 2.3 y el analista proponga aprobar.

### 5.2. Qué escala al Comité de Riesgos

El Comité de Riesgos conoce y decide:

- Cualquier propuesta de **mover el umbral operativo** de 0,160 o las fronteras de las bandas de la sección 2.1, incluida la revisión del supuesto de costos 5:1.
- Las excepciones que superen el tope trimestral del 5% previsto en la sección 3.3.
- La **puesta en producción de una nueva versión del modelo** y cualquier cambio en sus variables de entrada.
- Los resultados del monitoreo de equidad de la sección 6.

### 5.3. Quórum y registro

Toda decisión de comité deja constancia escrita con la fecha, los asistentes, el caso, la decisión y su motivación. La traza de aprobación debe permitir identificar quién aportó el análisis y quién tomó la decisión, en línea con el párrafo 43 de los Principios del Comité de Basilea para la gestión del riesgo de crédito.

## 6. Monitoreo obligatorio del modelo

### 6.1. Revisión periódica de la metodología

La metodología de otorgamiento y la relevancia de las variables se evalúan **como mínimo dos veces al año, al cierre de mayo y de noviembre**, conforme al numeral 1.3.2.3.1 literal b de la Circular Básica Contable y Financiera. La revisión verifica que las variables sigan discriminando y que la calibración no se haya desplazado.

### 6.2. Monitoreo de disparidad entre grupos

La entidad mide, en cada revisión semestral, la **tasa de rechazo y la tasa de falsos positivos por grupo demográfico** entre solicitantes que habrían pagado. La medición se reporta al Comité de Riesgos con el tamaño de cada grupo al lado de cada tasa.

Esta obligación no es preventiva: la disparidad **ya está medida** en la versión vigente del modelo. La razón de impacto dispar cae por debajo de 0,80 para nivel educativo y para edad, y entre solicitantes que habrían pagado la tasa de rechazo por error difiere hasta 10,3 puntos porcentuales según el nivel educativo. Está medido además que **retirarle al modelo las variables demográficas elimina solo entre el 9% y el 30% de esa brecha**: el resto viaja por variables de comportamiento correlacionadas, de modo que **no mirar el atributo protegido no vuelve equitativo al modelo**.

Operar el modelo es aceptar esas cifras de forma explícita, no ignorarlas. Cualquier comunicación interna o externa que presente el modelo como equitativo por no usar variables protegidas contradice esta política.

### 6.3. Límites conocidos que esta política asume

- El modelo **no cuenta con validación fuera de tiempo**, porque los datos con los que se construyó no tienen fecha de originación. Sus métricas describen la misma población y el mismo período, y no dicen nada sobre su degradación ante un cambio de ciclo económico.
- El modelo se construyó sobre una población y un período concretos, y no se afirma nada sobre su comportamiento fuera de ellos.
- La variable de mayor peso del modelo se apoya en códigos de estado de pago cuyo significado real no está documentado por la fuente de los datos.

## 7. Deberes frente al titular de la información

### 7.1. Uso de la información de centrales de riesgo

La consulta y el uso de la información financiera y crediticia del solicitante se sujetan a la Ley 1266 de 2008. En particular, la administración de esos datos obedece al **principio de finalidad**: la información se consulta para evaluar el riesgo de la relación contractual y no para ninguna otra finalidad.

### 7.2. Actualización de calificaciones y scores internos

Cuando un dato negativo se retire de las bases de datos, o cese el hecho que generó la disminución de la medición, **el score interno del solicitante se recalcula de manera simultánea**, conforme al parágrafo 3 del artículo 13 de la Ley 1266 de 2008. Un score que sigue castigando a un titular por un dato que ya fue retirado es un incumplimiento de esa norma.

### 7.3. Derecho del solicitante a conocer la calificación

El solicitante tiene derecho a conocer la calificación de riesgo de sus obligaciones con la entidad, conforme al numeral 1.3.2.3.1 literal a de la Circular Básica Contable y Financiera. La comunicación de un rechazo indica la banda resultante y los factores principales que la determinaron, en términos comprensibles, sin entregar el detalle interno del modelo.
