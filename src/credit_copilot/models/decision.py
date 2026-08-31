"""The operating point, and the sentences without which a probability is not readable.

**Why this module exists, and why it is not inside `agent/`.** These constants were defined
in `agent/tools.py`, which was the only consumer while the copilot was the only consumer.
The production deployment splits the system into two services - one that serves the model
and one that serves the agent - and the model service must not carry `langgraph`,
`chromadb`, `anthropic` or the embedding stack in order to return a probability. Importing
`agent/tools.py` to reach a threshold would drag all four. The alternative was to restate the
text in the API layer, which `docs/MODEL_CARD.md` section 11.4 ter already measured the cost
of: a hand-maintained copy of normative text diverges from its source in silence. So the
definitions moved down to where both consumers can reach them, and `agent/tools.py` imports
and re-exports every name it used to define, unchanged.

**Why the threshold is not restated here either.** `OPERATING_THRESHOLD` is an alias of
`models.estimators.PRODUCTION_OPERATING_THRESHOLD`, which is where the number the phase-2
scripts measured with lives. One control point, two names, and the second one exists only so
that a reader of a decision does not have to know it came from an estimator module.

**Why the caveats are constants and not prose written at each call site.** A probability of
0,19 read without the threshold is a number; read with the threshold but without the cost
assumption behind it, it is a verdict that looks like a measurement. Both sentences travel in
every response that carries a probability, which is a contract the API and the tools honour
identically because they read the same string.

**What this does not fix.** These strings transcribe sections 2.2 and 4.3 of the internal
credit policy by hand, and nothing checks them against the corpus. `docs/adr/0009` records
that as an open decision; moving the definition does not close it.
"""

from typing import Final, Literal

from credit_copilot.models.estimators import PRODUCTION_OPERATING_THRESHOLD

__all__ = [
    "CAUSAL_NOTE",
    "COST_ASSUMPTION",
    "COST_RATIO_FN_TO_FP",
    "DECISION_CAVEAT",
    "DIRECTION_NOTE",
    "OPERATING_THRESHOLD",
    "Decision",
    "decide",
]

OPERATING_THRESHOLD: Final[float] = PRODUCTION_OPERATING_THRESHOLD
"""The threshold the copilot decides at. Imported, never restated: one control point."""

COST_RATIO_FN_TO_FP: Final[int] = 5
"""Cost ratio the operating threshold is derived from: one false negative costs five false
positives. **Declared, not measured** - `docs/MODEL_CARD.md` section 5 records that the
dataset carries no exposure, recovery or margin data with which to estimate it, and that
moving the ratio between 3:1 and 10:1 moves 48.5% of the book."""

Decision = Literal["approve", "refuse"]
"""What the operating threshold recommends. A recommendation, never an authorisation."""

DIRECTION_NOTE: Final[str] = (
    "La dirección de cada variable es el signo de su valor SHAP PARA ESTE SOLICITANTE. "
    "No es el signo de la media poblacional de esa variable: una variable cuya media "
    "poblacional es negativa puede subir el score de este solicitante, y leer el signo "
    "poblacional produciría una frase fluida y falsa. Los valores SHAP suman el score del "
    "bosque sin calibrar, no la probabilidad calibrada sobre la que se decide."
)
"""Read by the language model with every explanation. See `explain/shap_service.py`."""

CAUSAL_NOTE: Final[str] = (
    "Esto es una afirmación sobre el MODELO, no sobre el mundo. Dice cómo evaluaría el "
    "modelo a un solicitante con esos atributos; NO dice qué pasaría si este cliente los "
    "cambiara. La primera es verificable reejecutando la herramienta; la segunda es una "
    "afirmación causal que datos observacionales no soportan. Sección 4.3 de la política "
    "interna y sección 8 de docs/MODEL_CARD.md."
)
"""Read by the language model with every simulation."""

_THRESHOLD_ES: Final[str] = f"{OPERATING_THRESHOLD:.3f}".replace(".", ",")
"""The operating threshold written the way the corpus and the analyst write it.

Spanish uses a comma as the decimal separator, and a sentence that mixes `0.160` with
`0,220` reads as two different quantities. Derived from the constant rather than typed out,
so the text cannot drift from the number the decision actually uses.
"""

COST_ASSUMPTION: Final[str] = (
    f"El umbral {_THRESHOLD_ES} NO es una propiedad del modelo: se deriva de "
    f"suponer que un falso negativo cuesta {COST_RATIO_FN_TO_FP} veces un falso positivo "
    f"({COST_RATIO_FN_TO_FP}:1). Ese supuesto fue declarado, no medido: el dataset no tiene "
    "datos de exposición, recuperación ni margen. Con 3:1 el umbral sería 0,220 y con 10:1, "
    "0,105; mover el cociente entre esos dos extremos desplaza al 48,5% del libro."
)
"""Travels with every score, because the number alone is not interpretable."""

DECISION_CAVEAT: Final[str] = (
    f"Ninguna banda autoriza un rechazo automático. En el umbral {_THRESHOLD_ES}, "
    "aproximadamente 6 de "
    "cada 10 solicitantes rechazados habrían pagado. Toda decisión de rechazo en bandas D y "
    "E requiere que un analista la revise y la firme (sección 2.2 de la política interna)."
)
"""Travels with every score, because a probability reads like a verdict and is not one."""


def decide(probability: float) -> Decision:
    """Apply the operating threshold. A recommendation, never an authorisation.

    Args:
        probability: A calibrated probability of default, in [0, 1].

    Returns:
        `refuse` at or above the threshold, `approve` below it. The comparison is
        inclusive at the bound because band D of the internal credit policy opens at
        exactly the threshold.
    """
    return "refuse" if probability >= OPERATING_THRESHOLD else "approve"
