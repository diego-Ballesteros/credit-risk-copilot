"""Train the production model on the full dataset and register it. The canonical path.

Run it with::

    uv run python scripts/run_training.py

===========================================================================
WHAT THIS SCRIPT IS: A NAME, NOT A SECOND IMPLEMENTATION
===========================================================================

This script **delegates entirely** to `scripts/register_production_model.py` and adds no
arithmetic of its own. It exists because the course brief asks for execution scripts named
for preprocessing, *training* and prediction, and a reader holding that list does not find
"training" under a file called `register_production_model.py`.

**Delegation and not a copy, deliberately.** Two files that both fit a pipeline are two
files that can disagree about how it is fitted, and section 6.3 of `docs/METHODOLOGY.md`
identifies that divergence as the most treacherous failure mode in this project: everything
green in one path, different numbers in the other, no error anywhere. There is exactly one
implementation, and this module imports it.

===========================================================================
WHAT IT TRAINS, AND WHERE EVERY NUMBER COMES FROM
===========================================================================

The artefact is one `Pipeline` that goes from the 23 raw columns to a probability:

    preprocess  23 raw columns -> 110 columns
      behaviour   22 derived features             ADR-0005
      education   collapse of undocumented codes  ADR-0004
      clip        ratio cap at percentile 99.5, learned in `fit`
      columns     one-hot / impute+robust-scale / passthrough
    model       RandomForestClassifier + sigmoid calibration

Every hyper-parameter is read from `models.estimators`, never restated here:

    n_estimators=300, max_depth=10, min_samples_leaf=18, max_features=0.3
    class_weight=None, sigmoid calibration (cv=3, ensemble=False)
    random_state=42, taken from `config.py`

**Where they come from, and how much they are worth.** Entry 006 of `docs/EVALUATION.md`
searched them with Optuna under nested cross-validation and measured the gain at **+0.0028
PR-AUC against a between-fold standard deviation of 0.0080** - inside the noise, below the
0.02 practical-significance threshold. They are adopted anyway, and ADR-0007 decision 2
records why: adopting them costs nothing, and `max_features=0.3` is the one dimension the
data did constrain, chosen by all five outer folds and by the final study. `class_weight`
is `None` because entry 005 measured that re-weighting buys no ranking and costs 0.0404 of
Brier. The sigmoid calibration is a **conservative decision with no measured gain**
(ADR-0007 decision 1), kept because its cost is bounded at +0.0008 Brier and exactly 0.0000
PR-AUC.

===========================================================================
ITS RELATIONSHIP TO VERSION 1, STATED PRECISELY
===========================================================================

**It fits the same artefact. It does not overwrite version 1.** A registry `log_model` call
always appends, so running this script produces version 2, then 3, and so on. Version 1 -
the version `docs/MODEL_CARD.md` describes, the version ADR-0010 pins into both containers,
and the version every figure in this repository was measured against - **stays exactly where
it is**. That pinning is why running this script cannot move what the API serves.

**How reproducible the fit is - measured against version 1, not asserted.** Running this
script produced version 2, and the two versions were then compared component by component
over 500 applicants:

    preprocessed matrix, 110 columns      0.000e+00   identical
    forest: trees / total nodes           300 = 300 / 151,734 = 151,734
    forest feature importances            0.000e+00   identical
    forest scores, before calibration     1.110e-16   thread-summation noise
    sigmoid calibrator a_, b_             differ in the 8th significant digit
    final calibrated probability          2.125e-09 max, 7.207e-10 mean
    decisions that differ at 0.160        0 of 500

**The forest is reproducible; the calibrator is not, and the mechanism is worth stating.**
scikit-learn seeds each tree deterministically from `random_state`, so a third party fits
the same 300 trees down to the node - the identical node count and identical importances are
what establish that. The sigmoid, however, is fitted by an iterative optimiser over the
out-of-fold forest scores, and those scores carry the 1e-16 threaded-summation noise that
section 7.4 of `docs/MODEL_CARD.md` documents. The optimiser therefore starts from marginally
different data and stops at a marginally different point, and its convergence tolerance turns
1e-16 in the input into ~3e-8 in the two parameters and **2e-9 in the probability**.

**So the honest claim is "the same model", not "the same bits".** Two point one nanounits is
**seventy-five million times** smaller than the 0.160 operating threshold, and none of the
500 decisions moved. But a test asserting bit equality between two fits would fail, and
anyone who writes one should know why before deleting it.

Note that this is a **larger** effect than the one section 7.4 of the Model Card records.
That section measures the same object called twice; this measures two independent fits, where
the noise has passed through an optimiser before reaching the output.

Exit code 0 when the model was registered, 1 when the tracking server could not be
configured, 2 when the registry refused the model. They are the delegated script's codes,
unchanged, so a caller can treat the two entry points as one.
"""

import sys
from pathlib import Path

# The sibling script is a module in this directory, not an installed package: running
# `python scripts/run_training.py` puts `scripts/` at `sys.path[0]`, but importing this file
# any other way would not. Deriving the path from `__file__` makes the import work regardless
# of the working directory, which is the same reason `config.PROJECT_ROOT` is derived from a
# module location rather than from `cwd`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from register_production_model import main  # noqa: E402  - must follow the path insert

if __name__ == "__main__":
    raise SystemExit(main())
