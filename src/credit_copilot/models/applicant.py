"""The applicant, as a contract: the 23 raw attributes the production model reads.

**Why this module exists, and why it is not inside `agent/`.** `ApplicantRecord` was defined
in `agent/tools.py`, which was its only consumer while the copilot was the only consumer.
The production deployment splits the system into two services, and the one that serves the
model must be able to validate an applicant without importing `langgraph`, `chromadb`,
`anthropic` or the embedding stack. The record moved down to where both services reach it;
`agent/tools.py` imports and re-exports it unchanged, so every existing import still works
and the contract stays single.

**Why there is exactly one of these and the API does not write its own.** Two Pydantic models
describing the same 23 columns is two places to change when the data contract changes, and
the second one is always the one nobody remembers. The API layer subclasses this record to
tighten it - see `api/schemas.py` - and never restates the fields.
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from credit_copilot.models.registry import PREDICTOR_COLUMNS, require_known_values

__all__ = ["ApplicantRecord"]


class ApplicantRecord(BaseModel):
    """The 23 raw attributes the production model reads, all of them required.

    **Why every field is required and none has a default.** A default is an imputation with
    better manners. `PAY_AMT3 = 0` means *"paid nothing in July"*, which is a business fact;
    an absent `PAY_AMT3` means *"we do not know"*, which is not. Section 2.3 of the internal
    credit policy sends the second case to full manual evaluation rather than to the model,
    and the only way to make that reachable is for the record to refuse to be built.

    **Why the names are the source's names.** `PAY_STATUS_*` is the project's canonical
    renaming of the source's `PAY_0, PAY_2..PAY_6`; everything else is verbatim what UCI
    ships. `tests/test_tools.py` asserts this field set equals `PREDICTOR_COLUMNS`, so the
    contract cannot drift from `schema.py` without a test failing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    LIMIT_BAL: int
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_STATUS_1: int
    PAY_STATUS_2: int
    PAY_STATUS_3: int
    PAY_STATUS_4: int
    PAY_STATUS_5: int
    PAY_STATUS_6: int
    BILL_AMT1: int
    BILL_AMT2: int
    BILL_AMT3: int
    BILL_AMT4: int
    BILL_AMT5: int
    BILL_AMT6: int
    PAY_AMT1: int
    PAY_AMT2: int
    PAY_AMT3: int
    PAY_AMT4: int
    PAY_AMT5: int
    PAY_AMT6: int

    @model_validator(mode="after")
    def _values_must_be_known(self) -> "ApplicantRecord":
        """Refuse categories and magnitudes the data contract does not recognise."""
        require_known_values(self.model_dump())
        return self

    def to_frame(self) -> pd.DataFrame:
        """Render the applicant as the single-row table the pipeline expects.

        Returns:
            One row, columns in `PREDICTOR_COLUMNS` order, integer dtype like the source.
        """
        values = self.model_dump()
        return pd.DataFrame(
            [[values[column] for column in PREDICTOR_COLUMNS]], columns=list(PREDICTOR_COLUMNS)
        ).astype("int64")
