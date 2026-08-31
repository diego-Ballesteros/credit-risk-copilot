"""Explaining one decision: what the model used, and what it would say under a change.

Two modules, and the difference between them is the difference between two sentences an
analyst is allowed to write in a credit file.

`shap_service` answers *"which attributes of this applicant pushed this score, and in which
direction"*. Its whole subtlety is that the direction belongs to **this** row: the sign of a
feature's population mean is a different quantity and using it here would produce a
confident, well-formatted, wrong sentence.

`counterfactual` answers *"what would the model say about an applicant with these attributes
instead"*. That is a statement about the model, never about the world - section 4.3 of the
internal credit policy draws the same line, and both modules repeat it because the sentence
that crosses it reads perfectly well.

Neither module trains anything. Both read the pinned registry artefact through
`models.registry`, so the number they explain is the number the copilot reported.
"""
