"""The HTTP surface: two applications that ship as two services.

**Why two and not one.** The model application answers with a probability. To do that it
needs `scikit-learn`, `mlflow` and `shap`. The agent application answers with prose, tool
records and citations, and to do that it needs `anthropic`, `langgraph`, `chromadb` and
`sentence-transformers`, which drags `torch` - about two gigabytes of dependency whose only
job is to embed a query. Serving one application from both stacks would make a request for
`P(default)` pay for a transformer it never calls, and would make a change to the embedding
model a reason to redeploy the scorer.

So the two applications are separate objects, in separate modules, with separate lifecycles
and separate health endpoints, and nothing in `model_app` imports anything from `agent/`.
`tests/test_api.py` asserts that in a fresh interpreter rather than trusting it, because the
guarantee is one that an innocent-looking import restores in silence.

**What they share, and why it is only this.** Both import `schemas` for the applicant
contract and the response shapes, and `dependencies` for logging, correlation identifiers,
error handling and artefact lifecycle. Both of those modules are deliberately free of any
agent import at module level: `dependencies.load_agent_service` reaches the graph through a
function-local import, which is what lets `model_app` import the same module without pulling
the agent stack behind it.

**What this package does not do.** It does not build images and it does not measure latency;
both are the next turn's work. It also imports nothing at package level, so importing
`credit_copilot.api` costs nothing and neither application is loaded by reaching for the
other.

Run them with::

    uv run uvicorn credit_copilot.api.model_app:app --port 8000
    uv run uvicorn credit_copilot.api.agent_app:app --port 8001
"""
