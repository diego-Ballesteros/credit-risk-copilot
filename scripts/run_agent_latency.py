"""Measure what an analyst waits for, and what it costs, on a few already-annotated queries.

Run it with::

    uv run uvicorn credit_copilot.api.agent_app:app --port 8001    # in another terminal
    uv run python scripts/run_agent_latency.py
    uv run python scripts/run_agent_latency.py --n 5 --api-url http://127.0.0.1:8001

**What this measures and what it deliberately does not.** Time and money, and nothing else.
The quality of the answers was measured in entry 012 of `docs/EVALUATION.md` with an
annotator, three arms and 57 runs; **re-judging here would spend the judge's tokens to
re-derive a number that already exists**, and a second measurement over five queries would
be noisier than the one it duplicated. So no judge is called, no claim is scored, and no
number in this script says anything about whether the copilot was right.

**Why it goes over HTTP against `/chat` and does not import the graph.** The runner is
reused, not duplicated - it is the same `agent.graph.run_query` behind the agent service - and
going through the API is what makes this the end-to-end latency: the request contract, the
applicant's twenty-three validations, the graph, and the serialisation of a response that
carries every tool call and every citation. Importing the graph would measure the graph, which
is the part nobody waits on alone.

**The queries come from the annotated set and are not written here.** `data/eval/agent_queries.yaml`
holds the nineteen queries entry 012 used. Taking the first `--n` of them in file order keeps
this run comparable with that entry and keeps a latency measurement from quietly becoming a
new, unannotated evaluation set.

Exit code 0 when the run completed, 1 when the API or MLflow could not be reached.
"""

import argparse
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import mlflow
import numpy as np
import yaml

from credit_copilot.config import settings
from credit_copilot.console import enable_unicode_console
from credit_copilot.data import schema
from credit_copilot.data.loader import load_raw_dataframe
from credit_copilot.models.registry import PREDICTOR_COLUMNS
from credit_copilot.models.tracking import MLflowConfigurationError, ensure_experiment

EXPERIMENT_NAME: Final[str] = "credit-risk-online"
"""Same experiment as the model's online run: both measure the deployed system."""

QUERIES_FILE: Final[str] = "agent_queries.yaml"
DEFAULT_API_URL: Final[str] = "http://127.0.0.1:8001"
DEFAULT_QUERIES: Final[int] = 5
"""How many queries to send by default.

Small on purpose. At the cost measured in entry 012 - 0.209 USD per query - nineteen queries
is about four dollars to re-measure a latency whose spread is already visible in five. The
sample is too small to estimate a percentile, and the report says so rather than printing a
p99 over five points as if it meant something.
"""

REQUEST_TIMEOUT_SECONDS: Final[float] = 300.0
"""A copilot query is up to seven language-model calls; a client that gave up sooner would
record a timeout that is its own and not the system's."""

READY_TIMEOUT_SECONDS: Final[float] = 900.0
"""The copilot's build downloads the registry artefact, an embedding model and opens the
index, so its start-up is the slower of the two services."""

# Published prices in USD per million tokens, for the two models the graph uses.
# Restated here rather than measured, and the report says so: a cost figure whose price table
# is invisible is a number nobody can check or update.
PRICES_USD_PER_MTOK: Final[Mapping[str, tuple[float, float]]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
}

RULE: Final[str] = "=" * 96


@dataclass(frozen=True)
class QueryLatency:
    """What one query cost in time and money.

    Attributes:
        query_id: Identifier from the annotated set.
        query: The question sent.
        seconds: Wall clock from request to parsed response.
        llm_calls: Calls the graph made to a language model.
        iterations: Planner cycles used.
        tools: Tool calls made, refusals included.
        citations: Distinct sources the answer rests on.
        answer_chars: Length of the answer.
        usd: Estimated cost from the reported token usage.
        input_tokens: Prompt tokens across every call.
        output_tokens: Generated tokens across every call.
        outcome: How the run ended.
        error: Why it failed, when it did.
    """

    query_id: str
    query: str
    seconds: float
    llm_calls: int
    iterations: int
    tools: int
    citations: int
    answer_chars: int
    usd: float
    input_tokens: int
    output_tokens: int
    outcome: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Read where to send the queries and how many.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the agent API.")
    parser.add_argument(
        "--n", type=int, default=DEFAULT_QUERIES, help="Queries to send, in file order."
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Skip logging. For a local dry run only."
    )
    return parser.parse_args()


def load_queries(path: Path, limit: int) -> list[Mapping[str, Any]]:
    """Read the first `limit` queries of the annotated set, in file order.

    Only the fields a latency run uses are read - the identifier, the text and the applicant
    row. The annotation fields are deliberately ignored: nothing here scores anything, and
    reading them would invite a future edit that started to.

    Args:
        path: The YAML file.
        limit: How many to take.

    Returns:
        One mapping per query.

    Raises:
        RuntimeError: The file is missing or declares no queries.
    """
    if not path.exists():
        raise RuntimeError(f"No existe {path}.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = payload.get("queries")
    if not entries:
        raise RuntimeError(f"{path} no declara ninguna consulta bajo `queries`.")
    return [
        {
            "id": str(entry["id"]),
            "query": str(entry["query"]),
            "applicant_row": entry.get("applicant_row"),
        }
        for entry in entries[:limit]
    ]


def load_applicant(client_id: int) -> dict[str, int]:
    """Read one client's raw attributes from the downloaded dataset.

    Args:
        client_id: Value of the source `ID` column.

    Returns:
        The 23 raw canonical columns of that client.

    Raises:
        LookupError: No row carries that identifier.
    """
    frame = load_raw_dataframe().rename(columns=dict(schema.RAW_TO_CANONICAL))
    matching = frame.loc[frame[schema.ID_COLUMN] == client_id]
    if matching.empty:
        raise LookupError(f"No hay ninguna fila con {schema.ID_COLUMN}={client_id}.")
    row = matching.iloc[0]
    return {column: int(row[column]) for column in PREDICTOR_COLUMNS}


def estimate_cost(usages: Sequence[Mapping[str, Any]]) -> tuple[float, int, int]:
    """Price one run's token usage.

    Per call and not on a total, because the graph uses two models at different prices and a
    single total would make the cost depend on a mix nobody could reconstruct. A model the
    price table does not know contributes zero and is named in the report rather than priced
    at a guess.

    Args:
        usages: The `token_usage` entries of a `/chat` response.

    Returns:
        The estimated USD, the input tokens and the output tokens.
    """
    usd = 0.0
    input_tokens = 0
    output_tokens = 0
    for usage in usages:
        model = str(usage.get("model", ""))
        prompt = int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_tokens", 0))
        completion = int(usage.get("output_tokens", 0))
        input_tokens += prompt
        output_tokens += completion
        price = PRICES_USD_PER_MTOK.get(model)
        if price is not None:
            usd += prompt / 1e6 * price[0] + completion / 1e6 * price[1]
    return usd, input_tokens, output_tokens


def wait_until_ready(client: httpx.Client, api_url: str) -> None:
    """Block until the copilot reports it can answer.

    Args:
        client: An open HTTP client.
        api_url: Base URL.

    Raises:
        RuntimeError: The service did not become ready in time.
    """
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last = "no response yet"
    while time.monotonic() < deadline:
        try:
            response = client.get(f"{api_url}/ready", timeout=10.0)
        except httpx.HTTPError as error:
            last = f"{type(error).__name__}: {error}"
        else:
            if response.status_code == 200:
                return
            last = f"HTTP {response.status_code}: {response.text[:200]}"
        time.sleep(2.0)
    raise RuntimeError(f"{api_url}/ready never reported ready. Last answer: {last}")


def send_query(client: httpx.Client, api_url: str, entry: Mapping[str, Any]) -> QueryLatency:
    """Send one query to `/chat` and time it end to end.

    Args:
        client: An open HTTP client.
        api_url: Base URL.
        entry: The query, its identifier and its applicant row.

    Returns:
        What the query cost in time and money. A failure is recorded, not raised.
    """
    body: dict[str, Any] = {"query": entry["query"]}
    if entry["applicant_row"] is not None:
        body["applicant"] = load_applicant(int(entry["applicant_row"]))

    started = time.perf_counter()
    try:
        response = client.post(f"{api_url}/chat", json=body, timeout=REQUEST_TIMEOUT_SECONDS)
        seconds = time.perf_counter() - started
    except httpx.HTTPError as error:
        return QueryLatency(
            query_id=str(entry["id"]),
            query=str(entry["query"]),
            seconds=time.perf_counter() - started,
            llm_calls=0,
            iterations=0,
            tools=0,
            citations=0,
            answer_chars=0,
            usd=0.0,
            input_tokens=0,
            output_tokens=0,
            outcome="transport_error",
            error=f"{type(error).__name__}: {error}",
        )

    if response.status_code != 200:
        return QueryLatency(
            query_id=str(entry["id"]),
            query=str(entry["query"]),
            seconds=seconds,
            llm_calls=0,
            iterations=0,
            tools=0,
            citations=0,
            answer_chars=0,
            usd=0.0,
            input_tokens=0,
            output_tokens=0,
            outcome=f"http_{response.status_code}",
            error=response.text[:300],
        )

    payload = response.json()
    usd, input_tokens, output_tokens = estimate_cost(payload.get("token_usage", []))
    return QueryLatency(
        query_id=str(entry["id"]),
        query=str(entry["query"]),
        seconds=seconds,
        llm_calls=int(payload.get("llm_calls", 0)),
        iterations=int(payload.get("iterations", 0)),
        tools=len(payload.get("tools_invoked", [])),
        citations=len(payload.get("citations", [])),
        answer_chars=len(str(payload.get("answer", ""))),
        usd=usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        outcome=str(payload.get("outcome", "")),
    )


def print_report(results: Sequence[QueryLatency], api_url: str, elapsed: float) -> None:
    """Print the per-query table and the summary, with the sample size next to every figure."""
    ok = [r for r in results if r.error is None]

    print("\n" + RULE)
    print("AGENT LATENCY AND COST")
    print(RULE)
    print(f"API      : {api_url}")
    print(f"Queries  : {len(results)} from data/eval/{QUERIES_FILE}, in file order")
    print("Measured : time and cost only. No judge was called; quality is entry 012.")

    print("\n" + "-" * 96)
    print(
        f"{'query':<8}{'s':>9}{'calls':>7}{'cyc':>5}{'tools':>7}{'cites':>7}"
        f"{'in tok':>9}{'out tok':>9}{'USD':>9}  outcome"
    )
    print("-" * 96)
    for r in results:
        if r.error is not None:
            print(
                f"{r.query_id:<8}{r.seconds:>9.2f}{'':>7}{'':>5}{'':>7}{'':>7}"
                f"{'':>9}{'':>9}{'':>9}  {r.outcome}: {r.error[:40]}"
            )
            continue
        print(
            f"{r.query_id:<8}{r.seconds:>9.2f}{r.llm_calls:>7}{r.iterations:>5}{r.tools:>7}"
            f"{r.citations:>7}{r.input_tokens:>9,}{r.output_tokens:>9,}{r.usd:>9.4f}"
            f"  {r.outcome}"
        )

    if not ok:
        print("\nNo query completed, so there is nothing to summarise.")
        return

    seconds = sorted(r.seconds for r in ok)
    print("\n" + "-" * 96)
    print("SUMMARY")
    print("-" * 96)
    print(f"  Completed        : {len(ok)} of {len(results)}")
    print(f"  Wall clock       : {elapsed:.1f} s for the whole run, sent one at a time")
    print(f"  Latency  min     : {seconds[0]:.2f} s")
    print(f"           median  : {statistics.median(seconds):.2f} s")
    print(f"           max     : {seconds[-1]:.2f} s")
    print(f"  Cost     total   : {sum(r.usd for r in ok):.4f} USD")
    print(f"           mean    : {statistics.fmean([r.usd for r in ok]):.4f} USD per query")
    print(
        f"  LLM calls mean   : {statistics.fmean([r.llm_calls for r in ok]):.2f} "
        f"(ceiling is 2*max_iterations + 1)"
    )
    print(f"  Citations mean   : {statistics.fmean([r.citations for r in ok]):.2f}")
    print(
        f"\n  n={len(ok)}: too small for a percentile. The min, the median and the max are\n"
        "  reported instead, because a p99 over five points is the maximum wearing a\n"
        "  statistic's name. Cost is an ESTIMATE from a price table restated in this script."
    )
    print(RULE)


def log_to_mlflow(results: Sequence[QueryLatency], api_url: str, elapsed: float) -> str:
    """Record the run.

    Args:
        results: Every query's outcome.
        api_url: Where they were sent.
        elapsed: Wall clock of the whole run.

    Returns:
        The run identifier.

    Raises:
        MLflowConfigurationError: The tracking server is not configured.
    """
    context = ensure_experiment(EXPERIMENT_NAME)
    ok = [r for r in results if r.error is None]
    with mlflow.start_run(experiment_id=context.experiment_id, run_name="agent-latency") as run:
        mlflow.set_tags(
            {
                "run_type": "agent-latency",
                "phase": "04-production",
                "transport": "http",
                "api_url": api_url,
                "judged": "false",
                "quality_reference": "EVALUATION.md entry 012",
            }
        )
        mlflow.log_params(
            {
                "n_queries": len(results),
                "queries_file": QUERIES_FILE,
                "random_state": settings.random_state,
                "concurrency": 1,
            }
        )
        metrics: dict[str, float] = {
            "queries_completed": float(len(ok)),
            "wall_clock_s": elapsed,
        }
        if ok:
            seconds = [r.seconds for r in ok]
            metrics.update(
                {
                    "latency_min_s": float(np.min(seconds)),
                    "latency_median_s": float(statistics.median(seconds)),
                    "latency_max_s": float(np.max(seconds)),
                    "latency_mean_s": float(statistics.fmean(seconds)),
                    "usd_total": float(sum(r.usd for r in ok)),
                    "usd_mean": float(statistics.fmean([r.usd for r in ok])),
                    "llm_calls_mean": float(statistics.fmean([r.llm_calls for r in ok])),
                    "citations_mean": float(statistics.fmean([r.citations for r in ok])),
                    "input_tokens_total": float(sum(r.input_tokens for r in ok)),
                    "output_tokens_total": float(sum(r.output_tokens for r in ok)),
                }
            )
        mlflow.log_metrics(metrics)
        rows = "query_id,seconds,llm_calls,iterations,tools,citations,usd,outcome\n" + "\n".join(
            f"{r.query_id},{r.seconds:.4f},{r.llm_calls},{r.iterations},{r.tools},"
            f"{r.citations},{r.usd:.6f},{r.outcome}"
            for r in results
        )
        mlflow.log_text(rows, "agent_latency.csv")
        return str(run.info.run_id)


def main() -> int:
    """Send the queries, time them, and record what they cost.

    Returns:
        0 on success, 1 if the API or MLflow could not be reached.
    """
    enable_unicode_console()
    args = parse_args()
    try:
        entries = load_queries(settings.eval_dir / QUERIES_FILE, args.n)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    results: list[QueryLatency] = []
    with httpx.Client() as client:
        try:
            wait_until_ready(client, args.api_url)
        except RuntimeError as error:
            print(f"The copilot never became ready:\n{error}", file=sys.stderr)
            return 1
        started = time.perf_counter()
        # One at a time, deliberately: this measures what one analyst waits for. Running them
        # concurrently would measure the throughput of a service nobody is using that way,
        # and would inflate every latency with contention the analyst never sees.
        for entry in entries:
            print(f"  -> {entry['id']}: {entry['query'][:70]}...", flush=True)
            results.append(send_query(client, args.api_url, entry))
        elapsed = time.perf_counter() - started

    print_report(results, args.api_url, elapsed)

    if args.no_mlflow:
        print("\nNot logged: --no-mlflow was given.")
        return 0
    try:
        run_id = log_to_mlflow(results, args.api_url, elapsed)
    except MLflowConfigurationError as error:
        print(f"\nMLflow is not configured:\n{error}", file=sys.stderr)
        return 1
    print(f"\nLogged to MLflow: experiment {EXPERIMENT_NAME}, run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
