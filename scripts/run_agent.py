"""Run one query through the copilot and print what it did, not only what it said.

Run it with::

    uv run python scripts/run_agent.py "¿A partir de qué número se rechaza una solicitud?"
    uv run python scripts/run_agent.py --applicant-row 42 "¿Apruebo a este solicitante?"
    uv run python scripts/run_agent.py --applicant-file cliente.json "¿Por qué sale así?"

**Why it prints the tool calls and the citations and not just the answer.** An answer is the
part that reads well; the calls and the citations are the part that can be checked. A copilot
whose output cannot be audited is a copilot nobody can be asked to trust with a credit file,
and section 4.2 of the internal credit policy requires the model, its version, the threshold
and the top five variables to be recorded - all of which are in the tool records rather than
in the prose.

**Why an applicant is loaded from the real dataset rather than typed in.** `--applicant-row`
takes the row with that `ID` from `data/raw/`, so the 23 attributes are a real client's and
not twenty-three numbers somebody invented to make the demonstration work. `--applicant-file`
takes a JSON object with the same 23 canonical columns, which is the shape an API request
would carry.

Exit code 0 when the query was answered, 1 when the copilot could not be configured or its
dependencies could not be loaded, 2 when the query itself failed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import anthropic

from credit_copilot.agent.graph import CopilotConfigurationError, run_query
from credit_copilot.agent.state import DEFAULT_MAX_ITERATIONS, AgentState, unique_citations
from credit_copilot.agent.tools import ToolExecutionError, build_tool_context
from credit_copilot.console import enable_unicode_console
from credit_copilot.data import schema
from credit_copilot.data.loader import RawDataUnavailableError, load_raw_dataframe
from credit_copilot.models.registry import PREDICTOR_COLUMNS, ModelUnavailableError

_RULE: Final[str] = "=" * 78
_SUBRULE: Final[str] = "-" * 78


def parse_args() -> argparse.Namespace:
    """Read the query and how to load the applicant, if there is one.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("query", help="La consulta del analista, entre comillas.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--applicant-row",
        type=int,
        default=None,
        metavar="ID",
        help="Cargar el solicitante con ese ID desde data/raw/.",
    )
    source.add_argument(
        "--applicant-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Cargar el solicitante desde un JSON con las 23 columnas canónicas.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Ciclos de replanificación permitidos (por defecto {DEFAULT_MAX_ITERATIONS}).",
    )
    return parser.parse_args()


def load_applicant_from_dataset(client_id: int) -> dict[str, int]:
    """Read one client's raw attributes from the downloaded dataset.

    Args:
        client_id: Value of the source `ID` column.

    Returns:
        The 23 raw canonical columns of that client.

    Raises:
        RawDataUnavailableError: The raw file has not been downloaded.
        LookupError: No row carries that identifier.
    """
    frame = load_raw_dataframe().rename(columns=dict(schema.RAW_TO_CANONICAL))
    matching = frame.loc[frame[schema.ID_COLUMN] == client_id]
    if matching.empty:
        raise LookupError(
            f"No hay ninguna fila con {schema.ID_COLUMN}={client_id} en data/raw/. "
            f"El dataset trae {len(frame):,} filas."
        )
    row = matching.iloc[0]
    return {column: int(row[column]) for column in PREDICTOR_COLUMNS}


def load_applicant_from_file(path: Path) -> dict[str, int]:
    """Read an applicant from a JSON object.

    Args:
        path: File holding one JSON object keyed by canonical column name.

    Returns:
        The attributes as integers, exactly as written. Missing columns are **not** filled
        in here; the tool contract refuses them, which is where that refusal belongs.

    Raises:
        ValueError: The file is not a JSON object, or a value is not an integer.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} no contiene un objeto JSON.")
    return {str(key): int(value) for key, value in payload.items()}


def print_answer(state: AgentState) -> None:
    """Print the answer, the tools that produced it, and the sources it rests on."""
    print("\n" + _RULE)
    print("RESPUESTA")
    print(_RULE)
    print(state["answer"] or "(el copiloto no produjo texto)")

    print("\n" + _RULE)
    print("HERRAMIENTAS INVOCADAS")
    print(_RULE)
    if not state["tool_records"]:
        print("  ninguna: la consulta no necesitó herramientas.")
    for index, record in enumerate(state["tool_records"], start=1):
        status = "OK    " if record.ok else "FALLÓ "
        print(f"{index}. [{status}] {record.name}")
        print(f"   argumentos: {json.dumps(record.arguments, ensure_ascii=False)}")
        if record.error:
            print(f"   error     : {record.error}")
        else:
            print(f"   resultado : {_summarise(record.result)}")

    print("\n" + _RULE)
    print("CITAS USADAS")
    print(_RULE)
    citations = unique_citations(state["citations"])
    if not citations:
        print("  ninguna. Ninguna afirmación normativa está respaldada en esta respuesta.")
    for index, citation in enumerate(citations, start=1):
        flags = []
        if citation.is_synthetic:
            flags.append("SINTÉTICO")
        if "DEROGADO" in citation.integrity_notice.upper():
            flags.append("DEROGADO")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{index}. {citation.citation}{suffix}")
        print(f"   fragmento: {citation.chunk_id}")

    print("\n" + _RULE)
    print("COSTO Y CIERRE")
    print(_RULE)
    print(f"  llamadas al LLM        : {state['llm_calls']}")
    print(f"  ciclos de planificación: {state['iterations']}")
    print(f"  evidencia suficiente   : {'sí' if state['sufficient'] else 'no'}")
    print(f"  cómo terminó           : {state['outcome']}")
    if state["gap"]:
        print(f"  hueco declarado        : {state['gap']}")


def _summarise(result: dict[str, object] | None, width: int = 240) -> str:
    """Render a tool result compactly, so the audit trail stays readable."""
    if result is None:
        return "(sin resultado)"
    rendered = json.dumps(result, ensure_ascii=False)
    return rendered if len(rendered) <= width else f"{rendered[:width]}... ({len(rendered)} chars)"


def main() -> int:
    """Load the applicant if asked, run the query, and print the audit trail.

    Returns:
        0 on success, 1 when the copilot could not be built, 2 when the query failed.
    """
    enable_unicode_console()
    args = parse_args()

    applicant: dict[str, int] | None = None
    try:
        if args.applicant_row is not None:
            applicant = load_applicant_from_dataset(args.applicant_row)
        elif args.applicant_file is not None:
            applicant = load_applicant_from_file(args.applicant_file)
    except (RawDataUnavailableError, LookupError, OSError, ValueError) as error:
        print(f"No se pudo cargar el solicitante: {error}", file=sys.stderr)
        return 1

    print(_RULE)
    print("CREDIT RISK COPILOT")
    print(_RULE)
    print(f"Consulta    : {args.query}")
    if applicant is None:
        print("Solicitante : ninguno (las herramientas que lo necesitan van a rechazar)")
    else:
        print(f"Solicitante : {len(applicant)} atributos crudos cargados")
        print(f"              {json.dumps(applicant, ensure_ascii=False)}")
    print(f"Ciclos máx. : {args.max_iterations}")
    print(_SUBRULE)
    print("Cargando el artefacto registrado, el explicador SHAP y el índice vectorial...")

    try:
        context = build_tool_context()
    except (ModelUnavailableError, ToolExecutionError, FileNotFoundError) as error:
        print(f"\nNo se pudo construir el copiloto: {error}", file=sys.stderr)
        return 1
    print("   listo.")

    try:
        state = run_query(
            query=args.query,
            context=context,
            applicant=applicant,
            max_iterations=args.max_iterations,
        )
    except CopilotConfigurationError as error:
        print(f"\nEl copiloto no está configurado: {error}", file=sys.stderr)
        return 1
    except anthropic.APIError as error:
        print(f"\nLa API falló: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print_answer(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
