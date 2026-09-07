"""Evaluation of the answer generator: run the real pipeline, then judge each answer.

Two independent rubric judgments per answer, each a single call to an external model stronger
than the production LLM - the one documented exception to the all-local approach, and only for
this offline evaluation.
"""

import asyncio
import logging
import os
import statistics
from dataclasses import asdict, dataclass, field, replace
from typing import Mapping, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tqdm import tqdm

from rag.config import AppConfig
from rag.evaluation.utils import EvalConfig, EvalRow, append_record, reset
from rag.pipeline.main import Components, answer, new_trace
from rag.pipeline.query_processor import QueryProcessingResult

logger = logging.getLogger(__name__)

HARNESS_ERROR = "HARNESS_ERROR"
STATUS_OK = "OK"

CORRECTNESS_PROMPT = (
    "Eres un evaluador experto en oposiciones a la Administración Pública española. Recibes una "
    "pregunta de examen, la respuesta correcta de referencia y la respuesta generada por un "
    "sistema. Puntúa de 0 a 3 en qué medida la respuesta generada coincide con la de referencia:\n"
    "3 - Coincide plenamente o es correcta en lo esencial, sin errores relevantes.\n"
    "2 - Parcialmente correcta: acierta el núcleo pero omite o confunde parte relevante.\n"
    "1 - Incorrecta, pero sobre el tema preguntado.\n"
    "0 - Contradice la referencia, no responde, o declara no disponer de información.\n\n"
    "Juzga solo el contenido, no el estilo ni la extensión. Una respuesta más detallada que la "
    "referencia no se penaliza si todo lo que añade es correcto. Si la pregunta incluye opciones "
    "a)-d), evalúa si la respuesta generada afirma el contenido de la opción correcta, no si cita "
    "su letra. Responde con la puntuación y una frase breve justificándola."
)

FAITHFULNESS_PROMPT = (
    "Eres un evaluador experto en oposiciones a la Administración Pública española. Recibes unos "
    "fragmentos de normativa (el contexto) y una respuesta generada por un sistema a partir de "
    "ellos. Puntúa de 0 a 3 en qué medida la respuesta se apoya en el contexto:\n"
    "3 - Todas las afirmaciones se deducen del contexto (o no afirma nada, si declara no disponer "
    "de información).\n"
    "2 - Parcialmente apoyada: parte de la respuesta no aparece en el contexto.\n"
    "1 - Mayoría de afirmaciones sin apoyo en el contexto, con algún punto de contacto.\n"
    "0 - Contradice el contexto o es una invención completa.\n\n"
    "Evalúa únicamente si lo afirmado está respaldado por el contexto, NO si es la respuesta "
    "correcta a la pregunta: una respuesta equivocada pero fiel al contexto puntúa alto."
)


class JudgeVerdict(BaseModel):
    """One rubric score with the reasoning behind it."""

    score: int = Field(description="Puntuación entera de 0 a 3.")
    reason: str = Field(description="Una frase breve justificando la puntuación.")


@dataclass(frozen=True)
class GenerationResult:
    """One pipeline run over one eval row, plus both judge scores."""

    row_id: str
    exam: str
    question_type: str
    question: str
    ground_truth_answer: str
    status: str  # PipelineStatus value, or HARNESS_ERROR
    relevance: str  # RelevanceVerdict value
    answer: str
    correctness: int = 0
    correctness_reason: str = ""
    faithfulness: int = 0
    faithfulness_reason: str = ""
    zeroed_reason: str | None = None  # set when scored 0 without a judge call
    durations_ms: dict[str, float] = field(default_factory=dict)
    judge_model: str = ""
    session_id: str = ""
    timestamp_utc: str = ""


async def _judge(
    client: genai.Client, config: EvalConfig, instruction: str, payload: str
) -> JudgeVerdict:
    """Ask the judge for one rubric score, schema-enforced so there is nothing to parse."""
    response = await client.aio.models.generate_content(
        model=config.judge_model,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=config.judge_temperature,
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
        ),
    )
    return response.parsed


async def score_answer(
    client: genai.Client, config: EvalConfig, row: EvalRow, generated: str, contexts: Sequence[str]
) -> tuple[JudgeVerdict, JudgeVerdict]:
    """Judge one answer for correctness and faithfulness, in parallel."""
    correctness_payload = (
        f"Pregunta:\n{row.question}\n\n"
        f"Respuesta de referencia:\n{row.ground_truth_answer}\n\n"
        f"Respuesta generada:\n{generated}"
    )
    faithfulness_payload = (
        f"Contexto:\n{chr(10).join(contexts)}\n\nRespuesta generada:\n{generated}"
    )
    return await asyncio.gather(
        _judge(client, config, CORRECTNESS_PROMPT, correctness_payload),
        _judge(client, config, FAITHFULNESS_PROMPT, faithfulness_payload),
    )


async def evaluate_row(
    row: EvalRow,
    query_result: QueryProcessingResult,
    components: Components,
    app_config: AppConfig,
    client: genai.Client,
    config: EvalConfig,
) -> GenerationResult:
    """Run one row through the real pipeline and judge the answer it produced.

    `query_result` is `run_query_processing`'s already-computed verdict + condensation for this row, so
    `answer()` skips calling the query processor a second time.
    """
    trace = new_trace(app_config, row.exam, [], row.question, session_id=f"eval-{row.row_id}")
    try:
        # Drained to exhaustion, never broken early: answer()'s `finally` is what sets
        # durations_ms["total"] and logs the trace.
        async for _token in answer(components, trace, query_result=query_result):
            pass
    except Exception as exc:
        # answer() never raises, but Qdrant or Ollama dying mid-run can. One bad row must not
        # end a multi-hour pass.
        logger.exception("Harness error on row '%s'.", row.row_id)
        return GenerationResult(
            row_id=row.row_id,
            exam=row.exam,
            question_type=row.question_type,
            question=row.question,
            ground_truth_answer=row.ground_truth_answer,
            status=HARNESS_ERROR,
            relevance=trace.relevance.value,
            answer="",
            zeroed_reason=f"harness_error={exc!r}",
            session_id=trace.session_id,
            timestamp_utc=trace.timestamp_utc,
        )

    result = GenerationResult(
        row_id=row.row_id,
        exam=row.exam,
        question_type=row.question_type,
        question=row.question,
        ground_truth_answer=row.ground_truth_answer,
        status=trace.status.value,
        relevance=trace.relevance.value,
        answer=trace.answer,
        durations_ms=dict(trace.durations_ms),
        judge_model=config.judge_model,
        session_id=trace.session_id,
        timestamp_utc=trace.timestamp_utc,
    )

    # A row the pipeline never answered scores zero without spending a judge call.
    if result.status != STATUS_OK:
        return replace(result, zeroed_reason=f"status={result.status}")

    contexts = [chunk.content for chunk in trace.final_chunks]
    try:
        correctness, faithfulness = await score_answer(
            client, config, row, result.answer, contexts
        )
    except Exception as exc:
        logger.exception("Judging failed for row '%s'.", row.row_id)
        return replace(result, zeroed_reason=f"judge_error={exc!r}")

    return replace(
        result,
        correctness=correctness.score,
        correctness_reason=correctness.reason,
        faithfulness=faithfulness.score,
        faithfulness_reason=faithfulness.reason,
    )


async def evaluate_generation(
    rows: Sequence[EvalRow],
    query_results: Mapping[str, QueryProcessingResult],
    components: Components,
    app_config: AppConfig,
    config: EvalConfig,
    path: str,
) -> None:
    """Run and judge every graded row, appending each record as soon as it completes.

    `query_results` maps `row_id -> QueryProcessingResult` - see `evaluate_row` for why the real
    query processor isn't called again here.

    Sequential by design: one CPU-bound Ollama instance serves both models, so concurrency would
    only make them evict each other. `keep_alive` is an idle timer reset on every request, so the
    models stay resident for the whole run.
    """
    client = genai.Client(
        vertexai=True,
        api_key=os.environ["GOOGLE_API_KEY"],
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=2, http_status_codes=[429]),
        )
    )
    reset(path)
    logger.info("Generating and judging %d rows.", len(rows))

    for row in tqdm(rows, desc="generate", unit="row"):
        result = await evaluate_row(row, query_results[row.row_id], components, app_config, client, config)
        append_record(path, asdict(result))


@dataclass(frozen=True)
class GenerationMetrics:
    """Aggregate judge scores, with the hard-zero count kept visible."""

    n_rows: int
    n_zeroed: int
    correctness: float
    faithfulness: float
    correctness_normalized: float
    faithfulness_normalized: float


def summarize(records: Sequence[dict], max_score: int) -> GenerationMetrics:
    """Mean correctness and faithfulness, raw and normalized to 0-1.

    Rows the pipeline failed to answer score 0 and stay in the denominator: excluding them would
    reward a system that refuses to answer with a smaller denominator.
    """
    if not records:
        return GenerationMetrics(0, 0, 0.0, 0.0, 0.0, 0.0)

    correctness = statistics.fmean(record["correctness"] for record in records)
    faithfulness = statistics.fmean(record["faithfulness"] for record in records)
    return GenerationMetrics(
        n_rows=len(records),
        n_zeroed=sum(1 for record in records if record.get("zeroed_reason")),
        correctness=correctness,
        faithfulness=faithfulness,
        correctness_normalized=correctness / max_score,
        faithfulness_normalized=faithfulness / max_score,
    )


def stage_latencies(records: Sequence[dict]) -> list[dict]:
    """Per-stage latency over the rows that actually reached each stage.

    No "processing" stage here: `run_generate` reuses `run_query_processing`'s condensed queries
    instead of running the query processor again, so `durations_ms` never carries it - see
    `query_processing.processing_latency` for that stage's timing.
    """
    latencies: list[dict] = []
    for stage in ("retrieve", "generate", "total"):
        values = [
            record["durations_ms"][stage]
            for record in records
            if stage in record.get("durations_ms", {})
        ]
        if not values:
            continue
        latencies.append(
            {
                "stage": stage,
                "n": len(values),
                "avg_s": round(statistics.fmean(values) / 1000, 2),
                "max_s": round(max(values) / 1000, 2),
            }
        )
    return latencies


def score_breakdown_rows(records: Sequence[dict], score_field: str, max_score: int) -> list[dict]:
    """One row per `question_type` plus a `total` row: a 0..max_score score histogram + average.

    `score_field` is `"correctness"` or `"faithfulness"`.
    """
    rows = []
    for question_type in sorted({record["question_type"] for record in records}):
        subset = [record for record in records if record["question_type"] == question_type]
        rows.append(_score_row(f"question_type={question_type}", subset, score_field, max_score))
    rows.append(_score_row("total", records, score_field, max_score))
    return rows


def _score_row(label: str, records: Sequence[dict], score_field: str, max_score: int) -> dict:
    """One score-histogram row: counts at each level 0..max_score, plus the average."""
    row = {
        "grupo": label,
        "n_rows": len(records),
        "n_zeroed": sum(1 for record in records if record.get("zeroed_reason")),
    }
    for level in range(max_score + 1):
        row[str(level)] = sum(1 for record in records if record[score_field] == level)
    row["avg"] = round(statistics.fmean(record[score_field] for record in records), 2) if records else 0.0
    return row
