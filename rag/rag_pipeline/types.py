"""Shared types for the query-side RAG pipeline: enums and dataclasses passed
between stages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class PipelineStatus(str, Enum):
    """Terminal outcome of a pipeline run."""

    OK = "OK"
    OFF_TOPIC = "OFF_TOPIC"
    NO_CONTEXT = "NO_CONTEXT"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    GENERATION_ERROR = "GENERATION_ERROR"


class RelevanceVerdict(str, Enum):
    """Outcome of the relevance-check stage."""

    RELEVANT = "RELEVANT"
    OFF_TOPIC = "OFF_TOPIC"
    UNKNOWN = "UNKNOWN"  # checker failed; treated as RELEVANT (fail open)


STATUS_MESSAGES: dict[PipelineStatus, str] = {
    PipelineStatus.OFF_TOPIC: (
        "Mi objetivo es responder preguntas relacionadas con exámenes de oposición. "
        "Parece que tu pregunta no es sobre este tema, prueba a formularla de otra manera."
    ),
    PipelineStatus.NO_CONTEXT: (
        "No he encontrado normativa relevante para responder esa pregunta en el temario "
        "seleccionado. Intenta reformularla y comprueba que el tema/oposición seleccionado"
        " sea el correcto."
    ),
    PipelineStatus.RETRIEVAL_ERROR: (
        "Ha ocurrido algún problema. Inténtalo de nuevo en unos segundos."
    ),
    PipelineStatus.GENERATION_ERROR: (
        "Ha ocurrido algún problema. Inténtalo de nuevo en unos segundos."
    ),
}


@dataclass(frozen=True)
class ChunkMetadata:
    """Normalized payload metadata. Mirrors the ingestion `ChunkRecord` with
    `sección` renamed to `seccion`, the only place that rename happens on the
    query side.
    """

    doc_id: str
    doc_name: str | None
    doc_date: str | None
    source_url: str | None
    exams_id: list[str]
    tema: list[str]
    tipo_seccion: str
    seccion: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ChunkMetadata":
        """Build from a raw Qdrant metadata dict, renaming `sección` -> `seccion`.

        Args:
            payload: The `metadata` sub-dict of a Qdrant point payload.

        Returns:
            Normalized chunk metadata.
        """
        return cls(
            doc_id=payload["doc_id"],
            doc_name=payload.get("doc_name"),
            doc_date=payload.get("doc_date"),
            source_url=payload.get("source_url"),
            exams_id=list(payload.get("exams_id") or []),
            tema=list(payload.get("tema") or []),
            tipo_seccion=payload.get("tipo_seccion", "none"),
            seccion=payload.get("sección"),
        )


@dataclass
class RetrievedChunk:
    """A single chunk as it flows through retrieval, reranking and selection."""

    point_id: str
    content: str
    metadata: ChunkMetadata
    retrieval_score: float  # hybrid RRF, uncalibrated
    rerank_score: float | None = None  # cross-encoder logit
    rerank_prob: float | None = None  # sigmoid(logit), calibrated 0..1


@dataclass(frozen=True)
class SourceRef:
    """Citation metadata for one bracketed index in the generated answer."""

    index: int  # 1-based, matches [1] in the answer
    doc_name: str | None
    seccion: str | None
    source_url: str | None
    doc_id: str
    point_id: str


@dataclass
class PipelineTrace:
    """Full record of one pipeline run: inputs, intermediate results, timings."""

    session_id: str
    timestamp_utc: str
    exam: str
    temas: list[str]
    raw_query: str
    condensed_query: str
    relevance: RelevanceVerdict
    status: PipelineStatus
    message: str = ""  # user-facing text for non-OK statuses
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    reranked: list[RetrievedChunk] = field(default_factory=list)
    final_chunks: list[RetrievedChunk] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    answer: str = ""
    durations_ms: dict[str, float] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    error: str | None = None
