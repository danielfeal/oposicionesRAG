"""SQLite storage for pipeline traces, used by the evaluation chapter."""

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from rag.rag_pipeline.types import PipelineTrace, RetrievedChunk

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc       TEXT    NOT NULL,
    session_id          TEXT    NOT NULL,
    exam                TEXT    NOT NULL,
    temas               TEXT,            -- JSON array
    raw_query           TEXT    NOT NULL,
    condensed_query     TEXT    NOT NULL,
    relevance           TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    message             TEXT,
    retrieved_ids       TEXT,            -- JSON [{point_id, doc_id, score}]
    reranked_ids        TEXT,            -- JSON [{point_id, doc_id, score, prob}]
    final_ids           TEXT,            -- JSON [point_id]
    sources             TEXT,            -- JSON [SourceRef]
    answer              TEXT,
    durations_ms        TEXT,            -- JSON {stage: ms}
    models              TEXT,            -- JSON {role: name}
    error               TEXT
);
"""


def _chunk_summary(chunks: Sequence[RetrievedChunk]) -> list[dict]:
    """Compact JSON-able summary of a chunk list, for the trace store."""
    return [
        {
            "point_id": c.point_id,
            "doc_id": c.metadata.doc_id,
            "retrieval_score": c.retrieval_score,
            "rerank_score": c.rerank_score,
            "rerank_prob": c.rerank_prob,
        }
        for c in chunks
    ]


class SqliteTraceStore:
    """Appends one row per query to a local SQLite DB, for the evaluation chapter.

    Opened with `check_same_thread=False` and guarded by a lock because `log`
    is called from an `asyncio.to_thread` worker, not the event loop thread.
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(CREATE_TABLE_SQL)
            self._conn.commit()

    def log(self, trace: PipelineTrace) -> None:
        """Serialize and insert `trace`. Never raises: logging must not break an answer."""
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO interactions (
                        timestamp_utc, session_id, exam, temas, raw_query,
                        condensed_query, relevance, status, message,
                        retrieved_ids, reranked_ids, final_ids, sources,
                        answer, durations_ms, models, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace.timestamp_utc,
                        trace.session_id,
                        trace.exam,
                        json.dumps(trace.temas, ensure_ascii=False),
                        trace.raw_query,
                        trace.condensed_query,
                        trace.relevance.value,
                        trace.status.value,
                        trace.message,
                        json.dumps(_chunk_summary(trace.retrieved), ensure_ascii=False),
                        json.dumps(_chunk_summary(trace.reranked), ensure_ascii=False),
                        json.dumps([c.point_id for c in trace.final_chunks], ensure_ascii=False),
                        json.dumps([s.__dict__ for s in trace.sources], ensure_ascii=False),
                        trace.answer,
                        json.dumps(trace.durations_ms, ensure_ascii=False),
                        json.dumps(trace.models, ensure_ascii=False),
                        trace.error,
                    ),
                )
                self._conn.commit()
        except Exception:
            logger.exception("Failed to log trace for session '%s'.", trace.session_id)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()


class NullTraceStore:
    """No-op store for tests and offline runs."""

    def log(self, trace: PipelineTrace) -> None:
        """Discard `trace`."""
