import logging
import re
from dataclasses import dataclass
from typing import TypedDict

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoTokenizer

from rag.ingest_documents.patterns import ARTICULO_ORDINAL, CITATION_LEADIN, EMBEDDED_DOCUMENT_TITLE, PATTERNS

logger = logging.getLogger(__name__)

# Documents whose PDF extraction is too mangled for any heading pattern to be trusted
# Chunked by token size only.
FORCE_FULL_DOCUMENT_IDS = frozenset({"doc198"})


@dataclass
class Chunk:
    doc_id: str
    tipo_seccion: str  # one of: "preambulo" | "articulo" | "disposicion" | "titulo" | "capitulo" | "anexo" | "none"
    sección: str | None = None
    content: str = ""
    n_tokens: int | None = None


class ChunkRecord(TypedDict):
    """A chunk together with the document metadata it inherits, ready for ingestion."""

    doc_id: str
    doc_name: str | None
    doc_date: str | None
    source_url: str | None
    exams_id: list[str]
    tema: list[str]
    tipo_seccion: str
    sección: str | None
    n_tokens: int | None
    content: str


# Required columns in the metadata CSV.
METADATA_CSV_COLUMNS = ["doc_id", "doc_name", "doc_date", "source_url", "exams_id", "tema"]


class DocumentChunker:
    """Segments documents into legal-structure chunks and splits any chunk that
    exceeds `max_length_tokens`.
    """

    def __init__(
        self,
        model_name: str,
        overlap_tokens: int = 50,
        max_length_tokens: int = 1024,
    ) -> None:
        """Load the tokenizer and build the fallback token splitter.

        `max_length_tokens` is a fixed cap chosen from the corpus's own chunk-size distribution"""

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._max_length = min(max_length_tokens, self._tokenizer.model_max_length)

        # The model prepends/appends special tokens ([CLS]/[SEP]) at embedding time, but
        # the splitter's length function counts raw tokens only. Reserve room for them.
        n_special_tokens = self._tokenizer.num_special_tokens_to_add()
        self._splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer,
            chunk_size=self._max_length - n_special_tokens,
            chunk_overlap=overlap_tokens,
        )

    def run(
        self,
        docs: dict[str, str],
        metadata_df: pd.DataFrame,
    ) -> tuple[list[ChunkRecord], dict[str, dict]]:
        """Segment every document in a corpus."""

        missing_columns = [c for c in METADATA_CSV_COLUMNS if c not in metadata_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in metadata_df: {missing_columns}")

        indexed_metadata = (
            metadata_df.set_index("doc_id")
            if metadata_df.index.name != "doc_id"
            else metadata_df
        )

        corpus_chunks: list[ChunkRecord] = []
        chunking_review: dict[str, dict] = {}
        for doc_id, text in tqdm(docs.items(), desc="Chunking documents"):
            if doc_id not in indexed_metadata.index:
                logger.warning("doc_id '%s' has no row in the metadata CSV, skipping.", doc_id)
                continue
            row = indexed_metadata.loc[doc_id].to_dict()
            doc_chunks, doc_review = self._process_document(doc_id, text, row)
            corpus_chunks.extend(doc_chunks)
            chunking_review[doc_id] = doc_review

        return corpus_chunks, chunking_review

    def _count_tokens(self, text: str) -> int:
        """Return how many tokens `text` encodes to, including special tokens."""
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def _find_matches(self, pattern: re.Pattern, text: str) -> list[re.Match]:
        """Return every match of `pattern` in `text`."""
        return list(pattern.finditer(text))

    def _extract_preamble(self, doc_id: str, text: str, first_match_start: int) -> Chunk | None:
        """Return a preamble Chunk for the text before `first_match_start`, or None if
        there is none.
        """
        if first_match_start == 0:
            return None
        preamble = text[:first_match_start].strip()
        return Chunk(doc_id=doc_id, tipo_seccion="preambulo", content=preamble) if preamble else None

    def _is_citation(self, text: str, match: re.Match) -> bool:
        """True if `match` is a heading-shaped false positive rather than a real heading.

        Two cases: a cross-reference sitting in the same, still-open sentence as a
        citation lead-in phrase (e.g. "...a que se refiere la\\nDisposición..."), or
        quoted replacement text for an article in a *different*, amended law - an
        amending law saying 'queda redactado como sigue: «Artículo 91 septies. ...»' -
        detected by the match sitting inside a still-open «...» span.

        An article headed by an ordinal word ("Artículo segundo.") is exempt from the
        quote check: it is always the amending law's own division, never quoted text,
        and a quoted block whose closing » was dropped (annulled content) would
        otherwise swallow the rest of the document.
        """

        last_open_quote = text.rfind("«", 0, match.start())
        last_close_quote = text.rfind("»", 0, match.start())
        return last_open_quote > last_close_quote

    def _numeric_sequence(self, matches: list[re.Match]) -> list[re.Match]:
        """Return the subset of bare-number matches forming a real enumeration (1, 2,
        3...), or an empty list if there is none.

        Numbers that don't continue the run are dropped, so a stray page number does
        not invalidate an otherwise valid enumeration. The run must still account for
        most of the matches - otherwise the numbers are incidental (sub-items inside
        ordinary articles) rather than the document's structure - and have at least 3
        items.
        """
        sequence: list[re.Match] = []
        expected = 1
        for match in matches:
            if int(match.group(1)) == expected:
                sequence.append(match)
                expected += 1
        if len(sequence) < 3 or len(sequence) < 0.8 * len(matches):
            return []
        return sequence

    def _add_numbered_articles(self, body_text: str, headings: list[dict]) -> list[dict]:
        """Prepend bare-numbered articles ("1.", "2.") to `headings`.

        Some documents number their articles without the word "Artículo", so only their
        disposiciones get picked up and the whole articulado ends up in the preamble.
        Only applies when no real article heading was found and the numbers before the
        first heading form a valid enumeration.
        """
        if any(h["tipo_seccion"] == "articulo" for h in headings):
            return headings

        articulado = body_text[:headings[0]["start"]]
        numero_matches = self._numeric_sequence(
            self._find_matches(PATTERNS["numero_bare"], articulado)
        )
        numbered = [
            {"start": m.start(), "heading": m.group(0).strip(), "tipo_seccion": "articulo"}
            for m in numero_matches
        ]
        return numbered + headings

    def _split_chunk_by_anexos(self, chunk: Chunk) -> list[Chunk]:
        """Split a chunk by anexo boundaries if they exist."""

        anexo_matches = self._find_matches(PATTERNS["anexo"], chunk.content)
        if not anexo_matches:
            return [chunk]

        result: list[Chunk] = []

        if anexo_matches[0].start() > 0:
            pre_content = chunk.content[:anexo_matches[0].start()].strip()
            if pre_content:
                result.append(
                    Chunk(
                        doc_id=chunk.doc_id,
                        tipo_seccion=chunk.tipo_seccion,
                        sección=chunk.sección,
                        content=pre_content,
                    )
                )

        for i, match in enumerate(anexo_matches):
            start = match.start()
            end = anexo_matches[i + 1].start() if i + 1 < len(anexo_matches) else len(chunk.content)
            result.append(
                Chunk(
                    doc_id=chunk.doc_id,
                    tipo_seccion="anexo",
                    sección=match.group(0).strip(),
                    content=chunk.content[start:end].strip(),
                )
            )

        return result

    def _build_article_headings(self, body_text: str) -> list[dict]:
        """Resolve the ordered list of `{start, heading, tipo_seccion}` article/
        disposición headings in `body_text` (the pre-anexo portion of the document).

        Drops `articulo_disposicion` matches that `_is_citation` rejects: an inline
        cross-reference, or quoted replacement text for an article in a different,
        amended law.

        Promotes bare ordinal items ("Primero.", "Segundo.") to real headings in two
        cases:
        (a) they appear before the document's first real heading;
        (b) they continue an unnumbered "Disposición <tipo>" / "DISPOSICIONES <TIPO>" category header

        A bare ordinal appearing after real structure has already begun, with no
        pending category, is left as plain body content.

        A category header with no ordinal items under it ("DISPOSICIÓN FINAL" followed
        straight by its text) becomes a heading in its own right.
        """

        tagged = []
        for m in self._find_matches(PATTERNS["articulo_disposicion"], body_text):
            if self._is_citation(body_text, m):
                continue
            heading = m.group(0).strip()
            tipo = "disposicion" if heading.lower().startswith("disposici") else "articulo"
            tagged.append({"start": m.start(), "kind": "heading", "heading": heading, "tipo_seccion": tipo})
        for m in self._find_matches(PATTERNS["disposicion_categoria"], body_text):
            tagged.append({"start": m.start(), "kind": "categoria", "heading": m.group(0).strip()})
        for m in self._find_matches(PATTERNS["ordinal_bare"], body_text):
            tagged.append({"start": m.start(), "kind": "ordinal", "heading": m.group(0).strip()})
        tagged.sort(key=lambda t: t["start"])

        headings = []
        pending_categoria = None
        seen_real_heading = False
        for i, t in enumerate(tagged):
            if t["kind"] == "categoria":
                seen_real_heading = True
                next_kind = tagged[i + 1]["kind"] if i + 1 < len(tagged) else None
                if next_kind == "ordinal":
                    pending_categoria = t["heading"]
                else:
                    pending_categoria = None
                    headings.append({
                        "start": t["start"],
                        "heading": t["heading"],
                        "tipo_seccion": "disposicion",
                    })
            elif t["kind"] == "ordinal":
                if pending_categoria is not None:
                    headings.append({
                        "start": t["start"],
                        "heading": f"{pending_categoria} {t['heading']}",
                        "tipo_seccion": "disposicion",
                    })
                elif not seen_real_heading:
                    headings.append({"start": t["start"], "heading": t["heading"], "tipo_seccion": "articulo"})
                # else: mid-document bare ordinal with no pending category - left as body content.
            else:
                pending_categoria = None
                seen_real_heading = True
                headings.append(t)
        return headings

    def _segment_by_article(self, doc_id: str, text: str, headings: list[dict]) -> list[Chunk]:
        """Split `text` into one Chunk per resolved heading."""

        chunks: list[Chunk] = []

        preamble = self._extract_preamble(doc_id, text, headings[0]["start"])
        if preamble:
            chunks.append(preamble)

        for i, h in enumerate(headings):
            start = h["start"]
            end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    tipo_seccion=h["tipo_seccion"],
                    sección=h["heading"],
                    content=text[start:end].strip(),
                )
            )
        return chunks

    def _segment_by_pattern(
        self,
        doc_id: str,
        text: str,
        matches: list[re.Match],
        tipo_seccion: str,
    ) -> list[Chunk]:
        """Fallback split by title, chapter, or bare ordinal/numeric headings."""

        chunks: list[Chunk] = []

        preamble = self._extract_preamble(doc_id, text, matches[0].start())
        if preamble:
            chunks.append(preamble)

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = match.group(0).strip()
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    tipo_seccion=tipo_seccion,
                    sección=heading,
                    content=text[start:end].strip(),
                )
            )
        return chunks

    def _segment_document(self, doc_id: str, text: str) -> tuple[list[Chunk], str]:
        """Apply the hierarchical segmentation strategy.

        A few known documents embed a second, self-contained document after the outer decree's own body. When one
        of those known titles is found, everything before it - including the outer
        document's own articles/disposiciones - becomes a single preamble chunk, and
        the embedded text is segmented again from scratch so its own real structure is kept.

        Otherwise, heading patterns are only searched for in the text before the
        first ANEXO: anexo content frequently has its own internal numbering that must not be mistaken for the
        parent document's headings. The anexo itself is still preserved in full and
        later carved into its own chunk(s).
        """

        # Documents with embedded documents
        embedded_match = EMBEDDED_DOCUMENT_TITLE.search(text)
        if embedded_match:
            preamble = text[:embedded_match.start()].strip()
            # Some embedded documents are reproduced as a quotation. The opening « would
            # otherwise make every heading inside look like quoted amending text.
            embedded_text = text[embedded_match.end():].lstrip().lstrip("«")
            embedded_chunks, embedded_level = self._segment_document(doc_id, embedded_text)
            chunks = [Chunk(doc_id=doc_id, tipo_seccion="preambulo", content=preamble)] if preamble else []
            chunks.extend(embedded_chunks)
            return chunks, embedded_level

        # Find Anexos to avoid segmenting them
        anexo_matches = self._find_matches(PATTERNS["anexo"], text)
        body_text = text[:anexo_matches[0].start()] if anexo_matches else text

        # Documents whose extraction is too mangled to trust any heading.
        if doc_id in FORCE_FULL_DOCUMENT_IDS:
            return self._segment_by_length(doc_id, text, anexo_matches), "documento completo"

        # Segment by article
        headings = self._build_article_headings(body_text)
        if headings:
            headings = self._add_numbered_articles(body_text, headings)
            return self._segment_by_article(doc_id, text, headings), "articulo_disposicion"

        # Fallback when articles have unconventional naming
        ordinal_matches = self._find_matches(PATTERNS["ordinal_bare"], body_text)
        if ordinal_matches:
            return self._segment_by_pattern(doc_id, text, ordinal_matches, "articulo"), "articulo_disposicion"

        numero_matches = self._numeric_sequence(self._find_matches(PATTERNS["numero_bare"], body_text))
        if numero_matches:
            return self._segment_by_pattern(doc_id, text, numero_matches, "articulo"), "articulo_disposicion"

        # Segment by Titulo
        title_matches = self._find_matches(PATTERNS["titulo"], body_text)
        if title_matches:
            return self._segment_by_pattern(doc_id, text, title_matches, "titulo"), "titulo"

        # Segment by Capitulo
        chapter_matches = self._find_matches(PATTERNS["capitulo"], body_text)
        if chapter_matches:
            return self._segment_by_pattern(doc_id, text, chapter_matches, "capitulo"), "capitulo"

        # Last-resort: fixed-size token chunking of the body text.
        return self._segment_by_length(doc_id, text, anexo_matches), "documento completo"

    def _segment_by_length(
        self,
        doc_id: str,
        text: str,
        anexo_matches: list[re.Match],
    ) -> list[Chunk]:
        """Fixed-size token chunking of the body text. Anexos are appended to the last
        part so they can still be segmented separately afterwards.
        """

        body_text = text[:anexo_matches[0].start()] if anexo_matches else text
        parts = self._splitter.split_text(body_text) if body_text.strip() else []
        if anexo_matches:
            anexo_tail = text[anexo_matches[0].start():]
            if parts:
                parts[-1] = parts[-1] + "\n" + anexo_tail
            else:
                parts = [anexo_tail]
        return [
            Chunk(doc_id=doc_id, tipo_seccion="none", content=part, n_tokens=self._count_tokens(part))
            for part in parts
        ]

    def _split_long_chunks(self, chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        """Split any chunk exceeding the embedding token limit into overlapping
        sub-chunks."""

        result: list[Chunk] = []
        n_split = 0

        for chunk in chunks:
            chunk.n_tokens = self._count_tokens(chunk.content)

            if chunk.n_tokens <= self._max_length:
                result.append(chunk)
                continue

            n_split += 1
            for part in self._splitter.split_text(chunk.content):
                result.append(
                    Chunk(
                        doc_id=chunk.doc_id,
                        tipo_seccion=chunk.tipo_seccion,
                        sección=chunk.sección,
                        content=part,
                        n_tokens=self._count_tokens(part),
                    )
                )

        return result, n_split

    def _process_document(
        self,
        doc_id: str,
        text: str,
        metadata_row: dict,
    ) -> tuple[list[ChunkRecord], dict]:
        """Segment and split a single document into chunk records, plus a review summary."""

        chunks, segmentation_level = self._segment_document(doc_id, text)

        # Note: anexos always fall in the last chunk.
        if chunks:
            chunks[-1:] = self._split_chunk_by_anexos(chunks[-1])

        sections = [chunk.sección or chunk.tipo_seccion for chunk in chunks]

        chunks, n_split = self._split_long_chunks(chunks)
        n_after_split = len(chunks)

        exams_id = [id for id in (metadata_row.get("exams_id") or "").split("|") if id]
        tema = [tema for tema in (metadata_row.get("tema") or "").split("|") if tema]

        doc_chunks: list[ChunkRecord] = []
        preamble_length = 0
        for chunk in chunks:
            if chunk.tipo_seccion == "preambulo":
                preamble_length += chunk.n_tokens
            doc_chunks.append(
                {
                    "doc_id": doc_id,
                    "doc_name": metadata_row.get("doc_name"),
                    "doc_date": metadata_row.get("doc_date"),
                    "source_url": metadata_row.get("source_url"),
                    "exams_id": exams_id,
                    "tema": tema,
                    "tipo_seccion": chunk.tipo_seccion,
                    "sección": chunk.sección,
                    "n_tokens": chunk.n_tokens,
                    "content": chunk.content,
                }
            )

        doc_review = {
            "segmentation_level": segmentation_level,
            "n_different_sections": len(sections),
            "sections": sections,
            "n_split_chunks": n_split,
            "final_n_chunks": n_after_split,
            "preamble_length_tokens": preamble_length,
        }
        return doc_chunks, doc_review
