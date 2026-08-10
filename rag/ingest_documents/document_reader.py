from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

from rag.ingest_documents.patterns import PATTERNS


class DocumentReader:
    """Extracts document text from PDFs or cached .txt files."""

    def run(
        self,
        source_dir: str,
        from_pdf: bool = False,
        export_dir: str | None = None,
    ) -> dict[str, str]:
        """Read every document in `source_dir` into a `{doc_id: text}` mapping.

        Args:
            source_dir: Folder containing either .pdf or .txt files.
            from_pdf: Read .pdf files and extract their text if True; read cached
                .txt files otherwise.
            export_dir: If given, write each document's text to `export_dir` as
                `<doc_id>.txt` after reading.

        Returns:
            Mapping of doc_id (file stem) to extracted text.
        """
        docs = self._read_pdfs(source_dir) if from_pdf else self._read_txts(source_dir)
        if export_dir:
            self._export_extracted_texts(docs, export_dir)
        return docs

    def _read_pdf_file(self, filename: str) -> str:
        """Extract text from a PDF file, dropping anything before "TEXTO CONSOLIDADO"."""

        texts = []
        with open(filename, "rb") as file:
            reader = PdfReader(file)
            for page in range(reader.get_num_pages()):
                texts.append(reader.pages[page].extract_text())

        doc_text = "\n".join(texts)

        # Only trust the heading "TEXTO CONSOLIDADO" as a cut point when it appears exactly once.
        matches = list(PATTERNS["texto_consolidado"].finditer(doc_text))
        if len(matches) == 1:
            doc_text = doc_text[matches[0].start():]

        return doc_text

    def _read_pdfs(self, folder: str) -> dict[str, str]:
        """Extract text from every PDF file in `folder`."""

        documents = {}
        pdf_paths = sorted(Path(folder).glob("*.pdf"))
        for file_path in tqdm(pdf_paths, desc="Reading PDFs"):
            documents[file_path.stem] = self._read_pdf_file(str(file_path))
        return documents

    def _read_txts(self, folder: str) -> dict[str, str]:
        """Read every .txt file in `folder`."""

        documents = {}
        txt_paths = sorted(Path(folder).glob("*.txt"))
        for file_path in tqdm(txt_paths, desc="Reading TXTs"):
            documents[file_path.stem] = file_path.read_text(encoding="utf-8")
        return documents

    def _export_extracted_texts(self, docs: dict[str, str], output_dir: str) -> None:
        """Write each document's text to `output_dir` as `<doc_id>.txt`."""

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for doc_id, text in docs.items():
            (out / f"{doc_id}.txt").write_text(text, encoding="utf-8")
