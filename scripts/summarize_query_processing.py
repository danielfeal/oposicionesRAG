"""Print (or write to markdown) a per-row summary of a query_processing.jsonl evaluation artifact.

Columns: original question, condensed question, relevance verdict, whether it was edited.

Usage (from repo root):
    python scripts/summarize_query_processing.py
    python scripts/summarize_query_processing.py --out documentation/evaluation/query_processing_summary.md
"""

import argparse
import json
from pathlib import Path

DEFAULT_PATH = "documentation/evaluation/query_processing.jsonl"


def load_rows(path: str) -> list[dict]:
    """Read a query_processing.jsonl file into a list of records."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_markdown(rows: list[dict]) -> str:
    """Render one table row per question: original, condensed, verdict, edited?"""
    lines = [
        "| row_id | pregunta original | pregunta condensada | verdict | editada |",
        "|--------|--------------------|----------------------|---------|---------|",
    ]
    for row in rows:
        edited = row["question"] != row["condensed_query"]
        lines.append(
            f"| {row['row_id']} | {row['question']} | {row['condensed_query']} | "
            f"{row['verdict']} | {edited} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=DEFAULT_PATH, help="Path to query_processing.jsonl")
    parser.add_argument("--out", default=None, help="Write markdown here instead of printing")
    args = parser.parse_args()

    rows = load_rows(args.path)
    markdown = to_markdown(rows)

    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"Written {len(rows)} rows to '{args.out}'.")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
