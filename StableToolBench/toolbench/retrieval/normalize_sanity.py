"""CLI utility for normalizing sanity-check retrieval logs.

Consumes the JSON sanity-check files produced by ``rapidapi.write_sanity_check``
and emits a tabular summary where both the ground-truth APIs and retrieved tools
share the same normalized schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableSequence, Sequence

from StableToolBench.server.utils import change_name, standardize, standardize_category


@dataclass
class NormalizedRecord:
    """Container for normalized and original tool identifiers."""

    category: str
    tool_name: str
    api_name: str
    original_category: str
    original_tool_name: str
    original_api_name: str


def _normalize_component(value: str | None, *, is_category: bool = False) -> str:
    if value is None:
        return ""
    text = str(value)
    if is_category:
        text = standardize_category(text)
    text = change_name(standardize(text))
    return text


def _coerce_sequence(entry: Sequence[object]) -> tuple[str | None, str | None, str | None]:
    if not entry:
        return None, None, None
    if len(entry) >= 3:
        category, tool_name, api_name = entry[:3]
        return (str(category) if category is not None else None,
                str(tool_name) if tool_name is not None else None,
                str(api_name) if api_name is not None else None)
    if len(entry) == 2:
        tool_name, api_name = entry
        return None, str(tool_name) if tool_name is not None else None, str(api_name) if api_name is not None else None
    return str(entry[0]) if entry[0] is not None else None, None, None  # pragma: no cover - degenerate input


def _extract_names(entry: object) -> tuple[str | None, str | None, str | None]:
    if isinstance(entry, Mapping):
        mapping = entry
        category = mapping.get("category") or mapping.get("category_name") or mapping.get("cate_name")
        tool_name = (
            mapping.get("tool_name")
            or mapping.get("tool")
            or mapping.get("toolName")
            or mapping.get("name")
        )
        api_name = mapping.get("api_name") or mapping.get("api") or mapping.get("apiName")
        return category, tool_name, api_name
    if isinstance(entry, (list, tuple)):
        return _coerce_sequence(entry)
    if entry is None:
        return None, None, None
    # Fallback: treat bare string as API name
    text = str(entry)
    return None, None, text


def normalize_entries(items: Iterable[object] | object | None) -> List[NormalizedRecord]:
    normalized: List[NormalizedRecord] = []
    if items is None:
        return normalized

    if isinstance(items, (str, bytes)):
        iterable: Iterable[object] = [items]
    elif isinstance(items, Mapping):
        iterable = [items]
    else:
        try:
            iterable = list(items)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover - defensive branch
            iterable = [items]

    for raw in iterable:
        category, tool_name, api_name = _extract_names(raw)
        normalized.append(
            NormalizedRecord(
                category=_normalize_component(category, is_category=True),
                tool_name=_normalize_component(tool_name),
                api_name=_normalize_component(api_name),
                original_category=str(category) if category is not None else "",
                original_tool_name=str(tool_name) if tool_name is not None else "",
                original_api_name=str(api_name) if api_name is not None else "",
            )
        )
    return normalized


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "    ") -> str:
    widths: List[int] = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = indent + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = indent + "-+-".join("-" * w for w in widths)
    body_lines = [
        indent + " | ".join((row[i] if i < len(row) else "").ljust(widths[i]) for i in range(len(headers)))
        for row in rows
    ]
    return "\n".join([header_line, separator, *body_lines]) if rows else indent + "(no entries)"


def _format_records(records: Sequence[NormalizedRecord], *, include_index: bool = True) -> str:
    headers: MutableSequence[str]
    if include_index:
        headers = ["#", "category", "tool_name", "api_name", "orig_category", "orig_tool", "orig_api"]
    else:
        headers = ["category", "tool_name", "api_name", "orig_category", "orig_tool", "orig_api"]

    rows: List[List[str]] = []
    for idx, record in enumerate(records, start=1):
        row = [
            str(idx),
            record.category or "—",
            record.tool_name or "—",
            record.api_name or "—",
            record.original_category or "—",
            record.original_tool_name or "—",
            record.original_api_name or "—",
        ]
        if not include_index:
            row = row[1:]
        rows.append(row)
    return _format_table(headers, rows)


def _iter_json_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        if path.suffix.lower() == ".json":
            yield path
        return
    if path.is_dir():
        for candidate in sorted(path.rglob("*.json")):
            if candidate.is_file():
                yield candidate


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_file(path: Path) -> str:
    try:
        payload = _load_json(path)
    except Exception as exc:  # pragma: no cover - I/O errors surfaced to user
        return f"{path}: failed to read JSON ({exc})"

    if not isinstance(payload, Mapping):
        return f"{path}: JSON root must be an object"

    query_id = payload.get("query_id")
    query = payload.get("query")
    ground_truth = normalize_entries(payload.get("ground_truth", []))

    header_lines = [f"File: {path}"]
    if query_id is not None:
        header_lines.append(f"  query_id: {query_id}")
    if query is not None:
        header_lines.append(f"  query: {query}")
    header_lines.append("  Ground truth (normalized):")
    header_lines.append(_format_records(ground_truth))

    iterations = payload.get("retrieval_iterations", [])
    if not isinstance(iterations, list):
        header_lines.append("  retrieval_iterations: expected a list")
        return "\n".join(header_lines)

    for position, iteration in enumerate(iterations):
        if not isinstance(iteration, Mapping):
            header_lines.append(f"  Iteration {position + 1}: expected an object")
            continue
        iter_no = iteration.get("iteration")
        lineage = iteration.get("lineage_index")
        desc = iteration.get("current_description")
        header = "  Iteration"
        if iter_no is not None:
            header += f" {iter_no}"
        if lineage is not None:
            header += f" (lineage {lineage})"
        header_lines.append(header + ":")
        if desc:
            header_lines.append(f"    description: {desc}")
        retrieved = normalize_entries(iteration.get("retrieved_tools", []) or [])
        header_lines.append(_format_records(retrieved))

    return "\n".join(header_lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a sanity-check JSON file or a directory containing such files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    files = list(_iter_json_files(args.path))
    if not files:
        parser.error(f"No JSON files found under {args.path}")

    for idx, file_path in enumerate(files):
        if idx:
            print()
        print(summarize_file(file_path))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
