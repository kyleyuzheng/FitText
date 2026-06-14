#!/usr/bin/env python3
"""Build StableToolBench description corpora directly from ToolEnv JSON files.

This deterministic builder is intended for release smoke runs when the original
ToolBench retrieval TSV is not staged locally. For exact paper reproduction,
prefer the official ToolBench retrieval corpus plus ``build_des_corpus.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from toolbench.utils import change_name, standardize


DEFAULT_GROUPS = ("G1", "G2", "G3")


def _iter_tool_files(tool_root: Path) -> Iterable[Path]:
    return sorted(path for path in tool_root.rglob("*.json") if path.is_file())


def _param_names(params: object) -> str:
    if not isinstance(params, list):
        return ""
    names = []
    for param in params:
        if isinstance(param, dict) and param.get("name"):
            names.append(str(param["name"]))
    return ", ".join(names)


def _functionality(
    *,
    category: str,
    tool_name_raw: str,
    tool_description: object,
    api_name_raw: str,
    api_description: object,
    required_parameters: object,
    optional_parameters: object,
) -> str:
    parts = [
        f'API "{api_name_raw}" from tool "{tool_name_raw}" in category "{category}".',
    ]
    if tool_description:
        parts.append(f"Tool description: {tool_description}")
    if api_description:
        parts.append(f"API description: {api_description}")
    required = _param_names(required_parameters)
    optional = _param_names(optional_parameters)
    if required:
        parts.append(f"Required parameters: {required}.")
    if optional:
        parts.append(f"Optional parameters: {optional}.")
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def build_records(tool_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in _iter_tool_files(tool_root):
        category = path.parent.name
        with path.open(encoding="utf-8") as fh:
            tool = json.load(fh)
        tool_name_raw = str(tool.get("tool_name") or tool.get("name") or path.stem)
        tool_name = standardize(tool_name_raw)
        tool_description = tool.get("tool_description") or ""
        for api in tool.get("api_list") or []:
            if not isinstance(api, dict):
                continue
            api_name_raw = str(api.get("name") or "")
            if not api_name_raw:
                continue
            api_name = change_name(standardize(api_name_raw))
            records.append(
                {
                    "functionality": _functionality(
                        category=category,
                        tool_name_raw=tool_name_raw,
                        tool_description=tool_description,
                        api_name_raw=api_name_raw,
                        api_description=api.get("description") or "",
                        required_parameters=api.get("required_parameters") or [],
                        optional_parameters=api.get("optional_parameters") or [],
                    ),
                    "cate_name": category,
                    "tool_name": tool_name,
                    "api_name": api_name,
                }
            )
    records.sort(key=lambda item: (item["cate_name"], item["tool_name"], item["api_name"]))
    return records


def write_group_corpora(records: list[dict[str, str]], output_dir: Path, groups: Iterable[str]) -> None:
    for group in groups:
        group_dir = output_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        out_path = group_dir / "des_corpus.json"
        with out_path.open("w", encoding="utf-8") as fh:
            for record in records:
                json.dump(record, fh, ensure_ascii=False)
                fh.write("\n")
        print(f"{out_path}: {len(records)} records")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build G1/G2/G3 des_corpus.json files from a ToolEnv tools root."
    )
    parser.add_argument(
        "--tool-root-dir",
        type=Path,
        required=True,
        help="Directory containing ToolEnv category folders with tool JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where G1/G2/G3 des_corpus.json files will be written.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Corpus group directories to write. Defaults to G1 G2 G3.",
    )
    args = parser.parse_args()

    tool_root = args.tool_root_dir.expanduser().resolve()
    if not tool_root.is_dir():
        raise SystemExit(f"tool root does not exist: {tool_root}")
    records = build_records(tool_root)
    if not records:
        raise SystemExit(f"no API records found under {tool_root}")
    write_group_corpora(records, args.output_dir.expanduser().resolve(), args.groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
