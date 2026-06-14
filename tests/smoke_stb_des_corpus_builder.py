"""Smoke tests for the deterministic StableToolBench corpus builder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_build_des_corpus_from_toolenv(tmp_path: Path) -> None:
    """ToolEnv JSON files should produce G1/G2/G3 des_corpus.json files."""
    tool_root = tmp_path / "toolenv"
    category_dir = tool_root / "Data"
    category_dir.mkdir(parents=True)
    (category_dir / "sample_tool.json").write_text(
        json.dumps(
            {
                "tool_name": "Sample Tool",
                "tool_description": "Looks up sample records.",
                "api_list": [
                    {
                        "name": "Get Record",
                        "description": "Return one sample record.",
                        "required_parameters": [{"name": "record_id"}],
                        "optional_parameters": [{"name": "format"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "retrieval" / "StableToolBench"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "StableToolBench")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "StableToolBench/toolbench/retrieval/build_des_corpus_from_toolenv.py"),
            "--tool-root-dir",
            str(tool_root),
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for group in ("G1", "G2", "G3"):
        corpus_path = out_dir / group / "des_corpus.json"
        assert corpus_path.is_file()
        rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
        assert rows == [
            {
                "functionality": (
                    'API "Get Record" from tool "Sample Tool" in category "Data". '
                    "Tool description: Looks up sample records. "
                    "API description: Return one sample record. "
                    "Required parameters: record_id. "
                    "Optional parameters: format."
                ),
                "cate_name": "Data",
                "tool_name": "sample_tool",
                "api_name": "get_record",
            }
        ]
