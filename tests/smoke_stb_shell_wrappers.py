"""Smoke tests for StableToolBench shell wrappers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "StableToolBench"))
sys.path.insert(0, str(REPO_ROOT / "StableToolBench/toolbench/inference"))


def test_run_inference_dry_run_redacts_keys() -> None:
    """Dry-run diagnostics must not print API keys from the caller environment."""
    openai_key = "test-openai-secret-value"
    toolbench_key = "test-toolbench-secret-value"
    env = dict(os.environ)
    env.update(
        {
            "OPENAI_API_KEY": openai_key,
            "TOOLBENCH_KEY": toolbench_key,
            "TOOL_ROOT_DIR": "data/toolenv/tools",
            "CORPUS_BASE_DIR": "data/retrieval/StableToolBench",
            "FITTEXT_RUNTIME_ROOT": "runs",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "StableToolBench/scripts/run_inference.sh"),
            "--strategy",
            "single_pass",
            "--dataset",
            "G1_instruction",
            "--no-planner",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert openai_key not in result.stdout
    assert toolbench_key not in result.stdout
    assert "<redacted-openai-key>" in result.stdout
    assert "<redacted-toolbench-key>" in result.stdout


def test_rapidapi_sanity_check_writes_under_output_dir(tmp_path: Path) -> None:
    """Retrieval sanity logs are runtime artifacts and must not land in source."""
    from toolbench.inference.Downstream_tasks.rapidapi import rapidapi_wrapper

    input_query = tmp_path / "release_smoke.json"
    input_query.write_text(
        '[{"query_id": 588, "query": "smoke", "relevant APIs": []}]',
        encoding="utf-8",
    )
    output_dir = tmp_path / "raw_output"
    args = SimpleNamespace(
        summarize_lineage_memory=False,
        lineage_summarizer="",
        dbd=False,
        refinement=False,
        dbd_refine_turns=0,
        disable_midexec_retrieval=False,
        refine_style="fittext",
        retrieve_mode="normal",
        retrieved_api_nums=5,
        scattershot=False,
        size=5,
        just_query=True,
        genetic=False,
        memetic=False,
        population_size=1,
        generation_num=1,
        similarity_threshold=0.95,
        memetic_toolret=False,
        base_temp=0.9,
        evolution_temperature=None,
        fitness_method="alpha",
        seed_model=None,
        refine_model=None,
        chatgpt_model="mock-model",
        base_url="http://localhost",
        openai_key="",
        method="DFS_woFilter_w2",
        input_query_file=str(input_query),
        output_answer_file=str(output_dir),
        tool_root_dir=str(tmp_path / "tools"),
        toolbench_key="",
        rapidapi_key="",
        use_rapidapi_key=False,
        api_customization=False,
        max_observation_length=1024,
        observ_compress_method="truncate",
    )

    wrapper = rapidapi_wrapper(
        query_json={"query": "smoke", "api_list": []},
        tool_descriptions=[],
        retriever=None,
        args=args,
        query_id=588,
    )
    wrapper.write_sanity_check(
        query_id=588,
        method="DFS_woFilter_w2",
        query="smoke",
        retrieval_iterations=[],
        instruction_file=str(input_query),
    )

    expected = output_dir / "sanity_check" / "588" / "sanity_588_DFS_woFilter_w2_release_smoke_retrieval0.json"
    assert expected.is_file()
    source_sanity = REPO_ROOT / "StableToolBench/toolbench/inference/Downstream_tasks/sanity_check"
    assert not source_sanity.exists()
