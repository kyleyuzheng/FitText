# Model Pin Policy

## The one rule

**`configs/model_pins.yaml` is the only place model identifiers live.**
Every script, CLI default, and config that needs a model name must read from it.
Hardcoded duplicates of a pinned value are bugs.

## Cardinal constraint: never change the judge when you change the agent

The STB pass-rate judge (`evaluators.judge`) and tool-simulator (`evaluators.simulator`)
are LLM-based and stochastic.  Changing them mid-experiment invalidates cross-cell
comparisons — the evaluation surface shifts.  **Fix the judge for the entire ablation
matrix; only change it between distinct experimental rounds and document it explicitly.**

## How to bump a model pin

1. Edit the relevant entry in `configs/model_pins.yaml`.
2. Use a provider-specific stable tag, never a floating alias.
   - Good: a dated or revisioned model identifier in `configs/model_pins.yaml`
   - Bad: a floating alias such as a family name or `latest`
3. If bumping an **agent** model (`agents.*`):
   - Re-run the reproduce-paper baseline (`configs/runs/reproduce_paper.yaml`) to confirm
     the new model produces comparable numbers before launching the full ablation matrix.
   - Update the `extra.note` field in affected run YAMLs to document the change.
4. If bumping the **judge or simulator** (`evaluators.*`):
   - Regenerate derived cost artifacts under `FITTEXT_ANALYSIS_ROOT` — judge costs flow into the overall budget.
   - Run `scripts/check_pin_consistency.py` to verify nothing hardcodes the old tag.
   - All prior eval results are now on a different judge surface — treat them as a
     separate experimental condition, not a fair comparison.
5. Run `python tests/smoke_pin_consistency.py` — must exit 0.
6. Run `python tests/smoke_runner.py` — must exit 0.
7. Commit with a message that names the old and new tags and the reason for the bump.

## Verification checklist after any pin change

- [ ] `python scripts/check_pin_consistency.py` exits 0
- [ ] `python tests/smoke_pin_consistency.py` exits 0
- [ ] `python tests/smoke_runner.py` exits 0
- [ ] No floating alias anywhere in tracked `.py` / `.yaml` / `.sh` / `.json` files

## Where pins are consumed

| Consumer | Key used |
|---|---|
| `StableToolBench/toolbench/inference/LLM/chatgpt_function_model.py` | `agents.gpt_4_1_mini` |
| `Toolret/strategy/LLM_model.py` | `agents.gpt_4_1_mini` |
| `StableToolBench/toolbench/inference/qa_pipeline*.py` | `agents.gpt_4_1_mini`, `agents.o3_mini` |
| `Toolret/eval_toolret.py` | `agents.gpt_4_1`, `agents.gpt_4_1_mini` |
| `Toolret/data_preprocess/build_des_corpus.py` | `agents.gpt_4_1_mini` |
| `configs/_base/evaluators.yaml` | `evaluators.judge`, `evaluators.simulator` |
| `configs/_base/embedder.yaml` | `embedders.toolret` |
| STB tooleval `evaluators/*/config.yaml` | matches `evaluators.judge` directly |
