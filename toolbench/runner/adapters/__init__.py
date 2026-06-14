"""Per-benchmark eval adapters.

Each adapter exposes one coroutine ``run_<benchmark>(cfg, ...)`` that the
unified executor (:func:`toolbench.runner.executor.execute_run`) dispatches to.

Adapters are thin glue around the existing eval drivers (``Toolret/eval_toolret``,
``StableToolBench/toolbench/inference/qa_pipeline_multithread``) — they remap
the new ``RunConfig`` fields onto the legacy wrapper-attribute conventions
and inject the unified :class:`ManifestWriter` / :class:`BudgetGuard` /
``BeliefTracer`` factory.

Module map:

  - ``toolret.py``        — ToolRet (4 domains: code/customized/web/toolbench)
  - ``stabletoolbench.py`` — StableToolBench (G1/G2/G3 complexity levels)
"""
