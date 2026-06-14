"""Config-driven runner for FitText experiments."""

from .schema import RunConfig, ModelSpec, FitTextHParams, BenchmarkSpec
from .schema import EvaluatorSpec, EmbedderSpec, InfraSpec, BudgetSpec
from .resolver import resolve_config, dump_resolved_config

# Executor depends on toolbench.observability (lives in StableToolBench/).
# That package is only on sys.path when conftest.py runs (tests) or when
# run.py prepends it. Tolerate the absence so that downstream consumers
# which only need schema + resolver (e.g. config tooling) still load.
try:
    from .executor import execute_run, ExecutionReport, aggregate_manifest
except ModuleNotFoundError:  # pragma: no cover — only triggers without StableToolBench on path
    execute_run = None  # type: ignore[assignment]
    ExecutionReport = None  # type: ignore[assignment]
    aggregate_manifest = None  # type: ignore[assignment]

__all__ = [
    "RunConfig",
    "ModelSpec",
    "FitTextHParams",
    "BenchmarkSpec",
    "EvaluatorSpec",
    "EmbedderSpec",
    "InfraSpec",
    "BudgetSpec",
    "resolve_config",
    "dump_resolved_config",
    "execute_run",
    "ExecutionReport",
    "aggregate_manifest",
]
