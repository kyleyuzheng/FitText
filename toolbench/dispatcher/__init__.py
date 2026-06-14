"""toolbench.dispatcher — parallel benchmark dispatcher.

Fans out N (technique x baseline x benchmark x split x model) cells across
processes (ProcessPoolExecutor) and remote hosts (SSH). Tracks live status
in ``dispatch_status.json`` and aggregates results via
``toolbench.observability.aggregator``.

Public API
----------
- :func:`load_dispatch_spec` — load and validate a dispatch YAML.
- :func:`expand_cells` — expand a spec to a list of :class:`Cell` plans.
- :class:`DispatchExecutor` — execute cells locally + remotely with
  resumability and per-cell budget enforcement.
- :class:`StatusTracker` — JSON-on-disk state for monitor + resume.

Wave 2 integration: cells call ``run.py --config <generated.yaml>`` for
FitText variants, or ``scripts/run_baseline.py --baseline <name> --config
<generated.yaml>`` for baselines. Both already write ``result.json`` and
``manifest.jsonl`` per cell.
"""

from .spec import (
    DispatchSpec,
    DispatchModelEntry,
    DispatchBenchmarkEntry,
    DispatchParallelism,
    DispatchBudget,
    load_dispatch_spec,
)
from .plan import Cell, expand_cells, technique_to_variant
from .status import StatusTracker, CellStatus
from .executor import DispatchExecutor, run_cell_subprocess

__all__ = [
    "DispatchSpec",
    "DispatchModelEntry",
    "DispatchBenchmarkEntry",
    "DispatchParallelism",
    "DispatchBudget",
    "load_dispatch_spec",
    "Cell",
    "expand_cells",
    "technique_to_variant",
    "StatusTracker",
    "CellStatus",
    "DispatchExecutor",
    "run_cell_subprocess",
]
