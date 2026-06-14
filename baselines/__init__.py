"""
Retrieval baselines for FitText.

Each baseline maps onto a degenerate case of FitText's evolutionary
retrieval (see §0 of EXECUTION_PLAN.md):

  less_is_more  -> Single-Pass degenerate (N=1, G=1, no revision)
  reinvoke      -> index-time synth-query expansion + multi-intent retrieval
  xu2024        -> Multi-Turn degenerate + explicit critic prompt
  colt          -> wrapper for quchangle1/COLT (dual-encoder + GCN)
  just_query    -> zero-retrieval floor (no retrieval, no tool catalog) — §5.6

All conform to the ``Baseline`` ABC defined in ``base.py``.
"""

from .base import Baseline, RetrievedTools, Retriever
from .less_is_more import LessIsMoreBaseline
from .reinvoke import ReInvokeBaseline
from .xu2024 import Xu2024Baseline
from .colt import COLTBaseline
from .just_query import JustQueryBaseline

__all__ = [
    "Baseline",
    "RetrievedTools",
    "Retriever",
    "LessIsMoreBaseline",
    "ReInvokeBaseline",
    "Xu2024Baseline",
    "COLTBaseline",
    "JustQueryBaseline",
]
