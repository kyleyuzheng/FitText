"""Xu et al. 2024 baseline (arXiv 2406.17465, EMNLP 2024 Findings).

Paper: "Enhancing Tool Retrieval with Iterative Feedback from Large Language
        Models" (Qiancheng Xu, Yongqi Li, Heming Xia, Wenjie Li;
        Hong Kong PolyU; arXiv 2406.17465v2).
Code:  https://github.com/travis-xu/TR-Feedback

Relation to FitText: **Multi-Turn degenerate case with explicit critic chain**.
Maps to FitText (population_size=1, generations=G, selection=fitness) augmented
with an explicit three-step feedback chain replacing implicit belief revision.

Spec (paper §4.2 "Feedback Generation", paraphrased)
----------------------------------------------------
At each iteration t with current instruction ``q^t``:
    Retrieve top-K tools {d_1^t, ..., d_K^t} = R(q^t).

    Step 1: COMPREHENSION (Eq. 2)
        F_C = LLM(P_C, q^t, {d_i^t})
        Prompt P_C asks the LLM to:
          (a) summarize abstract user goals, ignoring detailed entity info;
          (b) understand retrieved tools' category/name/description/IO.

    Step 2: ASSESSMENT (Eq. 3)
        F_A = LLM(P_A, q^t, {d_i^t}, F_C)
        Prompt P_A asks the LLM to assess:
          (1) which user goals can / cannot be solved by retrieved tools;
          (2) whether ranking order corresponds to tool importance.

    Step 3: REFINEMENT (Eq. 4)
        F_R = LLM(P_R, q^t, {d_i^t}, F_A)
        Prompt P_R asks the LLM to decide whether refinement is needed:
          - If ALL goals solved AND ALL appropriate tools top-ranked,
            return special token "N/A" (no refinement; convergence).
          - Else generate refined instruction q^{t+1} with:
              (i) detail/personalization on unsolved intents;
              (ii) scenario-specific tool-usage info for appropriate tools.

Iterate until "N/A" emitted or max_iterations reached.

Published headline (Table 2, ToolBench, nDCG@5):
    Iter=2 with TAS-B base retriever: I1 0.6235 / I2 0.4849 / I3 0.5681
    (consult paper Table 2 for the full grid; deeper iters give marginal gains)

Note: the paper also proposes "Iteration-Aware Feedback Training" (§4.3), a
*trained* retriever variant.  Our implementation is TRAINING-FREE per the
FitText eval protocol -- we only use the iterative inference chain (§4.2).

Design decisions in this implementation
---------------------------------------
- We collapse the 3 LLM calls per iteration into a single chain.  The
  Comprehension / Assessment / Refinement are 3 sequential LLM calls per
  iteration as written in Eqs. 2-4; we keep them separate to be faithful
  to the paper's wording.  An ``allow_collapsed_chain=True`` option exists
  for a cheaper 1-call variant for ablations.
- Convergence: we accept either "N/A" (paper's special token, §4.2) or an
  unchanged retrieved-set (Jaccard = 1.0) as early-stop signals.  The
  latter is defensive belt-and-suspenders for LLMs that don't emit N/A
  reliably.
- Iteration-Aware Training (§4.3, Eq. 5) is NOT implemented: we
  evaluate training-free baselines only.  Documented as
  a deviation in REPRODUCTION.md.

Verified faithfulness against paper: 2026-05-24 (baseline-verify track).
See ``baselines/REPRODUCTION.md`` for the deviation log and
``tests/test_baseline_reproduction.py`` for the spot-check.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar

from .base import Baseline, RetrievedTools, RetrieverAdapter
from StableToolBench.toolbench.inference.LLM.clients import ModelClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates (paper §4.2; paraphrased from the Eq. 2-4 specs)
# ---------------------------------------------------------------------------

# Step 1: COMPREHENSION (Eq. 2)
_COMPREHENSION_SYSTEM = (
    "You are a tool-retrieval comprehension assistant. Given a user instruction "
    "and a list of retrieved tools, you must produce a two-part comprehension: "
    "(1) summarize the user's abstract goals, ignoring specific entity/detail "
    "information; (2) understand each retrieved tool's functionality, focusing "
    "on category, name, description, input parameters, and output parameters. "
    "Be precise and concise."
)
_COMPREHENSION_USER = (
    "User instruction: {query}\n\n"
    "Retrieved tools:\n{tool_block}\n\n"
    "Provide the two-part comprehension:"
)

# Step 2: ASSESSMENT (Eq. 3)
_ASSESSMENT_SYSTEM = (
    "You are a tool-retrieval assessment assistant. Given the user instruction, "
    "the retrieved tools, and a prior comprehension, you must assess the "
    "retrieval quality from two perspectives: "
    "(1) Which of the user's goals CAN and CANNOT be solved by the retrieved "
    "tools? Give specific reasons for each. "
    "(2) Does the ranking order of retrieved tools correspond to their "
    "significance in addressing the user's intent? Give specific reasons."
)
_ASSESSMENT_USER = (
    "User instruction: {query}\n\n"
    "Retrieved tools:\n{tool_block}\n\n"
    "Prior comprehension:\n{comprehension}\n\n"
    "Provide the two-perspective assessment:"
)

# Step 3: REFINEMENT (Eq. 4)
_REFINEMENT_SYSTEM = (
    "You are a tool-retrieval refinement assistant. Given the user instruction, "
    "the retrieved tools, and a prior assessment, decide whether the user "
    "instruction should be refined to improve retrieval. "
    "Answer 'N/A' (exactly two characters plus slash) if AND ONLY IF both: "
    "(a) all user goals are solved by the currently retrieved tools, AND "
    "(b) all appropriate tools are given the highest ranking priorities. "
    "Otherwise, output a refined user instruction with: "
    "(1) more detailed / personalized content about user intents not yet solved, "
    "helping the retriever explore other relevant tools; "
    "(2) scenario-specific tool-usage information for the appropriate tools "
    "found so far, helping the retriever rank them higher. "
    "Output ONLY 'N/A' or ONLY the refined instruction text -- no preamble."
)
_REFINEMENT_USER = (
    "Original user instruction: {query}\n\n"
    "Retrieved tools:\n{tool_block}\n\n"
    "Prior assessment:\n{assessment}\n\n"
    "Output 'N/A' or the refined instruction:"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_block(
    tool_ids: list[str],
    tool_catalog: list[dict],
    max_desc_chars: int = 200,
) -> str:
    """Format retrieved tool IDs as a human-readable block for the LLM prompts.

    Args:
        tool_ids: List of composite tool_id strings.
        tool_catalog: Full catalog for description lookup.
        max_desc_chars: Max characters per description (truncate).

    Returns:
        Multi-line "- tool_name: description" block.
    """
    desc_map: dict[str, str] = {}
    for tool in tool_catalog:
        aname = tool.get("api_name") or tool.get("name") or ""
        desc = tool.get("description") or ""
        if aname:
            desc_map[aname] = desc[:max_desc_chars]

    lines: list[str] = []
    for tid in tool_ids:
        parts = tid.split("::")
        api_name = parts[2] if len(parts) >= 3 else tid
        desc = desc_map.get(api_name, "(no description)")
        lines.append(f"- {api_name}: {desc}")
    return "\n".join(lines) if lines else "(no tools retrieved)"


def _is_na_token(text: str) -> bool:
    """Detect paper's 'N/A' convergence signal (§4.2 Refinement)."""
    t = text.strip().upper()
    # Accept "N/A", "NA", or a leading N/A token in a 1-2 word answer.
    if t in ("N/A", "NA"):
        return True
    # Tolerate short answers like "N/A." or "N/A - no refinement needed"
    if t.startswith("N/A") and len(t) < 40:
        return True
    return False


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class Xu2024Baseline(Baseline):
    """Iterative Feedback Tool Retrieval (Xu et al. 2024, arXiv 2406.17465).

    Implements the 3-step Comprehension/Assessment/Refinement chain from
    paper §4.2 (Eqs. 2-4).  Training-free -- §4.3 Iteration-Aware Training
    is intentionally NOT implemented (training-free baselines only).

    Each iteration ``t``:
      1. Retrieve top-K tools for current query q^t.
      2. Comprehension LLM call -> F_C  (Eq. 2)
      3. Assessment LLM call    -> F_A  (Eq. 3)
      4. Refinement LLM call    -> F_R / 'N/A' (Eq. 4)
      5. If F_R == 'N/A' OR retrieved set unchanged: converged.  Else
         q^{t+1} = F_R; continue.

    Config kwargs
    -------------
    max_iterations : int, default 3
        Max C/A/R rounds (paper sweeps 1-3 in Table 2; gains plateau at 2-3).
    temperature : float, default 0.0
    seed : int, default 42
    max_tokens : int, default 384
        Comprehension and assessment outputs can be long.
    allow_collapsed_chain : bool, default False
        If True, collapse C/A/R into a single LLM call.  Used for cheap
        ablations; faithful path uses three separate calls.
    """

    name: ClassVar[str] = "xu2024"

    def __init__(
        self,
        model_client: ModelClient,
        retriever: Any,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_client, retriever, top_k=top_k, **kwargs)
        self._max_iterations: int = int(self.cfg.get("max_iterations", 3))
        self._collapsed: bool = bool(self.cfg.get("allow_collapsed_chain", False))

    async def retrieve(self, query: str, *, tool_catalog: list[dict]) -> RetrievedTools:
        """Iterative C/A/R retrieval chain (paper §4.2).

        Args:
            query: Initial user task query.
            tool_catalog: Full tool list from harness.

        Returns:
            RetrievedTools from the converging iteration.  Metadata contains
            the full trace with per-iter comprehension, assessment, and
            refinement (or 'N/A') outputs -- for the belief-instrumentation
            track.
        """
        t0 = time.monotonic()
        current_query = query
        prev_tool_ids: list[str] | None = None
        trace: list[dict[str, Any]] = []

        last_tool_ids: list[str] = []
        last_scores: list[float] = []

        for iteration in range(self._max_iterations):
            # --- Retrieve with current query --------------------------------
            hits = self.retriever.retrieve(current_query, top_k=self.top_k)
            tool_ids = [h[0] for h in hits]
            scores = [h[1] for h in hits]
            last_tool_ids, last_scores = tool_ids, scores

            entry: dict[str, Any] = {
                "iteration": iteration,
                "query": current_query,
                "tool_ids": list(tool_ids),
                "scores": list(scores),
            }

            # --- Set-stable convergence (belt-and-suspenders) --------------
            if prev_tool_ids is not None and set(tool_ids) == set(prev_tool_ids):
                entry["converged"] = True
                entry["convergence_reason"] = "stable_set"
                trace.append(entry)
                return RetrievedTools(
                    tool_ids=tool_ids,
                    scores=scores,
                    metadata={
                        "trace": trace,
                        "iterations_run": iteration + 1,
                        "converged": True,
                        "convergence_reason": "stable_set",
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    },
                )

            prev_tool_ids = list(tool_ids)

            # --- Final iteration: no further LLM work ----------------------
            if iteration == self._max_iterations - 1:
                entry["final"] = True
                trace.append(entry)
                break

            # --- C/A/R chain (paper Eqs. 2-4) ------------------------------
            tool_blk = _tool_block(tool_ids, tool_catalog)

            # Step 1: COMPREHENSION (Eq. 2)
            comp_resp = await self.model_client.chat_completion(
                messages=[
                    {"role": "system", "content": _COMPREHENSION_SYSTEM},
                    {"role": "user",
                     "content": _COMPREHENSION_USER.format(
                         query=current_query, tool_block=tool_blk
                     )},
                ],
                temperature=float(self.cfg.get("temperature", 0.0)),
                seed=int(self.cfg.get("seed", 42)),
                max_tokens=int(self.cfg.get("max_tokens", 384)),
            )
            comprehension = (comp_resp.content or "").strip()
            entry["comprehension"] = comprehension
            entry["comp_tokens"] = {
                "input": comp_resp.input_tokens,
                "output": comp_resp.output_tokens,
            }

            # Step 2: ASSESSMENT (Eq. 3)
            assess_resp = await self.model_client.chat_completion(
                messages=[
                    {"role": "system", "content": _ASSESSMENT_SYSTEM},
                    {"role": "user",
                     "content": _ASSESSMENT_USER.format(
                         query=current_query,
                         tool_block=tool_blk,
                         comprehension=comprehension,
                     )},
                ],
                temperature=float(self.cfg.get("temperature", 0.0)),
                seed=int(self.cfg.get("seed", 42)),
                max_tokens=int(self.cfg.get("max_tokens", 384)),
            )
            assessment = (assess_resp.content or "").strip()
            entry["assessment"] = assessment
            entry["assess_tokens"] = {
                "input": assess_resp.input_tokens,
                "output": assess_resp.output_tokens,
            }

            # Step 3: REFINEMENT (Eq. 4)
            refine_resp = await self.model_client.chat_completion(
                messages=[
                    {"role": "system", "content": _REFINEMENT_SYSTEM},
                    {"role": "user",
                     "content": _REFINEMENT_USER.format(
                         query=current_query,
                         tool_block=tool_blk,
                         assessment=assessment,
                     )},
                ],
                temperature=float(self.cfg.get("temperature", 0.0)),
                seed=int(self.cfg.get("seed", 42)),
                max_tokens=int(self.cfg.get("max_tokens", 256)),
            )
            refinement = (refine_resp.content or "").strip()
            entry["refinement"] = refinement
            entry["refine_tokens"] = {
                "input": refine_resp.input_tokens,
                "output": refine_resp.output_tokens,
            }

            # --- N/A early-stop (paper §4.2 Refinement) --------------------
            if _is_na_token(refinement):
                entry["converged"] = True
                entry["convergence_reason"] = "na_token"
                trace.append(entry)
                return RetrievedTools(
                    tool_ids=tool_ids,
                    scores=scores,
                    metadata={
                        "trace": trace,
                        "iterations_run": iteration + 1,
                        "converged": True,
                        "convergence_reason": "na_token",
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    },
                )

            entry["reformulated_query"] = refinement
            trace.append(entry)
            current_query = refinement

        # Final retrieve if we exited the loop without explicit convergence
        if last_tool_ids:
            tool_ids = last_tool_ids
            scores = last_scores
        else:
            hits = self.retriever.retrieve(current_query, top_k=self.top_k)
            tool_ids = [h[0] for h in hits]
            scores = [h[1] for h in hits]

        return RetrievedTools(
            tool_ids=tool_ids,
            scores=scores,
            metadata={
                "trace": trace,
                "iterations_run": self._max_iterations,
                "converged": False,
                "convergence_reason": "max_iter",
                "final_query": current_query,
                "latency_ms": (time.monotonic() - t0) * 1000.0,
            },
        )
