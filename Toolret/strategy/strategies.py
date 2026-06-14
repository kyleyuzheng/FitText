"""
All the strategies for dynamic tool retrieval: single-pass, DBD, Scattershot, and just_query.

Abstractions:
rapidapi.py <--- strategies.py <--- services.py <--- prompts.py
"""
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from .tokens import FUNC_DESC_PATTERN
from .services import (
    build_refine_or_regen_messages,
    FORMAT_INSTRUCTIONS_SYSTEM,
    FORMAT_INSTRUCTIONS_USER
)
from .prompts import build_scattershot_seeded_messages
from .LLM_model import ChatGPTFunction
import json


class strategy_wrapper():
    def __init__(self, strategy: str,
                 retriever,
                 retrieved_api_nums: int,
                 example_num: int,
                 dbd_refine_turns: int = 0,
                 scattershot_size: int = 5,
                 refinement: bool = False,
                 api_key: str = "",):
        self.strategy = strategy
        self.retriever = retriever
        self.retrieved_api_nums = retrieved_api_nums
        self.example_num = example_num
        self.dbd_refine_turns = dbd_refine_turns
        self.scattershot_size = scattershot_size
        self.api_key = api_key
        self.refinement = refinement
        self.input_description = ""


# ---------- helpers ----------

def _only_blocks(text: str) -> List[str]:
    return [m.strip() for m in FUNC_DESC_PATTERN.findall(text or "") if m and m.strip()]

def _extract_single_block(text: str):
    matches = FUNC_DESC_PATTERN.findall(text or "")
    if not matches:
        return None
    for m in matches:
        s = (m or "").strip()
        if s:
            return s
    return None

# ---------- single-pass ----------

def run_single_pass_strategy(llm_output: str, wrapper: strategy_wrapper) -> Tuple[List[Dict[str, Any]], List[dict]]:
    descriptions = [m.strip() for m in FUNC_DESC_PATTERN.findall(llm_output) if m.strip()]
    retrieval_summary: List[dict] = []
    api_keys: List[Dict[str, Any]] = []

    for i, desc in enumerate(descriptions):
        tool_ids, tool_des, tool_scores = wrapper.retriever.retrieving(desc, wrapper.retrieved_api_nums)
        retrieval_summary.append({
            "lineage_index": i,
            "current_description": desc,
            "retrieved_tools": tool_ids,
        })
        api_keys.append({
            'tool_ids': tool_ids,
            'tool_descriptions': tool_des,
            'tool_scores': tool_scores,
            'lineage_index': i,
        })

    return api_keys, retrieval_summary

# ---------- DBD ----------

def run_dbd_strategy(llm_output: str, wrapper: strategy_wrapper, llm) -> Tuple[List[Dict[str, Any]], List[dict]]:
    ancestors = [m.strip() for m in FUNC_DESC_PATTERN.findall(llm_output) if m.strip()]
    retrieval_iterations: List[dict] = []
    if not ancestors:
        return [], retrieval_iterations

    turns = max(0, getattr(wrapper, "dbd_refine_turns", 0))
    final_descs: List[str] = []

    for lineage_index, ancestor in enumerate(ancestors):
        current = ancestor
        examples: List[str] = []
        seen = set()
        retrieval_iteration = {}
        retrieval_iteration['ancestor_description'] = ancestor
        retrieval_iteration['lineage_index'] = lineage_index

        if turns == 0:
            final_descs.append(current)
            continue

        for t in range(1, turns + 1):
            tool_ids, tool_des, tool_scores = wrapper.retriever.retrieving(current, wrapper.example_num)
            retrieval_iteration[t] = {
                "current_description": current,
                "retrieved_tools": tool_ids,
                "retrieved_tools_scores": tool_scores,
            }
            for id, des in zip(tool_ids, tool_des):
                if id not in seen:
                    seen.add(id); examples.append({"tool_id": id, "tool_des": des})

            msgs = build_refine_or_regen_messages(
                mode="dbd",
                query=wrapper.input_description,
                refinement=getattr(wrapper, "refinement", False),
                ancestor_desc=ancestor,
                current_desc=current,
                lineage_examples=examples,
                turn=t,
            )
            msg, _, _ = llm.parse_with_messages(messages=msgs, tools=[], process_id=getattr(wrapper, "process_id", 0))
            content = (msg or {}).get("content", "") or ""
            next_desc = _extract_single_block(content)
            if next_desc:
                current = next_desc
            
        retrieval_iterations.append(retrieval_iteration)
        final_descs.append(current)

    # final retrieval on stabilized lineages
    api_keys: List[Dict[str, Any]] = []
    for i, desc in enumerate(final_descs):
        tool_ids, tool_des, tool_scores = wrapper.retriever.retrieving(desc, wrapper.retrieved_api_nums)
        retrieval_iterations.append({
            "iteration": "final",
            "lineage_index": i,
            "current_description": desc,
            "retrieved_tools": tool_ids,
            "retrieved_tools_scores": tool_scores
        })
        api_keys.append({
            'tool_ids': tool_ids,
            'tool_descriptions': tool_des,
            'tool_scores': tool_scores,
            'lineage_index': i,
        })

    return api_keys, retrieval_iterations

# ---------- Scattershot ----------

def run_scattershot_strategy(llm_output: str, wrapper: strategy_wrapper, llm) -> Tuple[List[Dict[str, Any]], List[dict], Dict[str, Any]]:
    """
    Per-lineage Scattershot:
      • Seed retrieval per ancestor (local exemplars) -> qj_seed (for allowed keys) & mt_seed (for logs)
      • Generate 'size' children; retrieve each with top_k = retrieved_api_nums
      • Vote using child mt lists (rank/score), but only allow winners present in the combined seed/child qj api_lists
      • Tie-break: votes desc, avg rank asc, avg score desc
      • Truncate to retrieved_api_nums winners per lineage
      • Return resolvable API keys + trace
    """
    from itertools import chain

    retrieval_iterations: List[dict] = []
    size = max(1, int(getattr(wrapper, "scattershot_size", 5)))
    top_k_child = int(getattr(wrapper, "retrieved_api_nums", 15))

    ancestors = _only_blocks(llm_output)
    if not ancestors:
        payload = {
            "strategy": "scattershot",
            "lineages": [],
        }
        return [], retrieval_iterations, payload

    api_keys: List[Dict[str, Any]] = []

    # seed per ancestor
    exemplars_per_ancestor: List[Dict[str, Any]] = []

    for ai, ancestor in enumerate(ancestors):
        tool_ids, tool_des, tool_scores = wrapper.retriever.retrieving(ancestor, wrapper.example_num)
        retrieval_iterations.append({
            "phase": "seed_retrieval",
            "ancestor_index": ai,
            "ancestor_description": ancestor,
            "retrieved_tools": tool_ids,
        })
        exemplars_per_ancestor.append({"tools_id": tool_ids, "tools_des": tool_des})

    # children per lineage
    lineage_summaries: List[Dict[str, Any]] = []

    for ai, (ancestor, exemplar_blurbs) in enumerate(
        zip(ancestors, exemplars_per_ancestor)
    ):
        votes: Dict[str, int] = defaultdict(int)
        ranks: Dict[str, List[int]] = defaultdict(list)
        scores: Dict[str, List[float]] = defaultdict(list)
        tools_descriptions: Dict[str, str] = {}

        # Resolvable winners must come from the combined seed/child qj api_lists.
        allowed: set[tuple] = set()

        exemplar_block = "\n".join(
            (f"{i}. {s.strip()}" if i == 1
            else f"        {i}. {s.strip()}")
            for i, s in enumerate(exemplar_blurbs['tools_des'], 1)
        )

        child_descriptions: List[str] = []

        for si in range(size):
            msgs = build_scattershot_seeded_messages(
                query=wrapper.input_description,
                ancestor_desc=ancestor,
                exemplar_block=exemplar_block,
            )
            msg, _, _ = llm.parse_with_messages(
                messages=msgs, tools=[], process_id=0, temperature=0.9
            )
            child_desc = _extract_single_block((msg or {}).get("content", ""))
            if not child_desc:
                retrieval_iterations.append({
                    "phase": "child_gen_empty",
                    "ancestor_index": ai,
                    "sample_index": si,
                })
                continue

            tool_ids, tool_des, tool_scores = wrapper.retriever.retrieving(child_desc, top_k_child)
            retrieval_iterations.append({
                "phase": "child_retrieval",
                "ancestor_index": ai,
                "sample_index": si,
                "child_description": child_desc,
                "retrieved_tools": tool_ids,
            })
            child_descriptions.append(child_desc)
            for it, desc in zip(tool_ids, tool_des):
                allowed.add(it)
                tools_descriptions[it] = desc

            # vote using ranked mt_child for signals
            for rank_idx, (it, score) in enumerate(zip(tool_ids, tool_scores)):
                key = it
                votes[key] += 1
                ranks[key].append(rank_idx)
                scores[key].append(score)

        def _avg(xs, default):
            xs2 = [v for v in xs if v is not None]
            return (sum(xs2) / len(xs2)) if xs2 else default

        def _sort_key(k):
            return (-votes[k], _avg(ranks[k], 9999.0), -_avg(scores[k], -1e9))

        lineage_keys = sorted(votes.keys(), key=_sort_key)
        # keep only resolvable winners
        lineage_keys = [k for k in lineage_keys if k in allowed]
        # truncate
        top_keys = lineage_keys[:top_k_child]

        api_keys.append({
            "tool_ids": top_keys,
            "tool_descriptions": [tools_descriptions[k] for k in top_keys],
            "tool_scores": [_avg(scores[k], 0) for k in top_keys],
            "lineage_index": ai,
        })

        lineage_summaries.append({
            "lineage_index": ai,
            "ancestor": ancestor,
            "generated_children": child_descriptions,
            "winning_keys": top_keys,
            "votes": votes,
            "ranks": ranks,
        })

    return api_keys, retrieval_iterations, lineage_summaries

# ---------- just_query (zero-retrieval floor baseline) ----------

def run_just_query_strategy(query: str, wrapper: strategy_wrapper, llm) -> Tuple[List[str], List[str], List[float], Dict[str, Any]]:
    """Zero-retrieval floor baseline (ToolRet mirror) — direct LLM answer, no tools.

    Asks the LLM the query directly with no retrieval, no tool catalog, and no
    second pass. Measures the floor of what prior knowledge alone can solve.
    The empty ``tool_ids`` list is the point — no retrieval is happening.

    Args:
        query: The user query (also assigned to ``wrapper.input_description``).
        wrapper: The strategy wrapper. ``input_description`` and any
            ``process_id`` attr are read.
        llm: An initialized ``ChatGPTFunction`` (or compatible) LLM client.

    Returns:
        Four-tuple matching the ToolRet ``select_and_run_strategy`` contract,
        plus an extra ``payload`` for downstream eval/judging:
          - tool_ids:           []   (zero-retrieval floor)
          - tool_descriptions:  []
          - tool_scores:        []
          - payload:            {"strategy": "just_query", "answer": <text>}
    """
    messages = [
        {"role": "system",
         "content": "Answer the user's question directly to the best of your "
                    "knowledge. Do not call any tools."},
        {"role": "user",
         "content": f"Answer the following question: {query}"},
    ]
    msg, _, _ = llm.parse_with_messages(
        messages=messages,
        tools=[],
        process_id=getattr(wrapper, "process_id", 0),
    )
    answer = (msg or {}).get("content", "") or ""
    payload: Dict[str, Any] = {"strategy": "just_query", "answer": answer}
    return [], [], [], payload


# ---------- router ----------

def select_and_run_strategy(query: str, wrapper: strategy_wrapper, plan_llm_model, refine_llm_model, detailed_result_path, base_url_plan=None, base_url_refine=None) -> Tuple[List[Dict[str, Any]], List[dict]]:
    wrapper.input_description = query

    if wrapper.strategy == "just_query":
        # Zero-retrieval floor: single LLM call, no tool catalog, no retrieval.
        # Uses refine_llm_model as the answerer (planner is irrelevant when
        # there's no plan-then-retrieve loop).
        answer_llm = ChatGPTFunction(
            model=refine_llm_model, openai_key=wrapper.api_key, base_url=base_url_refine
        )
        tool_ids, tool_des, tool_scores, payload = run_just_query_strategy(
            query, wrapper, answer_llm
        )
        if detailed_result_path:
            with open(detailed_result_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        return tool_ids, tool_des, tool_scores

    # Initialize the LLM
    plan_llm = ChatGPTFunction(model=plan_llm_model, openai_key=wrapper.api_key, base_url=base_url_plan)
    system_msg = []
    system_msg.append({"role": "system", "content": FORMAT_INSTRUCTIONS_SYSTEM})
    user_msg = FORMAT_INSTRUCTIONS_USER.replace("{input_description}", query)
    system_msg.append({"role": "user", "content": user_msg})

    llm_output, _, _ = plan_llm.parse_with_messages(messages=system_msg, tools=[])
    llm_output = llm_output['content']

    refine_llm = ChatGPTFunction(model=refine_llm_model, openai_key=wrapper.api_key, base_url=base_url_refine)

    if wrapper.strategy == "dbd":
        api_keys, retrieval_iterations = run_dbd_strategy(llm_output, wrapper, refine_llm)
    elif wrapper.strategy == "scattershot":
        api_keys, _ , lineage_summaries = run_scattershot_strategy(llm_output, wrapper, refine_llm)
    elif wrapper.strategy == "single_pass":
        api_keys, retrieved_summary = run_single_pass_strategy(llm_output, wrapper)

    retrieved_tool_ids = []
    retrieved_tool_descriptions = []
    retrieved_tool_scores = []

    for keys in api_keys:
        retrieved_tool_ids.extend(tool_id for tool_id in keys['tool_ids'])
        retrieved_tool_descriptions.extend(tool_desc for tool_desc in keys['tool_descriptions'])
        retrieved_tool_scores.extend(tool_score for tool_score in keys['tool_scores'])

    if wrapper.strategy == 'dbd':
        with open(detailed_result_path, "a") as f:
            f.write(json.dumps(retrieval_iterations) + "\n")
    elif wrapper.strategy == "scattershot":
        with open(detailed_result_path, "a") as f:
            f.write(json.dumps(lineage_summaries) + "\n")
    elif wrapper.strategy == "single_pass":
        with open(detailed_result_path, "a") as f:
            f.write(json.dumps(retrieved_summary) + "\n")

    return retrieved_tool_ids, retrieved_tool_descriptions, retrieved_tool_scores
