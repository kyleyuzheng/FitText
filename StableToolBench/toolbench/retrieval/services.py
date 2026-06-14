"""
Helper functions for various strategies.py used in dynamic tool retrieval.
"""
# toolbench/inference/retrieval/services.py
from typing import List, Dict, Any, Literal
from toolbench.inference.LLM.prompts import *
from toolbench.inference.LLM.tokens import FUNC_DESC_PATTERN

import os

Mode = Literal["dbd"]

def retrieve_rapidapi_tools(retriever, query: str, top_k: int, tool_root_dir: str):
    """
    Runs `retriever.retrieving(...)` and resolves only those entries whose
    category/tool JSON files actually exist under tool_root_dir.
    Returns:
      query_json: {"api_list":[{"category_name", "tool_name", "api_name"}, ...]}
      mt_descriptions: [{"category","tool_name","api_name","api_description","score"}, ...]
    """
    retrieved_tools = retriever.retrieving(query, top_k=top_k)

    query_json = {"api_list": []}
    mt_descriptions = []

    for tool in retrieved_tools:
        category = tool["category"]
        tool_name = tool["tool_name"]
        api_name  = tool["api_name"]
        corpus_id = tool["corpus_id"]
        score     = tool["score"]

        api_description = retriever.corpus[corpus_id]

        # Only include entries that resolve to a real JSON file on disk
        if len(query_json["api_list"]) < top_k:
            cat_dir = os.path.join(tool_root_dir, category)
            if os.path.isdir(cat_dir):
                tool_json_path = os.path.join(cat_dir, tool_name + ".json")
                if os.path.exists(tool_json_path):
                    query_json["api_list"].append({
                        "category_name": category,
                        "tool_name": tool_name,
                        "api_name": api_name,
                    })

        # Always record the full retrieval info for sanity/debug
        mt_descriptions.append({
            "category": category,
            "tool_name": tool_name,
            "api_name": api_name,
            "api_description": api_description,
            "score": score,
        })

    return query_json, mt_descriptions

def normalize_api_descs(items: List[Any]) -> List[str]:
    """Mixed-shape tolerant; safe even if dict-only."""
    out: List[str] = []
    for it in items or []:
        if isinstance(it, dict):
            s = (it.get("api_description") or "").strip()
        elif isinstance(it, (tuple, list)) and len(it) >= 3:
            s = (it[2] or "").strip()
        else:
            s = ""
        if s:
            out.append(s)
    return out

# format a numbered block as exemplars during refinement/regeneration; skip empties; prettifies if list input
def _fmt_block(items):
    return "\n".join(
        f"{i}. {s.strip()}"
        for i, s in enumerate(items, 1)
        if s and str(s).strip()
    )
def build_refine_or_regen_messages(
    *,
    mode: Mode,                      # "dbd"
    query: str,
    refinement: bool,                # True=refine, False=regen
    # DBD fields:
    ancestor_desc = None,
    current_desc = None,
    lineage_examples = None,
    turn = None,
    # DBD refine-prompt style switch (root-node ablation):
    refine_style: str = 'fittext',
):
    """
    Prompt construction for DBD refinement/regeneration turns.
    - DBD uses (ancestor_desc, current_desc, lineage_examples, turn).
    - refine_style='fittext' (default) refines a pseudo-tool description (FitText axis-a).
    - refine_style='xu'      refines the USER INSTRUCTION (Xu et al. 2024 axis-a). Only
      applies when refinement=True. Mechanical interface (BEGIN/END
      tokens, downstream parser) is preserved either way.
    """
    lineage_block = _fmt_block(lineage_examples or [])
    if refinement:
        if refine_style == 'xu':
            return make_dbd_refine_messages_xu(
                query=query,
                ancestor_desc=ancestor_desc or "",
                current_desc=current_desc or "",
                lineage_examples_block=lineage_block,
                turn=int(turn or 0),
            )
        return make_dbd_refine_messages(
            query=query,
            ancestor_desc=ancestor_desc or "",
            current_desc=current_desc or "",
            lineage_examples_block=lineage_block,
            turn=int(turn or 0),
        )
    # regen
    return make_dbd_regen_messages(
        query=query,
        ancestor_desc=ancestor_desc or "",
        lineage_examples_block=lineage_block,
    )