"""
Helper functions for various strategies.py used in dynamic tool retrieval.
"""
from typing import Literal
from .prompts import *

Mode = Literal["dbd"]

# format a numbered block as exemplars during refinement/regeneration; skip empties; prettifies if list input
def _fmt_block(items):
    return "\n".join(
        (f"{i}. {s['tool_des'].strip()}" if i == 1 
         else f"        {i}. {s['tool_des'].strip()}")
        for i, s in enumerate(items, 1)
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
):
    """
    Prompt construction for DBD refinement/regeneration turns.
    DBD uses (ancestor_desc, current_desc, lineage_examples, turn).
    """
    lineage_block = _fmt_block(lineage_examples or [])
    if refinement:
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