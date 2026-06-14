import re

BEGIN = "<|begin_func_description|>"
END = "<|end_func_description|>"

# Strict pattern — matches a properly closed BEGIN…END block. Used for
# pseudo-tool extraction (run_memetic_strategy, _compare_with_tool_memory).
FUNC_DESC_PATTERN = re.compile(rf'{re.escape(BEGIN)}(.*?){re.escape(END)}', re.DOTALL)

# Tolerant pattern — also matches BEGIN-without-END from tool-eager reasoning
# models. Captures everything from BEGIN to the next BEGIN, or to
# end-of-string, whichever comes first. Used ONLY for *dispatch detection*
# in DFS.py — extraction stays on FUNC_DESC_PATTERN so we don't leak
# trailing tool-call markup into the pseudo-tool text.
FUNC_DESC_PATTERN_TOLERANT = re.compile(
    rf'{re.escape(BEGIN)}(.*?)(?:{re.escape(END)}|(?={re.escape(BEGIN)})|$)',
    re.DOTALL,
)


def has_func_description(text: str) -> bool:
    """True iff ``text`` contains at least one BEGIN marker (closed or not).

    DFS.py uses this as the protocol-dispatch detector: if the agent
    *intended* to retrieve (it wrote BEGIN), we route through
    select_and_run_strategy regardless of whether the agent also
    erroneously emitted a tool call in the same turn.
    """
    return bool(text) and BEGIN in text


def extract_func_descriptions(text: str, *, tolerant: bool = False) -> list[str]:
    """Return the list of pseudo-tool description bodies.

    ``tolerant=True`` accepts unclosed BEGIN blocks (used at dispatch).
    ``tolerant=False`` (default) accepts only closed BEGIN…END blocks
    (used for memetic extraction where we need clean text).
    """
    if not text:
        return []
    pat = FUNC_DESC_PATTERN_TOLERANT if tolerant else FUNC_DESC_PATTERN
    return [b.strip() for b in pat.findall(text) if b and b.strip()]
