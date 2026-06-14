"""
vLLM client via OpenAI-compatible endpoint.

Handles open-weight models (Qwen-*, DeepSeek-*, etc.) served by a local vLLM
instance that exposes the OpenAI chat completions API.

Configuration via environment variables
----------------------------------------
VLLM_BASE_URL       Base URL of the vLLM server  (default: http://localhost:8000/v1)
VLLM_API_KEY        API key sent to vLLM          (default: EMPTY)
VLLM_CLIENT_DEBUG   Set to 1 to emit per-request debug lines on stderr
                    (request shaping, response shape, prose-call recovery).

No hardcoded ports or paths — all from env.
"""

from __future__ import annotations

import os
import re
import ast
import json
import sys
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import ModelClient, NormalizedResponse


def _debug_log(line: str) -> None:
    """Print a debug line to stderr iff VLLM_CLIENT_DEBUG=1.

    Gates the per-request instrumentation (request shaping, raw response
    shape, prose tool-call recovery) so production runs stay quiet.
    """
    if os.environ.get("VLLM_CLIENT_DEBUG") == "1":
        print(line, file=sys.stderr, flush=True)

# FIX 1 (2026-05-29): Qwen3-30B frequently emits tool calls as ReAct-style PROSE
# in the content channel ("Call: name(args)", "Function call: name()", "Action:
# name(...)") instead of structured tool_calls. The vLLM qwen3_coder parser then
# returns no tool_calls and DFSDT prunes the turn as a salvage give_up — the
# dominant (~56%) cause of Qwen3-30B dynamic-strategy give-ups in root-cause
# analysis. Recover the FIRST prose call into a synthetic structured
# tool_call so DFSDT executes it for real (discarding any hallucinated "Call
# result:" the model wrote after it).
# Match the marker anywhere (Qwen writes "Thought: … Call: fn(args)" inline, not
# only at line start). The ":NAME(" shape is the strong disambiguator, so prose
# like "Call result: foo" (no NAME immediately after the colon) won't match, and
# \b avoids matching inside words like "Recall". Longer alternatives first.
_TEXTCALL_RE = re.compile(
    r'\b(?:Function\s+call|Tool\s+call|Call|Action)\s*:\s*'
    r'([A-Za-z_]\w*)\s*\((.*?)\)',
    re.IGNORECASE | re.DOTALL,
)


# FIX 1b recognized tool-call WRAPPERS. Only JSON the model EXPLICITLY marked as a
# call is recovered: the STB-native delimiter
# <|begin_func_call|>{json}<|end_func_call|>, the Qwen <|tool_call>/<tool_call>
# delimiter, and fenced ```json/```tool blocks. Bare unwrapped JSON is intentionally
# NOT recovered — that prevented executing an echoed tool SCHEMA or example JSON in
# a Thought/<|begin_func_description|> block as if it were a real call.
_CALL_OPEN_RE = re.compile(r'<\|?(?:begin_)?(?:tool|func)_call\|?>', re.IGNORECASE)
# Newline after the fence info-string is OPTIONAL so single-line ```json {...}```
# is also caught.
_FENCE_OPEN_RE = re.compile(r'```[ \t]*(?:json|tool|tool_call|python)?[ \t]*\r?\n?', re.IGNORECASE)
_MAX_SCAN = 20000  # cap balanced-bracket scan; one assistant turn is small


def _balanced_bracket(s: str, start: int) -> Optional[str]:
    """Return the string-aware balanced {...}/[...] substring beginning at s[start].

    Uses a stack of expected closers (so ``{`` and ``[`` cannot cross-close) and
    respects JSON string literals (braces inside quotes don't affect depth). Scans
    at most ``_MAX_SCAN`` chars. Returns None on a malformed/unbalanced fragment.
    """
    pair = {"{": "}", "[": "]"}
    if start >= len(s) or s[start] not in pair:
        return None
    stack: List[str] = []
    in_str = False
    esc = False
    end = min(len(s), start + _MAX_SCAN)
    i = start
    while i < end:
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(pair[ch])
            elif ch in "}]":
                if not stack or stack[-1] != ch:
                    return None  # mismatched bracket type
                stack.pop()
                if not stack:
                    return s[start:i + 1]
        i += 1
    return None


def _normalize_args(args: Any) -> Optional[str]:
    """Return a valid JSON-OBJECT string for synthetic tool_call.arguments, else None.

    DFSDT downstream does ``json.loads`` on this string, so it must be a JSON
    object. A string is itself json-parsed (handles double-encoded args) and must
    yield a dict; non-dict (list/scalar/unparseable) is rejected. JSON-schema-shaped
    args (a dict with both ``type`` and ``properties``) are rejected — that is a
    tool SCHEMA echoed by the model, not a call's arguments.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return None
    if not isinstance(args, dict):
        return None
    if "type" in args and "properties" in args:
        return None
    try:
        return json.dumps(args)
    except Exception:
        return None


def _json_call_after(content: str, pos: int):
    """(name, args_str) for the first balanced JSON tool-call at/after pos, else None."""
    m = re.search(r"[{\[]", content[pos:pos + _MAX_SCAN])
    if not m:
        return None
    start = pos + m.start()
    frag = _balanced_bracket(content, start)
    if not frag:
        return None
    try:
        obj = json.loads(frag)
    except Exception:
        try:
            obj = ast.literal_eval(frag)  # tolerate single-quoted dicts
        except Exception:
            return None
    call = obj[0] if isinstance(obj, list) and obj else obj
    if not isinstance(call, dict):
        return None
    name = call.get("name")
    if not isinstance(name, str) or not name:
        return None
    if "arguments" not in call and "parameters" not in call:
        return None
    args_str = _normalize_args(call.get("arguments", call.get("parameters")))
    if args_str is None:
        return None
    return name, args_str


def _find_first_json_toolcall(content: str):
    """Earliest tool-call inside a recognized WRAPPER (delimiter or fenced block).

    FIX 1b (2026-05-30): Qwen3-30B emits STRUCTURED tool-calls the vLLM qwen3_coder
    parser misses — the STB-native ``<|begin_func_call|>{json}<|end_func_call|>``,
    the ``<|tool_call>{json}</tool_call>`` delimiter, and fenced ```json blocks.
    These were the dominant remaining parser-miss give-ups (~21% of the
    residual). A tool-call is a JSON object (or first
    element of a JSON array) with a string ``name`` and an ``arguments``/``parameters``
    mapping. Returns (wrapper_start, name, args_str) for the earliest match, else
    None — wrapper_start lets the caller drop the whole delimiter from content.
    """
    best = None
    for rgx in (_CALL_OPEN_RE, _FENCE_OPEN_RE):
        for m in rgx.finditer(content):
            hit = _json_call_after(content, m.end())
            if hit and (best is None or m.start() < best[0]):
                best = (m.start(), hit[0], hit[1])
    return best


def _reparse_text_tool_call(content: Optional[str]):
    """Reparse an unparsed tool-call in `content` into (tool_calls, clean_content).

    Two recovery paths, picking whichever appears EARLIEST in the content:
      (A) wrapped JSON tool-call — ``<|begin_func_call|>``/``<|tool_call>`` delimiters
          or fenced ```json blocks (the dominant Qwen3-30B parser-miss formats);
      (B) ReAct prose — ``Call/Action/Function call: name(args)``.
    Returns ([], content) if neither matches. On success returns ONE synthetic
    tool_call and the content truncated to the reasoning that preceded the call
    (drops any fabricated call/result continuation). Only ever invoked when the
    server returned NO structured tool_calls, so it cannot clobber a parsed turn.
    """
    if not content:
        return [], content

    candidates: List[tuple] = []  # (start, name, args_str)

    # (A) Wrapped structured JSON tool-call.
    jhit = _find_first_json_toolcall(content)
    if jhit:
        candidates.append(jhit)

    # (B) ReAct-style prose call: Call/Action/Function call: name(args).
    m = _TEXTCALL_RE.search(content)
    if m:
        name = m.group(1)
        argstr = (m.group(2) or "").strip()
        args_dict: Dict[str, Any] = {}
        if argstr:
            try:
                call_node = ast.parse(f"_f({argstr})", mode="eval").body
                for kw in getattr(call_node, "keywords", []):
                    if kw.arg is None:
                        continue
                    try:
                        args_dict[kw.arg] = ast.literal_eval(kw.value)
                    except Exception:
                        args_dict[kw.arg] = None
            except Exception:
                # Lenient k='v' / k="v" / k=number fallback for non-literal args.
                for k, v in re.findall(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,)]+)", argstr):
                    args_dict[k] = v.strip().strip("'\"")
        candidates.append((m.start(), name, json.dumps(args_dict)))

    if not candidates:
        return [], content

    # Earliest call in document order wins (don't let a later JSON beat an earlier
    # prose call, and vice versa). clean = reasoning before the chosen call.
    start, name, args_str = min(candidates, key=lambda c: c[0])
    clean = content[:start].rstrip() or content[:start]
    return [{
        "id": "call_reparse_0",
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }], clean

# Module-level client; created lazily so tests that skip vLLM don't fail on
# import.  A new instance is created if VLLM_BASE_URL changes at runtime.
_vllm_client: Optional[AsyncOpenAI] = None
_vllm_base_url_at_init: Optional[str] = None


def _get_vllm_client() -> AsyncOpenAI:
    """
    Return the module-level ``AsyncOpenAI`` vLLM client.

    Reads ``VLLM_BASE_URL`` and ``VLLM_API_KEY`` at call time so that env
    variables set after import are respected.  Rebuilds the client if the URL
    has changed (e.g. tests pointing at a different server).

    Returns
    -------
    AsyncOpenAI
    """
    global _vllm_client, _vllm_base_url_at_init
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    if _vllm_client is None or _vllm_base_url_at_init != base_url:
        _vllm_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        _vllm_base_url_at_init = base_url

    return _vllm_client


class VLLMClient(ModelClient):
    """
    ModelClient implementation for vLLM-served open-weight models.

    Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint exposed by
    vLLM.  All serialisation is identical to ``OpenAIClient`` — only the
    ``base_url`` differs.

    Parameters
    ----------
    model : str
        Model identifier as registered with the vLLM server
        (e.g. ``"Qwen/Qwen3-30B-A3B"``).
    base_url : str | None
        Override ``VLLM_BASE_URL`` env var for this instance.
    api_key : str | None
        Override ``VLLM_API_KEY`` env var for this instance.
    **kwargs
        Ignored (forward-compat for factory).
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, cache=cache)
        # Per-instance client if overrides are provided; otherwise use module-level.
        if base_url is not None or api_key is not None:
            self._client: Optional[AsyncOpenAI] = AsyncOpenAI(
                base_url=base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
            )
        else:
            self._client = None  # resolved lazily via _get_vllm_client()

    def _resolve_client(self) -> AsyncOpenAI:
        """Return the per-instance or module-level client."""
        return self._client if self._client is not None else _get_vllm_client()

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """
        Call the vLLM OpenAI-compat chat completions endpoint.

        Estimates a safe ``max_completion_tokens`` budget from the input length
        to avoid context overflow on smaller vLLM context windows.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format conversation history.
        tools : list[dict] | None
            OpenAI-format tool definitions.
        temperature : float
            Sampling temperature.
        seed : int
            Reproducibility seed (forwarded to vLLM; server may ignore it).
        max_tokens : int | None
            Max completion tokens; auto-calculated if not provided.
        **kwargs
            Extra parameters forwarded to ``create()``.

        Returns
        -------
        NormalizedResponse
        """
        client = self._resolve_client()
        t0 = self._now_ms()

        # vLLM context budget estimation: ~4 chars per token.
        # FIX 2 (2026-05-29): the hardcoded model_ctx=8192 + min(1024,…) output
        # cap was the secondary give-up cause — the dynamic retrieval preamble
        # blew past 8192 (→ HTTP 400, swallowed → salvage give_up) and the output
        # ceiling collapsed toward 256, truncating tool-call emission. The served
        # window is now 32768 (serve_vllm*.sh --max-model-len). Read it from env
        # (VLLM_MODEL_CTX) and raise the per-turn output ceiling so Qwen3-30B has
        # room to emit a full structured tool_call.
        if max_tokens is None:
            est_input = sum(len(m.get("content", "") or "") for m in messages) // 4
            model_ctx = int(os.environ.get("VLLM_MODEL_CTX", "32768"))
            max_tokens = max(512, min(4096, model_ctx - est_input - 256))

        # DIVERSITY FIX (2026-05-29): do NOT pin a fixed seed for Qwen/vLLM.
        # vLLM honors `seed` STRICTLY, so the upstream default seed=42 made every
        # sample at temp>0 byte-identical — collapsing scattershot's `size`
        # children and memetic's population to ZERO diversity (gpt-4.1-mini over
        # OpenAI hid this because OpenAI's seed is best-effort). Omitting seed =>
        # vLLM draws a fresh random seed per request => real diversity at temp 1.5.
        # A caller can still force reproducibility by passing seed=<int> in kwargs.
        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = tools

        # Qwen3.x thinking-mode disable + multi-turn preservation.
        # Without this, the model consumes the entire response window with
        # <think> tokens and emits content but no tool_calls, which DFSDT
        # then salvages as give_up (0% pass rate). vLLM ≥0.21 honors the flag.
        if isinstance(self.model, str) and self.model.lower().startswith("qwen"):
            existing_eb = create_kwargs.get("extra_body") or {}
            ctk = dict(existing_eb.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            ctk.setdefault("preserve_thinking", True)
            existing_eb["chat_template_kwargs"] = ctk
            create_kwargs["extra_body"] = existing_eb

            # Qwen3.x chat template requires system messages first. DFSDT
            # interleaves system messages mid-stream (planner_for_dfsdt and
            # function-response system frames) which triggers a vLLM 400:
            # "System message must be at the beginning." Consolidate all
            # system content to the front so the rest of the conversation is
            # a valid user/assistant alternation for Qwen.
            msgs = create_kwargs.get("messages") or []
            sys_parts = [str((m.get("content") or "")) for m in msgs if m.get("role") == "system"]
            others = [m for m in msgs if m.get("role") != "system"]
            if sys_parts:
                merged_sys = "\n\n".join(p for p in sys_parts if p.strip())
                create_kwargs["messages"] = ([{"role": "system", "content": merged_sys}] if merged_sys else []) + others
            _bu = getattr(client, 'base_url', None)
            _debug_log(f"[vllm_client patch] base_url={_bu!r} model={self.model} extra_body={existing_eb} msg_count={len(messages)} max={create_kwargs.get('max_completion_tokens')} tools_count={len(create_kwargs.get('tools') or [])}")

        # Gemma 4 system-frame consolidation + tuned sampling.
        # The gemma4 chat template (chat_template.jinja:179/186) only renders a
        # `system` turn when it is messages[0]; DFSDT inserts system frames
        # mid-stream (planner + function-response frames), which the template
        # re-emits as stray mid-conversation <|turn>system blocks Gemma was not
        # trained on — degrading tool calling. Consolidate all system content to
        # the front (same transform as the Qwen branch above). Thinking is OFF by
        # default (enable_thinking defaults false, jinja:182/359), so no
        # enable_thinking kwarg is needed; the server's --reasoning-parser gemma4
        # strips the empty <|channel>thought block. Apply Gemma's documented
        # sampling (top_k=64, top_p=0.95 — generation_config.json) via extra_body,
        # using setdefault so an explicit caller override is never clobbered.
        elif isinstance(self.model, str) and "gemma" in self.model.lower():
            existing_eb = create_kwargs.get("extra_body") or {}
            existing_eb.setdefault("top_k", 64)
            create_kwargs["extra_body"] = existing_eb
            create_kwargs.setdefault("top_p", 0.95)
            msgs = create_kwargs.get("messages") or []
            sys_parts = [str((m.get("content") or "")) for m in msgs if m.get("role") == "system"]
            others = [m for m in msgs if m.get("role") != "system"]
            if sys_parts:
                merged_sys = "\n\n".join(p for p in sys_parts if p.strip())
                create_kwargs["messages"] = ([{"role": "system", "content": merged_sys}] if merged_sys else []) + others
            _bu = getattr(client, 'base_url', None)
            _debug_log(f"[vllm_client gemma patch] base_url={_bu!r} model={self.model} msg_count={len(create_kwargs.get('messages') or [])} max={create_kwargs.get('max_completion_tokens')} tools_count={len(create_kwargs.get('tools') or [])} top_k={existing_eb.get('top_k')} top_p={create_kwargs.get('top_p')}")

        response = await client.chat.completions.create(**create_kwargs)
        latency_ms = self._now_ms() - t0

        choice = response.choices[0]
        msg = choice.message

        if isinstance(self.model, str) and (self.model.lower().startswith("qwen") or "gemma" in self.model.lower()):
            _c = msg.content or ""
            _tc_count = len(msg.tool_calls or [])
            _rc = getattr(msg, "reasoning_content", None) or ""
            _debug_log(f"[vllm_client resp] content_chars={len(_c)} tc_count={_tc_count} rc_chars={len(_rc)} finish={choice.finish_reason} content_head={_c[:150]!r} rc_head={_rc[:150]!r}")

        tool_calls: List[Dict[str, Any]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        # FIX 1: recover prose tool-calls when the structured parser found none.
        final_content = msg.content
        if (not tool_calls) and isinstance(self.model, str) and self.model.lower().startswith("qwen"):
            _rtc, _clean = _reparse_text_tool_call(msg.content)
            if _rtc:
                tool_calls = _rtc
                final_content = _clean
                _debug_log(f"[vllm_client reparse] recovered prose tool_call -> {_rtc[0]['function']['name']}({_rtc[0]['function']['arguments'][:80]})")

        usage = response.usage
        input_tokens: int = usage.prompt_tokens if usage else 0
        output_tokens: int = usage.completion_tokens if usage else 0

        try:
            raw = response.model_dump()
        except AttributeError:
            raw = {}

        return NormalizedResponse(
            content=final_content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model_revision=response.model,
            input_tokens=input_tokens,
            cached_input_tokens=0,  # vLLM does not report caching telemetry
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            provider="vllm",
            raw_response=raw,
        )
