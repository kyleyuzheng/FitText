"""
Schema conversion utilities between OpenAI and Anthropic tool/message formats.

Key conversions
---------------
- ``openai_tools_to_anthropic``  : OpenAI tool defs → Anthropic tool defs
- ``anthropic_response_to_normalized`` : Anthropic Message → NormalizedResponse
- ``messages_openai_to_anthropic``     : OpenAI conversation history →
                                         Anthropic messages + system string
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .base import NormalizedResponse


def openai_tools_to_anthropic(
    openai_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert OpenAI function-tool definitions to Anthropic tool definitions.

    OpenAI format::

        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "...",
                "parameters": {...json-schema...}
            }
        }

    Anthropic format::

        {
            "name": "get_weather",
            "description": "...",
            "input_schema": {...json-schema...}
        }

    Parameters
    ----------
    openai_tools : list[dict]
        Tool definitions in OpenAI format.

    Returns
    -------
    list[dict]
        Tool definitions in Anthropic format.
    """
    anthropic_tools: List[Dict[str, Any]] = []
    for tool in openai_tools:
        if tool.get("type") != "function":
            # Skip non-function tools (unsupported on Anthropic side).
            continue
        fn = tool["function"]
        anthropic_tool: Dict[str, Any] = {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        }
        anthropic_tools.append(anthropic_tool)
    return anthropic_tools


def anthropic_response_to_normalized(
    response: Any,  # anthropic.types.Message
    latency_ms: float,
    model: str,
) -> NormalizedResponse:
    """
    Convert an Anthropic SDK ``Message`` object to a ``NormalizedResponse``.

    Anthropic returns content as a list of typed blocks:
    - ``TextBlock``    → becomes ``content``
    - ``ToolUseBlock`` → becomes an element in ``tool_calls`` (OpenAI shape)

    Parameters
    ----------
    response : anthropic.types.Message
        Native Anthropic response object.
    latency_ms : float
        Wall-clock time for the API call.
    model : str
        Model identifier as submitted (may differ from echoed revision).

    Returns
    -------
    NormalizedResponse
    """
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            # Convert to OpenAI tool_calls shape.
            arguments = block.input
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": arguments,
                    },
                }
            )

    content: Optional[str] = "\n".join(text_parts) if text_parts else None

    # Map Anthropic stop reasons to OpenAI equivalents.
    stop_reason_map = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    finish_reason = stop_reason_map.get(response.stop_reason or "", response.stop_reason)

    usage = response.usage
    input_tokens: int = getattr(usage, "input_tokens", 0) or 0
    # cache_read_input_tokens: tokens served from the prompt cache (billed ~10%).
    cached_tokens: int = getattr(usage, "cache_read_input_tokens", 0) or 0
    output_tokens: int = getattr(usage, "output_tokens", 0) or 0

    # Serialise raw response for provenance hashing (use model_dump if available).
    try:
        raw = response.model_dump()
    except AttributeError:
        raw = {}

    return NormalizedResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        model_revision=response.model,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        provider="anthropic",
        raw_response=raw,
    )


def messages_openai_to_anthropic(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Convert an OpenAI-format conversation history to Anthropic format.

    Anthropic differences from OpenAI:
    - No ``"system"`` role in messages; system prompt is a top-level param.
    - Tool results use role ``"user"`` with content blocks of type
      ``"tool_result"``, not role ``"tool"`` / ``"function"``.
    - ``tool_call_id`` maps to ``tool_use_id`` in ``tool_result`` blocks.
    - Assistant messages with tool calls use content blocks of type
      ``"tool_use"``, not the ``"tool_calls"`` key.

    Parameters
    ----------
    messages : list[dict]
        OpenAI-format conversation history.

    Returns
    -------
    (anthropic_messages, system_prompt)
        ``anthropic_messages`` is the converted list (system messages removed).
        ``system_prompt`` is the concatenated system prompt string, or None.
    """
    anthropic_messages: List[Dict[str, Any]] = []
    system_parts: List[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        # Skip invalid / flagged messages (toolbench convention).
        if msg.get("valid") is False:
            continue

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        if role in ("function", "tool"):
            # OpenAI tool result message → Anthropic tool_result content block.
            tool_use_id = msg.get("tool_call_id") or msg.get("name", "")
            tool_result_block: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content or "",
            }
            # Merge consecutive tool results into the previous user message
            # or start a new one.
            if (
                anthropic_messages
                and anthropic_messages[-1]["role"] == "user"
                and isinstance(anthropic_messages[-1]["content"], list)
            ):
                anthropic_messages[-1]["content"].append(tool_result_block)
            else:
                anthropic_messages.append(
                    {"role": "user", "content": [tool_result_block]}
                )
            continue

        if role == "assistant":
            # Check for tool_calls (OpenAI style) and convert to tool_use blocks.
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                content_blocks: List[Dict[str, Any]] = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    try:
                        parsed_input = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed_input = {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": parsed_input,
                        }
                    )
                anthropic_messages.append(
                    {"role": "assistant", "content": content_blocks}
                )
                continue

            # Plain assistant message.
            anthropic_messages.append(
                {"role": "assistant", "content": content or ""}
            )
            continue

        if role == "user":
            anthropic_messages.append({"role": "user", "content": content or ""})
            continue

        # Unknown roles: skip silently (toolbench adds non-standard roles).

    system_prompt: Optional[str] = "\n".join(system_parts) if system_parts else None
    return anthropic_messages, system_prompt
