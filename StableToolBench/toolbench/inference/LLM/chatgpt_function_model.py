"""
Back-compat shim: ChatGPTFunction + chat_completion_request.

All routing logic has moved to ``clients/factory.py``.
This module preserves the exact callable surface that DFSDT and other callers
expect, but delegates to ``make_client`` internally.

Callers that currently do:
    response = chat_completion_request(key, base_url, messages, tools=tools, model=model)
    message  = response["choices"][0]["message"]
    tokens   = response["usage"]["total_tokens"]
continue to work unchanged.
"""

from __future__ import annotations

import asyncio
import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from termcolor import colored

from .clients import make_client, NormalizedResponse
from toolbench.observability.pins import load_pins as _load_pins

# ---------------------------------------------------------------------------
# Default model — sourced from configs/model_pins.yaml (agents.gpt_4_1_mini).
# Never change this constant directly; edit the pin file instead.
# ---------------------------------------------------------------------------
_PINS = _load_pins(Path(__file__).parent.parent.parent.parent.parent)
_DEFAULT_AGENT_MODEL: str = _PINS["agents"]["gpt_4_1_mini"]


# ---------------------------------------------------------------------------
# Module-level function (back-compat)
# ---------------------------------------------------------------------------

def chat_completion_request(
    key: str,
    base_url: Optional[str],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice=None,
    key_pos=None,
    model: str = _DEFAULT_AGENT_MODEL,
    stop=None,
    process_id: int = 0,
    **args: Any,
) -> Dict[str, Any]:
    """
    Thin shim around ``make_client().chat_completion_sync()``.

    Returns an OpenAI-shaped dict so that existing callers need no changes::

        response["choices"][0]["message"]  # the assistant message
        response["usage"]["total_tokens"]  # total token count

    Parameters
    ----------
    key : str
        API key; passed to the client constructor as ``api_key``.
    base_url : str | None
        For vLLM models, the OpenAI-compat base URL.
    messages : list[dict]
        Conversation history (may contain ``valid: False`` entries which are
        filtered internally by the client).
    tools : list[dict] | None
        OpenAI-format tool definitions.
    tool_choice : any
        Forwarded to the underlying client (OpenAI/vLLM only).
    key_pos : any
        Legacy parameter; ignored.
    model : str
        Full model identifier.
    stop : str | list | None
        Stop sequences; forwarded verbatim.
    process_id : int
        For logging only.
    **args
        Extra kwargs forwarded to the client's ``chat_completion``.

    Returns
    -------
    dict
        OpenAI ChatCompletion-shaped dict, or ``{"error": ..., "total_tokens": 0}``
        on failure.
    """
    # Filter invalid messages (toolbench convention).
    use_messages = [
        m for m in messages
        if not (m.get("valid") is False)
    ]
    # Strip legacy function_call key that the API no longer accepts.
    for m in use_messages:
        m.pop("function_call", None)

    client_kwargs: Dict[str, Any] = {}
    if key:
        client_kwargs["api_key"] = key
    if base_url:
        client_kwargs["base_url"] = base_url

    extra: Dict[str, Any] = {}
    if stop is not None:
        extra["stop"] = stop
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice
    extra.update(args)

    try:
        client = make_client(model, **client_kwargs)
        response: NormalizedResponse = client.chat_completion_sync(
            messages=use_messages,
            tools=tools if tools else None,
            **extra,
        )
        return response.to_openai_dict()
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        traceback.print_exc()
        return {"error": str(e), "choices": [], "usage": {"total_tokens": 0}}


# ---------------------------------------------------------------------------
# Class (back-compat)
# ---------------------------------------------------------------------------

class ChatGPTFunction:
    """
    Stateful conversation wrapper delegating to the unified ModelClient.

    The public interface (``add_message``, ``change_messages``, ``parse``,
    ``parse_with_messages``) is unchanged; all provider routing is handled
    internally by ``make_client``.
    """

    def __init__(
        self,
        model: str = _DEFAULT_AGENT_MODEL,
        openai_key: str = "",
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.conversation_history: List[Dict[str, Any]] = []
        self.openai_key = openai_key
        self.base_url = base_url
        self.time = time.time()

    def add_message(self, message: Dict[str, Any]) -> None:
        """Append a message to the conversation history."""
        self.conversation_history.append(message)

    def change_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Replace the conversation history."""
        self.conversation_history = messages

    def display_conversation(self, detailed: bool = False) -> None:
        """Pretty-print the conversation history (debug helper)."""
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print" + "*" * 50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message.get('content', '')} "
            if "function_call" in message:
                print_obj += f"function_call: {message['function_call']}"
            if "tool_calls" in message:
                print_obj += f"tool_calls: {message['tool_calls']}"
                print_obj += f"number of tool calls: {len(message['tool_calls'])}"
            print(
                colored(
                    print_obj,
                    role_to_color.get(message["role"], "white"),
                )
            )
        print("end_print" + "*" * 50)

    def parse(
        self,
        tools: List[Dict[str, Any]],
        process_id: int,
        key_pos=None,
        **args: Any,
    ):
        """
        Run the LLM on the current conversation history.

        Returns
        -------
        (message_dict, error_code, total_tokens)
            ``message_dict`` is an OpenAI-shaped message dict.
            ``error_code`` is 0 on success, -1 on total failure.
            ``total_tokens`` is the sum of prompt + completion tokens.
        """
        self.time = time.time()
        conversation_history = self.conversation_history

        response = chat_completion_request(
            self.openai_key,
            self.base_url,
            conversation_history,
            tools=tools if tools else None,
            process_id=process_id,
            key_pos=key_pos,
            model=self.model,
            **args,
        )
        try:
            usage = response.get("usage") or {}
            total_tokens = usage.get("total_tokens", 0)
            message = response["choices"][0]["message"]

            if process_id == 0:
                print(f"[process({process_id})]total tokens: {total_tokens}")

            return message, 0, total_tokens
        except (KeyError, IndexError, TypeError) as e:
            # Tenacity inside chat_completion_request already retried 4× with
            # exponential backoff. A malformed response here means we're out of
            # retries — surface the error.
            print(f"[process({process_id})]Parsing Exception: {repr(e)}")
            traceback.print_exc()
            if response is not None:
                print(f"[process({process_id})]Return: {response}")
            return {"role": "assistant", "content": str(response)}, -1, 0

    def parse_with_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        process_id: int = 0,
        key_pos=None,
        **args: Any,
    ):
        """
        One-off LLM call with a custom conversation history.

        Does not mutate ``self.conversation_history``.

        Returns
        -------
        (message_dict, error_code, total_tokens)
        """
        from copy import deepcopy
        backup = deepcopy(self.conversation_history)
        self.conversation_history = messages
        try:
            return self.parse(
                tools=tools or [],
                process_id=process_id,
                key_pos=key_pos,
                **args,
            )
        finally:
            self.conversation_history = backup
