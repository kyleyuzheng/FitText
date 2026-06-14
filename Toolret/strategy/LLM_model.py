"""
Back-compat shim: ChatGPTFunction + chat_completion_request (Toolret copy).

All routing logic has moved to
``StableToolBench/toolbench/inference/LLM/clients/factory.py``.

This module re-exports the same interface as
``toolbench.inference.LLM.chatgpt_function_model`` so that Toolret callers
need no changes.  Both files now share one code path.
"""

from __future__ import annotations

import json
import sys
import os
import time
import traceback
from typing import Any, Dict, List, Optional

from termcolor import colored

# ---------------------------------------------------------------------------
# Resolve the shared clients package regardless of working directory.
# Adds StableToolBench to sys.path when not already importable.
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_this_dir, "..", ".."))
_stabletoolbench = os.path.join(_repo_root, "StableToolBench")
if _stabletoolbench not in sys.path:
    sys.path.insert(0, _stabletoolbench)

from toolbench.inference.LLM.clients import make_client, NormalizedResponse  # noqa: E402
from toolbench.observability.pins import load_pins as _load_pins  # noqa: E402

# ---------------------------------------------------------------------------
# Default model — sourced from configs/model_pins.yaml (agents.gpt_4_1_mini).
# Never change this constant directly; edit the pin file instead.
# ---------------------------------------------------------------------------
_PINS = _load_pins(os.path.normpath(os.path.join(_this_dir, "..", "..")))
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

    Returns an OpenAI-shaped dict compatible with:
    - ``response["choices"][0]["message"]``
    - ``response["usage"]["total_tokens"]``

    Parameters are identical to the original ``chat_completion_request``
    signature so existing call sites require no changes.
    """
    use_messages = [
        m for m in messages
        if not (m.get("valid") is False)
    ]
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

    Identical interface to ``toolbench.inference.LLM.chatgpt_function_model.ChatGPTFunction``.
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
            # exponential backoff. A malformed response here means we're out
            # of retries — surface the error.
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


# ---------------------------------------------------------------------------
# Keep the test/demo that was in the original file.
# ---------------------------------------------------------------------------

def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Stub tool for local testing."""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    return json.dumps({"location": location, "temperature": "unknown"})


if __name__ == "__main__":
    llm = ChatGPTFunction(openai_key="", model=_DEFAULT_AGENT_MODEL)
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    llm.change_messages(messages)
    output, error_code, token_usage = llm.parse(tools=tools, process_id=0)
    print(output)
