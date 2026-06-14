"""
Smoke tests for the unified ModelClient abstraction.

Tests
-----
1. OpenAI client: gpt-4.1-mini-2025-04-14 — live 1-token request, NormalizedResponse fields.
2. Anthropic client: claude-sonnet-4-6 — live request with tool, adapter output OpenAI-shaped.
3. vLLM client: constructed only (live call skipped if VLLM_BASE_URL not set).
4. chat_completion_request shim: offline schema-conversion path returns old shape.
5. Adapter unit tests: openai_tools_to_anthropic, messages_openai_to_anthropic — no API needed.

Run with:
    pytest tests/smoke_modelclient.py -v

Skips cleanly when API keys are absent.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Make the StableToolBench package importable from the repo root.
# ---------------------------------------------------------------------------
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_tests_dir, ".."))
_stabletoolbench = os.path.join(_repo_root, "StableToolBench")
if _stabletoolbench not in sys.path:
    sys.path.insert(0, _stabletoolbench)

from toolbench.inference.LLM.clients import NormalizedResponse, make_client
from toolbench.inference.LLM.clients.adapters import (
    messages_openai_to_anthropic,
    openai_tools_to_anthropic,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi"},
]


# ---------------------------------------------------------------------------
# Test 1 — OpenAI live call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
async def test_openai_client_live():
    """
    Build an OpenAI client for gpt-4.1-mini-2025-04-14, make a minimal request,
    assert NormalizedResponse fields are populated.
    """
    client = make_client("gpt-4.1-mini-2025-04-14")
    resp = await client.chat_completion(
        messages=[{"role": "user", "content": "Reply with exactly the word: OK"}],
        max_tokens=5,
    )

    assert isinstance(resp, NormalizedResponse), "Expected NormalizedResponse"
    assert resp.provider == "openai"
    assert isinstance(resp.model_revision, str) and resp.model_revision
    assert resp.output_tokens > 0, "output_tokens should be positive"
    assert resp.input_tokens > 0, "input_tokens should be positive"
    assert isinstance(resp.latency_ms, float) and resp.latency_ms > 0
    assert resp.content is not None, "content should not be None for a text response"

    # Check to_openai_dict() round-trip.
    d = resp.to_openai_dict()
    assert d["choices"][0]["message"]["role"] == "assistant"
    assert d["usage"]["total_tokens"] == resp.input_tokens + resp.output_tokens


# ---------------------------------------------------------------------------
# Test 2 — Anthropic live call with tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
async def test_anthropic_client_live_with_tool():
    """
    Build an Anthropic client for claude-sonnet-4-6, make a request with a tool,
    assert tool_calls adapter output has OpenAI shape.
    """
    client = make_client("claude-sonnet-4-6")
    resp = await client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ],
        tools=SAMPLE_OPENAI_TOOLS,
        max_tokens=256,
    )

    assert isinstance(resp, NormalizedResponse), "Expected NormalizedResponse"
    assert resp.provider == "anthropic"
    assert isinstance(resp.model_revision, str) and resp.model_revision
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert isinstance(resp.latency_ms, float) and resp.latency_ms > 0

    # When the model calls a tool, tool_calls should be populated.
    if resp.tool_calls:
        tc = resp.tool_calls[0]
        assert tc["type"] == "function", "tool_calls[0].type must be 'function'"
        assert "name" in tc["function"]
        assert "arguments" in tc["function"]
        # arguments must be valid JSON string.
        parsed = json.loads(tc["function"]["arguments"])
        assert isinstance(parsed, dict)

    # OpenAI-shaped dict must be accessible.
    d = resp.to_openai_dict()
    assert "choices" in d and len(d["choices"]) == 1
    msg = d["choices"][0]["message"]
    assert msg["role"] == "assistant"


# ---------------------------------------------------------------------------
# Test 3 — vLLM client construction (live call skipped)
# ---------------------------------------------------------------------------

def test_vllm_client_construction():
    """
    Build a vLLM client for Qwen — no API call made.
    Skips live call when VLLM_BASE_URL is absent.
    """
    client = make_client("Qwen/Qwen3-30B-A3B")
    from toolbench.inference.LLM.clients.vllm_client import VLLMClient
    assert isinstance(client, VLLMClient)
    assert client.model == "Qwen/Qwen3-30B-A3B"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("VLLM_BASE_URL"),
    reason="VLLM_BASE_URL not set — skipping live vLLM call",
)
async def test_vllm_client_live():
    """Live vLLM smoke: one-token completion."""
    client = make_client("Qwen/Qwen3-30B-A3B")
    resp = await client.chat_completion(
        messages=[{"role": "user", "content": "Say: OK"}],
        max_tokens=5,
    )
    assert isinstance(resp, NormalizedResponse)
    assert resp.provider == "vllm"
    assert resp.output_tokens > 0


# ---------------------------------------------------------------------------
# Test 4 — chat_completion_request shim (offline)
# ---------------------------------------------------------------------------

def test_chat_completion_request_shim_offline(monkeypatch):
    """
    Verify that chat_completion_request still returns the old OpenAI dict shape,
    with the new ModelClient under the hood (mocked).
    """
    from toolbench.inference.LLM.chatgpt_function_model import chat_completion_request

    # Patch make_client to avoid a real API call.
    fake_resp = NormalizedResponse(
        content="Hello",
        tool_calls=[],
        finish_reason="stop",
        model_revision="gpt-4.1-mini-2025-04-14",
        input_tokens=10,
        cached_input_tokens=0,
        output_tokens=3,
        latency_ms=42.0,
        provider="openai",
    )

    class FakeClient:
        def chat_completion_sync(self, **kwargs):
            return fake_resp

    monkeypatch.setattr(
        "toolbench.inference.LLM.chatgpt_function_model.make_client",
        lambda *a, **kw: FakeClient(),
    )

    result = chat_completion_request(
        key="sk-test",
        base_url=None,
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-4.1-mini-2025-04-14",
    )

    assert "choices" in result
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["choices"][0]["message"]["content"] == "Hello"
    assert result["usage"]["total_tokens"] == 13


# ---------------------------------------------------------------------------
# Test 5 — Adapter unit tests (no API needed)
# ---------------------------------------------------------------------------

def test_openai_tools_to_anthropic_schema():
    """Convert OpenAI tool defs → Anthropic; verify field mapping."""
    anthropic_tools = openai_tools_to_anthropic(SAMPLE_OPENAI_TOOLS)

    assert len(anthropic_tools) == 1
    t = anthropic_tools[0]
    assert t["name"] == "get_weather"
    assert t["description"] == "Get the current weather for a location."
    assert "input_schema" in t
    assert t["input_schema"]["type"] == "object"
    assert "location" in t["input_schema"]["properties"]


def test_messages_openai_to_anthropic_system_extraction():
    """System message is extracted; remaining messages converted correctly."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    anthropic_msgs, system = messages_openai_to_anthropic(msgs)

    assert system == "You are helpful."
    assert len(anthropic_msgs) == 2
    assert anthropic_msgs[0]["role"] == "user"
    assert anthropic_msgs[1]["role"] == "assistant"


def test_messages_openai_to_anthropic_tool_result():
    """tool role messages → Anthropic tool_result content blocks."""
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_abc", "content": '{"temperature": "10°C"}'},
    ]
    anthropic_msgs, system = messages_openai_to_anthropic(msgs)

    assert system is None
    # First: assistant with tool_use block
    assert anthropic_msgs[0]["role"] == "assistant"
    content = anthropic_msgs[0]["content"]
    assert isinstance(content, list)
    tool_use = content[0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"] == "call_abc"
    assert tool_use["name"] == "get_weather"
    assert tool_use["input"] == {"location": "Tokyo"}

    # Second: user with tool_result block
    assert anthropic_msgs[1]["role"] == "user"
    tr = anthropic_msgs[1]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "call_abc"


def test_messages_openai_to_anthropic_filters_invalid():
    """Messages with valid=False are silently dropped."""
    msgs = [
        {"role": "user", "content": "keep me"},
        {"role": "user", "content": "drop me", "valid": False},
    ]
    anthropic_msgs, _ = messages_openai_to_anthropic(msgs)
    assert len(anthropic_msgs) == 1
    assert anthropic_msgs[0]["content"] == "keep me"


def test_factory_routing():
    """make_client routes to correct client class without calling API."""
    from toolbench.inference.LLM.clients.openai_client import OpenAIClient
    from toolbench.inference.LLM.clients.anthropic_client import AnthropicClient
    from toolbench.inference.LLM.clients.vllm_client import VLLMClient

    assert isinstance(make_client("gpt-4.1-mini-2025-04-14"), OpenAIClient)
    assert isinstance(make_client("claude-sonnet-4-6"), AnthropicClient)
    assert isinstance(make_client("Qwen/Qwen3-30B"), VLLMClient)
    assert isinstance(make_client("deepseek-v3"), VLLMClient)

    with pytest.raises(NotImplementedError):
        make_client("llama-3-8b-unknown")
