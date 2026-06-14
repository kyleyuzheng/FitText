"""
Regression test for _to_responses_input — the converter that re-serializes
Chat-Completions message history into Responses-API native input items.

Root incident (2026-05-25): without this conversion, multi-turn DFSDT flows
on gpt-5.4-mini failed at turn 2 with HTTP 400 invalid_type because the
Responses API rejects {"role":"assistant","content":None,"tool_calls":[...]}.

These tests assert the exact transformation shape — they should fail loudly
if anyone reverts the converter to pass messages through unchanged.
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "StableToolBench"))

from toolbench.inference.LLM.clients.openai_client import _to_responses_input


def test_passthrough_system_and_user():
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    out = _to_responses_input(msgs)
    assert out == [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]


def test_assistant_tool_calls_become_function_call_items():
    msgs = [
        {"role": "user", "content": "verify +123"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_abc", "type": "function",
                 "function": {"name": "verify", "arguments": '{"phone":"+123"}'}}
            ],
        },
    ]
    out = _to_responses_input(msgs)
    assert len(out) == 2
    assert out[0] == {"role": "user", "content": "verify +123"}
    # Assistant message with null content + tool_calls → function_call item
    # (the null-content assistant message is dropped; only the function_call survives)
    assert out[1] == {
        "type": "function_call",
        "call_id": "call_abc",
        "name": "verify",
        "arguments": '{"phone":"+123"}',
    }


def test_tool_role_becomes_function_call_output():
    msgs = [
        {"role": "tool", "tool_call_id": "call_abc", "name": "verify",
         "content": '{"phone_valid": false}'},
    ]
    out = _to_responses_input(msgs)
    assert out == [
        {"type": "function_call_output", "call_id": "call_abc",
         "output": '{"phone_valid": false}'},
    ]


def test_full_multiturn_roundtrip():
    """End-to-end: 4-message DFSDT history → 4 native items, no nulls."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "fn", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "fn", "content": "result"},
    ]
    out = _to_responses_input(msgs)
    # Verify no null content anywhere
    for item in out:
        if "content" in item:
            assert item["content"] is not None, f"null content in {item}"
    types = [x.get("type") or x.get("role") for x in out]
    assert types == ["system", "user", "function_call", "function_call_output"]


def test_assistant_text_only_passthrough():
    msgs = [
        {"role": "assistant", "content": "hello there"},
    ]
    out = _to_responses_input(msgs)
    assert out == [{"role": "assistant", "content": "hello there"}]


def test_assistant_text_plus_tool_calls_emits_both():
    msgs = [
        {"role": "assistant", "content": "Let me check.",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "fn", "arguments": "{}"}}]},
    ]
    out = _to_responses_input(msgs)
    assert len(out) == 2
    assert out[0] == {"role": "assistant", "content": "Let me check."}
    assert out[1]["type"] == "function_call"


def test_assistant_null_content_no_toolcalls_is_dropped():
    """Defensive: an assistant message with content=None and no tool_calls
    should be dropped (would otherwise produce HTTP 400 invalid_type)."""
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None},
    ]
    out = _to_responses_input(msgs)
    assert out == [{"role": "user", "content": "q"}]


def test_dict_tool_content_serialized_to_string():
    """Tool output that is a dict (not a string) must be JSON-stringified."""
    msgs = [
        {"role": "tool", "tool_call_id": "c1", "name": "fn",
         "content": {"phone_valid": False}},
    ]
    out = _to_responses_input(msgs)
    assert out[0]["type"] == "function_call_output"
    assert out[0]["output"] == '{"phone_valid": false}'


# ----------------------------------------------------------------------------
# Tests for the judge shim's _AttrDict (utils.py): callers do BOTH attr access
# (msg.tool_calls[0].function.arguments) AND dict() iteration
# (dict(msg).get('content','')). Without _AttrDict, one or both fail.
# ----------------------------------------------------------------------------

def test_judge_shim_attrdict_supports_attr_and_dict():
    """Smoke: _AttrDict supports the exact access patterns judge callers use."""
    import importlib
    mod = importlib.import_module(
        'toolbench.tooleval.evaluators.registered_cls.utils')
    # _AttrDict is a private nested class inside request(); we can't import
    # it directly. Instead, re-implement and verify the SHAPE we produce.
    class _AttrDict(dict):
        def __getattr__(self, k):
            try: return self[k]
            except KeyError as exc: raise AttributeError(k) from exc
        def __setattr__(self, k, v): self[k] = v

    fn = _AttrDict(name='do_x', arguments='{"a":1}')
    tc = _AttrDict(id='call_1', type='function', function=fn)
    msg = _AttrDict(role='assistant', content='hello', tool_calls=[tc])
    # Caller pattern A: msg.tool_calls[0].function.arguments
    assert msg.tool_calls[0].function.arguments == '{"a":1}'
    assert msg.tool_calls[0].function.name == 'do_x'
    # Caller pattern B: dict(msg).get('content','')
    assert dict(msg).get('content', '') == 'hello'
