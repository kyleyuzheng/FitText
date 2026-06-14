import os
import json
import sys
from pathlib import Path
from typing import List,Dict
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt

from openai import OpenAI
import random

_stable_toolbench_root = Path(__file__).resolve().parents[4]
if str(_stable_toolbench_root) not in sys.path:
    sys.path.insert(0, str(_stable_toolbench_root))

__registered_evaluators__ = {}

def register_evaluator(cls):
    """
    Decorator function to register classes with the registered_evaluators list.
    """
    __registered_evaluators__[cls.__name__] = cls
    return cls

def get_evaluator_cls(clsname):
    """
    Return the evaluator class with the given name.
    """
    try:
        return __registered_evaluators__.get(clsname)
    except Exception:
        raise ModuleNotFoundError('Cannot find evaluator class {}'.format(clsname))


class OpenaiPoolRequest:
    def __init__(self, pool_json_file=None):
        self.pool:List[Dict] = []
        # Guarantee now_pos is set before any request(). Upstream code only sets
        # it inside the API_POOL_FILE or path-exists branches; under the fallback
        # path (only an env-var key set, no pool file on disk) the attribute was
        # never initialized → AttributeError when concurrent threads call
        # request(). -1 is the correct seed because request() does
        # `(now_pos + 1) % len(pool)`, yielding 0 first.
        self.now_pos = -1
        __pool_file = pool_json_file
        if os.environ.get('API_POOL_FILE',None) is not None:
            __pool_file = os.environ.get('API_POOL_FILE')
            self.now_pos = random.randint(-1, len(self.pool))
        if __pool_file and os.path.exists(__pool_file):
            self.pool = json.load(open(__pool_file))
            self.now_pos = random.randint(-1, len(self.pool))
        # Single-key fallback: accept OPENAI_KEY (tooleval's historical variable)
        # or OPENAI_API_KEY (used by the rest of the pipeline) so either works.
        _key = os.environ.get('OPENAI_KEY') or os.environ.get('OPENAI_API_KEY')
        if _key is not None:
            self.pool.append({
                'api_key': _key,
                'organization':os.environ.get('OPENAI_ORG',None),
                'api_type':os.environ.get('OPENAI_TYPE',None),
                'api_version':os.environ.get('OPENAI_VER',None)
            })

    # @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(10),reraise=True)
    def request(self,messages,**kwargs):
        self.now_pos = (self.now_pos + 1) % len(self.pool)
        key_pos = self.now_pos
        item = self.pool[key_pos]
        api_key = item['api_key']
        api_base = item.get('api_base', None)
        client = OpenAI(api_key=api_key,base_url=api_base)
        # Route reasoning models through the Responses API.
        # Chat Completions silently disables reasoning when reasoning_effort
        # is omitted, AND rejects reasoning_effort when tools are present.
        # Responses API has neither limitation + better cache utilization.
        model = kwargs.get('model', '')
        _is_reasoning = isinstance(model, str) and model.startswith(('gpt-5', 'o1', 'o3', 'o4'))
        if _is_reasoning:
            # Strip Chat-Completions-specific params + remap names.
            _kw = {k: v for k, v in kwargs.items()
                   if k not in ('temperature', 'top_p', 'response_format', 'max_tokens')}
            # Map max_tokens → max_output_tokens (≥25K reasoning floor).
            _max_out = max(int(kwargs.get('max_tokens') or 0), 25000)
            _kw['max_output_tokens'] = _max_out
            # Convert tools shape (nested function → flat) if present.
            if 'tools' in _kw and isinstance(_kw['tools'], list):
                _flat = []
                for t in _kw['tools']:
                    if isinstance(t, dict) and t.get('type') == 'function' and 'function' in t:
                        f = t['function']
                        _flat.append({'type': 'function', 'name': f.get('name',''),
                                     'description': f.get('description',''),
                                     'parameters': f.get('parameters', {})})
                    else:
                        _flat.append(t)
                _kw['tools'] = _flat
            # Convert tool_choice the same way. Chat:
            # {"type":"function","function":{"name":N}} →
            # Responses: {"type":"function","name":N}. Without this, Responses
            # API returns 400 `Unknown parameter: tool_choice.function`.
            if 'tool_choice' in _kw and isinstance(_kw['tool_choice'], dict):
                _tc = _kw['tool_choice']
                if _tc.get('type') == 'function' and 'function' in _tc:
                    _kw['tool_choice'] = {'type': 'function',
                                          'name': _tc['function'].get('name', '')}
            # Convert Chat-Completions message history → Responses-API input items.
            # Responses API rejects assistant.content=None + assistant.tool_calls
            # embedding. Reuse the converter from the agent-side OpenAI client.
            from toolbench.inference.LLM.clients.openai_client import _to_responses_input
            _input = _to_responses_input(messages)
            r = client.responses.create(input=_input, **_kw)
            # Callers in tooleval/registered_cls/tooleval.py access the result
            # via attribute chains AND dict(message) — BOTH of:
            #   - res.choices[0].message.tool_calls[0].function.arguments
            #     (attribute chain on tool_call → function → arguments)
            #   - dict(res.choices[0].message).get('content','')
            #     (dict(message) iteration)
            # So the shim must produce attribute-accessible objects AND a
            # dict-iterable message. _AttrDict satisfies both.
            class _AttrDict(dict):
                """Dict that also supports attribute access on its keys."""
                def __getattr__(self, k):
                    try:
                        return self[k]
                    except KeyError as exc:
                        raise AttributeError(k) from exc
                def __setattr__(self, k, v):
                    self[k] = v
            parts = []
            tool_calls = []
            for item in (r.output or []):
                if getattr(item, 'type', None) == 'message':
                    for c in (getattr(item, 'content', []) or []):
                        t = getattr(c, 'text', None)
                        if t:
                            parts.append(t)
                elif getattr(item, 'type', None) == 'function_call':
                    fn = _AttrDict(
                        name=getattr(item, 'name', ''),
                        arguments=getattr(item, 'arguments', '{}'),
                    )
                    tc = _AttrDict(
                        id=getattr(item, 'call_id', None) or getattr(item, 'id', None),
                        type='function',
                        function=fn,
                    )
                    tool_calls.append(tc)
            msg = _AttrDict(
                role='assistant',
                content=''.join(parts) if parts else '',
                tool_calls=tool_calls or None,
            )
            ch = _AttrDict(message=msg, finish_reason='stop', index=0)
            usage = getattr(r, 'usage', None)
            u = _AttrDict(
                prompt_tokens=getattr(usage, 'input_tokens', 0) if usage else 0,
                completion_tokens=getattr(usage, 'output_tokens', 0) if usage else 0,
            )
            u.total_tokens = u.prompt_tokens + u.completion_tokens
            r_shim = _AttrDict(
                choices=[ch],
                usage=u,
                model=getattr(r, 'model', model),
                id=getattr(r, 'id', ''),
            )
            return r_shim
        # Legacy path for non-reasoning chat models.
        response = client.chat.completions.create(messages=messages,**kwargs)
        return response
    
    def __call__(self,messages,**kwargs):
        return self.request(messages,**kwargs)
