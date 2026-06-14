"""
Unified ModelClient abstraction for FitText.

Public API
----------
ModelClient       – Abstract base class for all provider clients.
NormalizedResponse – OpenAI-shaped response container with telemetry fields.
make_client        – Factory: ``make_client(model, **kwargs) -> ModelClient``.

Usage example
-------------
>>> from toolbench.inference.LLM.clients import make_client
>>> client = make_client(model)
>>> resp = await client.chat_completion(messages=[...], tools=[...])
>>> message = resp.to_openai_dict()["choices"][0]["message"]
"""

from .base import ModelClient, NormalizedResponse
from .factory import make_client

__all__ = [
    "ModelClient",
    "NormalizedResponse",
    "make_client",
]
