"""A minimal client for any OpenAI-compatible chat-completions server.  (README "The teacher")"""
from __future__ import annotations

from typing import Protocol, Sequence


class TeacherClient(Protocol):
    def chat(self, messages: Sequence[dict], n: int = 1, temperature: float = 0.7, max_tokens: int = 1500) -> list[str]: ...


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 300, session=None):
        """Store the settings. `session` defaults to requests.Session() (import requests lazily);
        the tests pass a fake object with a .post(url, json=..., headers=..., timeout=...) method."""
        raise NotImplementedError("TODO(m12): implement OpenAICompatClient.__init__")

    def chat(self, messages: Sequence[dict], n: int = 1, temperature: float = 0.7, max_tokens: int = 1500) -> list[str]:
        """Send `n` separate requests and return the n reply texts.

        Each request: POST f"{base_url.rstrip('/')}/chat/completions"
            json    = {"model": model, "messages": list(messages), "temperature": temperature,
                       "max_tokens": max_tokens, "n": 1}
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            timeout = timeout
        Call response.raise_for_status(), then take response.json()["choices"][0]["message"]["content"].
        """
        raise NotImplementedError("TODO(m12): implement OpenAICompatClient.chat")
