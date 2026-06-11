"""LLM backbones.

OllamaLLM talks to a local Ollama server over its REST API using only the
standard library, so the framework has zero runtime dependencies. MockLLM
provides scripted responses for tests and offline prototyping.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class OllamaConnectionError(RuntimeError):
    pass


class BaseLLM:
    """Interface for chat-capable language models."""

    def chat(self, messages: List[Dict[str, str]], **options: Any) -> str:
        raise NotImplementedError

    def ask(self, prompt: str, system: Optional[str] = None, **options: Any) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **options)


class OllamaLLM(BaseLLM):
    """Chat with a model served by Ollama (https://ollama.com).

    Example:
        llm = OllamaLLM(model="llama3.1")
        llm.ask("Say hi")
    """

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        timeout: float = 300.0,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: str = "15m",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.options = dict(options or {})
        # Keep the model resident between calls; servers configured with
        # OLLAMA_KEEP_ALIVE=0 otherwise reload the model cold on every request.
        self.keep_alive = keep_alive

    def chat(self, messages: List[Dict[str, str]], **options: Any) -> str:
        opts = {"temperature": self.temperature, **self.options, **options}
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": opts,
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.host} ({exc}). "
                "Is the server running? Try: ollama serve"
            ) from exc
        return body["message"]["content"]

    def __repr__(self) -> str:
        return f"OllamaLLM(model={self.model!r}, host={self.host!r})"


class MockLLM(BaseLLM):
    """Scripted LLM for tests and offline prototyping.

    Responses are returned in order; the last one repeats once exhausted.
    All received message lists are recorded in `.calls` for inspection.
    """

    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = list(responses or [])
        self.calls: List[List[Dict[str, str]]] = []
        self._index = 0

    def chat(self, messages: List[Dict[str, str]], **options: Any) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise RuntimeError("MockLLM has no scripted responses")
        response = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return response
