"""VLM clients for the router.

The router talks to whatever vision-language model is resident on the Thor
through an **OpenAI-compatible** `/v1/chat/completions` endpoint — the same
surface `voice_server.py` already uses for the Qwen3.8-27B NVFP4 brain. That
keeps the router decoupled from how the model is served (a TensorRT Edge-LLM
child process, or the `bebop-models` supervisor).

`VLMClient` is a tiny protocol so tests can substitute a scripted reply and
so a smaller/faster backend can be swapped in later without touching the
skill or the service.
"""
from __future__ import annotations

import base64
from typing import Any, Protocol, runtime_checkable


class VLMError(RuntimeError):
    """The backend could not produce a reply."""


@runtime_checkable
class VLMClient(Protocol):
    """A chat-completions style VLM."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str: ...

    @property
    def describe(self) -> str: ...


def image_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


class ChatVLMClient:
    """OpenAI-compatible `/v1/chat/completions` client (sync, bounded)."""

    def __init__(
        self,
        base_url: str,
        model: str = "",
        *,
        api_key: str = "not-needed",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)

    @property
    def describe(self) -> str:
        return f"chat:{self.base_url} model={self.model or '(default)'}"

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        import httpx

        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if self.model:
            payload["model"] = self.model
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001 - normalised for the skill
            raise VLMError(f"{type(exc).__name__}: {exc}") from exc
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMError(f"malformed completion: {str(data)[:200]}") from exc
        return str(msg.get("content") or "").strip()


class ScriptedVLMClient:
    """Test double: returns queued replies in order (then repeats the last)."""

    def __init__(self, replies: list[str] | str, *, model: str = "scripted") -> None:
        self.replies = [replies] if isinstance(replies, str) else list(replies)
        self.model = model
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def describe(self) -> str:
        return f"scripted model={self.model}"

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(messages)
        if not self.replies:
            return ""
        if len(self.replies) == 1:
            return self.replies[0]
        return self.replies.pop(0)
