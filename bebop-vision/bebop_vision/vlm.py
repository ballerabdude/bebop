"""OpenAI-compatible chat client for the vision-language model.

The places importer asks the resident VLM for a pending place's name and
facts (`POST /places/pending/{id}/suggest`, operator-requested only) through
an **OpenAI-compatible** `/v1/chat/completions` endpoint — the same surface
`voice_server.py` uses for the Qwen3.8-27B NVFP4 brain. That keeps the
caller decoupled from how the model is served (a TensorRT Edge-LLM child
process, or the `bebop-models` supervisor).
"""
from __future__ import annotations

from typing import Any


class VLMError(RuntimeError):
    """The backend could not produce a reply."""


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
        except Exception as exc:  # noqa: BLE001 - normalised for the caller
            raise VLMError(f"{type(exc).__name__}: {exc}") from exc
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMError(f"malformed completion: {str(data)[:200]}") from exc
        return str(msg.get("content") or "").strip()
