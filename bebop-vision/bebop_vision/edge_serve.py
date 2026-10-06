"""Launcher for the TensorRT Edge-LLM OpenAI server, with small runtime patches.

Drop-in replacement for the ``tensorrt-edgellm-serve`` console script: it takes
the same arguments and runs the same server, but first applies compatibility
patches for bugs in the installed Edge-LLM build.

Patch (Edge-LLM 0.11.0):
    ``experimental.server.runtime.engine.load_model`` routes a ``qwen3_tts``
    checkpoint to the standalone ``TTS`` runtime, but forgets to strip the
    VLM-only ``max_image_tokens`` / ``max_image_tokens_per_image`` kwargs that
    the server always forwards. ``TTS.__init__`` does not accept them, so the
    server dies with ``TypeError: TTS.__init__() got an unexpected keyword
    argument 'max_image_tokens'`` before it can serve ``/v1/audio/speech``.

    We strip those two kwargs inside ``TTS.__init__`` (guarded by a marker so
    it is idempotent). If a future Edge-LLM release fixes it upstream, the
    patch is a harmless no-op.

Usage::

    python -m bebop_vision.edge_serve <model> --cache-dir ... --port ... [...]
"""

from __future__ import annotations


def _patch_tts_kwargs() -> None:
    try:
        from experimental.server.runtime.engine import TTS
    except Exception:  # noqa: BLE001 - server not importable; let it fail loudly
        return

    original = TTS.__init__
    if getattr(original, "_bebop_patched", False):
        return

    def __init__(self, *args, **kwargs):  # noqa: N807 - matches __init__
        kwargs.pop("max_image_tokens", None)
        kwargs.pop("max_image_tokens_per_image", None)
        return original(self, *args, **kwargs)

    __init__._bebop_patched = True  # type: ignore[attr-defined]
    TTS.__init__ = __init__  # type: ignore[method-assign]


def main() -> None:
    _patch_tts_kwargs()
    from experimental.server.cli import main as serve_main

    serve_main()


if __name__ == "__main__":
    main()
