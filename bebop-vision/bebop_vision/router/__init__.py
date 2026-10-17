"""System 1 visual skill router.

A fast, bounded decision layer that looks at a small observation bundle (a
few camera frames, task context, retrieved memories) and picks the next
skill.  It is not a planner and not a controller: it emits one structured
decision, and an executor applies it.

First skill: **place memory** — "should I remember this place?"

    from bebop_vision.router import (
        ChatVLMClient, PlaceMemoryExecutor, PlaceMemorySkill, RouterService,
    )

    vlm = ChatVLMClient("http://127.0.0.1:8012", model="RadixArk/Qwen3.8-27B-NVFP4")
    skill = PlaceMemorySkill(store, encoder, vlm)
    executor = PlaceMemoryExecutor(store, encoder)
    router = RouterService(skill, executor, capture=capture)
    router.start()

See `docs/place-router.md`.
"""
from .place_memory import PlaceAssessment, PlaceMemoryExecutor, PlaceMemorySkill
from .prompt import SYSTEM_PROMPT, build_messages
from .schema import (
    DEFER,
    PLACE_MEMORY_DECISIONS,
    REMEMBER_PLACE,
    SKIP,
    UPDATE_PLACE,
    Candidate,
    DecisionParseError,
    Observation,
    SkillDecision,
    View,
    defer_fallback,
    extract_json_object,
    normalize_decision,
    parse_decision,
)
from .service import RouterService
from .vlm import (
    ChatVLMClient,
    ScriptedVLMClient,
    VLMClient,
    VLMError,
    image_data_url,
)

__all__ = [
    "ChatVLMClient",
    "ScriptedVLMClient",
    "VLMClient",
    "VLMError",
    "image_data_url",
    "PlaceMemorySkill",
    "PlaceMemoryExecutor",
    "PlaceAssessment",
    "RouterService",
    "Observation",
    "View",
    "Candidate",
    "SkillDecision",
    "DecisionParseError",
    "parse_decision",
    "extract_json_object",
    "normalize_decision",
    "defer_fallback",
    "build_messages",
    "SYSTEM_PROMPT",
    "PLACE_MEMORY_DECISIONS",
    "REMEMBER_PLACE",
    "UPDATE_PLACE",
    "SKIP",
    "DEFER",
]
