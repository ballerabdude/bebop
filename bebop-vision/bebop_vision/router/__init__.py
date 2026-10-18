"""System 1 visual skill router.

A fast, bounded decision layer that looks at a small observation bundle (a
few camera frames, task context, retrieved memories) and picks the next
skill.  It is not a planner and not a controller: it emits one structured
decision, and an executor applies it.

First skill: **place memory** — "should I remember this place?"
Second skill: **map capture** — "should I persist this local structure map?"

    from bebop_vision.router import (
        CaptureMapExecutor, CaptureMapSkill, ChatVLMClient,
        PlaceMemoryExecutor, PlaceMemorySkill, RouterService, VoiceBusyProbe,
    )

    vlm = ChatVLMClient("http://127.0.0.1:8012", model="RadixArk/Qwen3.8-27B-NVFP4")
    place_skill = PlaceMemorySkill(store, encoder, vlm)
    place_exec = PlaceMemoryExecutor(store, encoder)
    capture_skill = CaptureMapSkill(store, encoder, vlm)   # :9096/structure
    capture_exec = CaptureMapExecutor(store)

    router = RouterService(
        place_skill, place_exec, capture=capture,
        skills=[(capture_skill, capture_exec)],            # extra registrations
        voice_busy=VoiceBusyProbe(),                       # :9093, fail-open
    )
    router.start()

`RouterService(skill, executor, ...)` is still the single-skill form.  Each
registration has its own cooldown (and an optional `event_trigger` hook for a
future skill); the tick thread, the one-in-flight guard, the event history and
the decision log are shared.  Place memory and map capture are both assessed
on every tick, so the brain weighs in even on an unchanged scene.  The
map-capture executor fetches the live structure snapshot itself and never
accepts a model-authored geometry payload.  A manual assessment can target a
registration (`assess_once(skill="capture_map")`, or `{"skill": ...}` /
`?skill=...` on `/router/assess`), and `status()["skills"][i]["gate"]` exposes
the cached capture gate without probing.

See `docs/place-router.md`.
"""
from .capture_map import (
    DEFAULT_STRUCTURE_URL,
    CaptureAssessment,
    CaptureMapExecutor,
    CaptureMapSkill,
    fetch_structure,
    summarize_structure,
)
from .place_memory import PlaceAssessment, PlaceMemoryExecutor, PlaceMemorySkill
from .prompt import (
    CAPTURE_IMAGE_MAX_SIDE,
    CAPTURE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_capture_messages,
    build_messages,
    downscale_jpeg,
)
from .schema import (
    ALL_DECISIONS,
    CAPTURE_DECISIONS,
    CAPTURE_MAP,
    DEFER,
    PLACE_MEMORY_DECISIONS,
    REFINE_MAP,
    REJECT_CAPTURE,
    REMEMBER_PLACE,
    SKIP,
    UPDATE_PLACE,
    Candidate,
    CaptureContext,
    DecisionParseError,
    Observation,
    Recognition,
    SkillDecision,
    StructureSummary,
    View,
    defer_fallback,
    extract_json_object,
    normalize_decision,
    parse_decision,
)
from .service import RouterService, SkillRegistration
from .vlm import (
    ChatVLMClient,
    ScriptedVLMClient,
    VLMClient,
    VLMError,
    image_data_url,
)
from .voice_busy import BUSY_KEYS, DEFAULT_VOICE_URL, VoiceBusyProbe

__all__ = [
    "ChatVLMClient",
    "ScriptedVLMClient",
    "VLMClient",
    "VLMError",
    "image_data_url",
    "PlaceMemorySkill",
    "PlaceMemoryExecutor",
    "PlaceAssessment",
    "CaptureMapSkill",
    "CaptureMapExecutor",
    "CaptureAssessment",
    "fetch_structure",
    "summarize_structure",
    "DEFAULT_STRUCTURE_URL",
    "RouterService",
    "SkillRegistration",
    "VoiceBusyProbe",
    "DEFAULT_VOICE_URL",
    "BUSY_KEYS",
    "Observation",
    "CaptureContext",
    "StructureSummary",
    "Recognition",
    "View",
    "Candidate",
    "SkillDecision",
    "DecisionParseError",
    "parse_decision",
    "extract_json_object",
    "normalize_decision",
    "defer_fallback",
    "build_messages",
    "build_capture_messages",
    "downscale_jpeg",
    "CAPTURE_IMAGE_MAX_SIDE",
    "SYSTEM_PROMPT",
    "CAPTURE_SYSTEM_PROMPT",
    "PLACE_MEMORY_DECISIONS",
    "CAPTURE_DECISIONS",
    "ALL_DECISIONS",
    "REMEMBER_PLACE",
    "UPDATE_PLACE",
    "CAPTURE_MAP",
    "REFINE_MAP",
    "REJECT_CAPTURE",
    "SKIP",
    "DEFER",
]
