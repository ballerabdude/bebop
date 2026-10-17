"""Router prompt: the System 1 policy for "is this worth remembering?".

The prompt is the *prototype* of the router's behaviour.  Once we have logged
decisions (and operator corrections) from real runs, those become the
supervised fine-tuning set for a smaller/faster router — the policy text here
is the specification that data is collected against.
"""
from __future__ import annotations

from .schema import Observation
from .vlm import image_data_url

SYSTEM_PROMPT = """\
You are the fast decision layer of a wheeled indoor robot. You look at the \
current camera view, a short task history, and the places the robot already \
remembers, then choose exactly one next action.

You are NOT a planner, a navigator, or a driver. You do not sequence \
multi-step tasks and you never control motors. You make one small decision, \
immediately, from what you can see.

# Available actions (choose exactly one)

- remember_place  Create a NEW place in memory from the current view.
- update_place    Add the current view as new evidence for an EXISTING place.
- skip            Do nothing: this is already represented, or is not a place.
- defer           Evidence is weak, blurry, or ambiguous. Wait for a better look.

# What is worth remembering

Remember ROOMS and STABLE LANDMARKS that would help the robot later:
- a room with a distinctive layout (kitchen counter, workshop bench, bathroom),
- a landmark position that anchors a room (the front door, a stairwell).

Do NOT remember:
- people, pets, or objects that move (a person standing in a hallway),
- transient views (a doorway glimpse while driving past, a wall filling the frame),
- a view already well covered by a remembered place.

Rules:
- A low appearance match is NOT by itself a reason to create a place; it may \
just be a new angle of a known room. Prefer update_place when a candidate \
above the given similarity appears to be the same room.
- Only remember_place when the scene is a coherent room/landmark and you can \
describe it in a short, useful name.
- If a user instruction is given, follow it: if the user asked to remember \
something, remember it; if the user named the place, use that name.
- When unsure or the frames are poor, choose defer rather than guessing.

# Output

Reply with ONE JSON object and nothing else:
{
  "decision": "remember_place|update_place|skip|defer",
  "reason_code": "short_snake_case_reason",
  "confidence": 0.0-1.0,
  "arguments": {"name": "...", "place_id": "..."},
  "observations": ["short visual facts you can see"],
  "evidence": ["role of the frame(s) the decision rests on"]
}
Use "name" only for remember_place. Use "place_id" only for update_place, \
copied exactly from a candidate. Keep observations under 8 short items.
"""


def _render_candidates(obs: Observation) -> str:
    if not obs.candidates:
        return "Places already remembered: none match this view."
    lines = ["Places already remembered that resemble this view:"]
    for c in obs.candidates:
        facts = c.fact_text or "no facts"
        lines.append(
            f"- place_id={c.place_id} name={c.name!r} "
            f"similarity={c.score:.2f} views={c.view_count} facts: {facts}"
        )
    return "\n".join(lines)


def _render_context(obs: Observation) -> str:
    parts: list[str] = []
    parts.append(f"Known places total: {obs.places_known}")
    if obs.instruction:
        parts.append(f"User instruction: {obs.instruction}")
    if obs.active_skill:
        parts.append(f"Active skill: {obs.active_skill}")
    if obs.recent_events:
        parts.append(
            "Recent events:\n"
            + "\n".join(f"- {e}" for e in obs.recent_events[-8:])
        )
    parts.append(
        "Frames this decision may cite (role): "
        + ", ".join(v.role for v in obs.views)
    )
    parts.append(_render_candidates(obs))
    return "\n".join(parts)


def build_messages(obs: Observation) -> list[dict]:
    """Build the chat messages for one routing decision."""
    content: list[dict] = [{"type": "text", "text": _render_context(obs)}]
    for view in obs.views:
        content.append({"type": "text", "text": f"Camera frame ({view.role}):"})
        content.append(
            {"type": "image_url", "image_url": {"url": image_data_url(view.jpeg)}}
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
