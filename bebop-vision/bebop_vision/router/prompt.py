"""Router prompt: the System 1 policy for "is this worth remembering?".

The prompt is the *prototype* of the router's behaviour.  Once we have logged
decisions (and operator corrections) from real runs, those become the
supervised fine-tuning set for a smaller/faster router — the policy text here
is the specification that data is collected against.
"""
from __future__ import annotations

from .schema import CaptureContext, Observation
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


# --- map-capture skill -----------------------------------------------------

#: Longest image side sent to the brain.  Measured on the Thor: ~1.0 s at
#: 1280x800 (509 prompt tokens) vs ~834 ms at 640x400 (273 tokens), same
#: answer.  The brain is shared with live voice, so keep the image small.
CAPTURE_IMAGE_MAX_SIDE = 640

CAPTURE_SYSTEM_PROMPT = """\
You are the fast decision layer of a wheeled indoor robot. The robot has \
recognized the place it is standing in and built a live local structure map. \
Decide whether that map is worth persisting in the place's memory.

You are NOT a planner or a driver. You do not author geometry, coordinates, \
cells, walls or grids. The robot reads the live map itself after your answer; \
you only choose one action.

# Available actions (choose exactly one)

- capture_map    Store the FIRST map for this recognized place (it has none).
- refine_map     Add this capture's evidence to the place's existing map.
- reject_capture Do not store this capture. The live map is weak, transient, \
or not worth keeping.
- skip           Nothing to do: the stored map already covers this view.
- defer          Evidence is weak or ambiguous; wait for a better capture.

# Rules

- A map is worth keeping only when gravity and pose are valid, the grid is \
substantial, and the robot is standing still. Those checks already passed if \
you are being asked.
- Prefer capture_map when the place has no stored map and the view is coherent.
- Prefer refine_map when the place has a stored map and the live structure \
shows materially new area (cells gained), not when it merely repeats it.
- Choose reject_capture or skip when the map is thin, mostly empty, or a \
near-duplicate of what is already stored.
- Never invent place ids. Use the recognized place_id given below.
- Keep observations short and visual.

# Output

Reply with ONE JSON object and nothing else:
{
  "decision": "capture_map|refine_map|reject_capture|skip|defer",
  "reason_code": "short_snake_case_reason",
  "confidence": 0.0-1.0,
  "arguments": {"place_id": "..."},
  "observations": ["short facts about the live scene"]
}
"""


def downscale_jpeg(jpeg: bytes, max_side: int = CAPTURE_IMAGE_MAX_SIDE,
                   quality: int = 80) -> bytes:
    """Shrink a JPEG so its longest side is at most ``max_side`` pixels.

    The vision brain's image latency scales with prompt tokens, and it is
    shared with the live voice cascade.  Falls back to the original bytes if
    the frame cannot be decoded (or cv2 is unavailable).
    """
    try:
        import cv2
        import numpy as np

        buf = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            return jpeg
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= int(max_side) or longest <= 0:
            return jpeg
        scale = float(max_side) / float(longest)
        resized = cv2.resize(
            image,
            (max(1, int(round(width * scale))),
             max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        ok, out = cv2.imencode(
            ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return out.tobytes() if ok else jpeg
    except Exception:  # noqa: BLE001 - never fail a decision over resizing
        return jpeg


def _preferred_capture_view(views):
    for view in views:
        if view.role == "far":
            return view
    return views[0] if views else None


def _render_capture_context(ctx: CaptureContext) -> str:
    s = ctx.structure
    r = ctx.recognition
    lines = [
        f"Recognized place: place_id={ctx.place_id} name={ctx.place_name!r} "
        f"(similarity={r.score:.2f}, margin={r.margin:.2f})",
        f"Stored map: {'yes' if ctx.has_map else 'no'}"
        + (f" revision={ctx.map_revision}" if ctx.has_map else "")
        + (f" age={ctx.map_age_s:.0f}s" if ctx.map_age_s is not None else "")
        + f" known_cells={ctx.map_known_cells}",
        "Live structure snapshot:",
        f"- gravity_valid={s.gravity_valid} pose_valid={s.pose_valid}",
        f"- pose x={s.pose_x} y={s.pose_y} yaw_deg={s.pose_yaw_deg}",
        f"- occupied_cells={s.occupied_cells} free_cells={s.free_cells} "
        f"known_cells={s.known_cells}",
        f"- walls={s.wall_count}",
        f"- pose change while sampling: {s.pose_delta_m:.3f} m, "
        f"{s.pose_delta_yaw_deg:.2f} deg",
        f"- cells gained since the stored map: {ctx.cells_gained}",
    ]
    return "\n".join(lines)


def build_capture_messages(ctx: CaptureContext, views) -> list[dict]:
    """Build the chat messages for one map-capture decision.

    One downscaled camera frame is attached (the forward view when present);
    the live snapshot itself is summarized as text.
    """
    content: list[dict] = [{"type": "text",
                            "text": _render_capture_context(ctx)}]
    view = _preferred_capture_view(list(views or []))
    if view is not None:
        content.append({"type": "text", "text": f"Camera frame ({view.role}):"})
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url(downscale_jpeg(view.jpeg))},
        })
    return [
        {"role": "system", "content": CAPTURE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
