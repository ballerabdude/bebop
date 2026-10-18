# System 1 place-memory router

> **Default path (2026-10-18): lofi scout + operator import.** Place capture
> now runs as a cheap DINOv3 scout (about 1 Hz, no LLM) that writes **pending**
> places. The operator reviews a batch in the app, optionally asks the model
> for a name suggestion, edits it, and confirms the import. The LLM is never
> called on the tick. `--place-router` below is the legacy every-tick router;
> it remains available for experiments but costs an LLM call per tick (measured
> ~12 W average at a 45 s interval) and is off in normal runs. See
> [Pending places and batch import](#pending-places-and-batch-import).

A fast, bounded **visual decision layer** for the robot. It looks at a small
observation bundle — a couple of camera frames, a short task history, and the
places already in memory — and picks one next action.

It is **not** a planner and **not** a controller. It emits one structured
decision; an executor applies it. `RouterService` holds one or more
`(skill, executor)` registrations and evaluates them in turn on a single loop.
The router owns two skills:

- **place memory** — *"should I remember this place?"*
- **map capture** — *"should I persist this local structure map for the place
  I am standing in?"* (capture / refine / reject, decided by the brain on
  **every tick**, with a deterministic gate that acts as a write
  precondition).

Under `--place-router` both are registered: place memory always, map capture
when `--live-slam` is on (it needs `:9096/structure`).

```
 far/near JPEGs   task context   recent events   DINOv3 candidates
        └──────────────┬──────────────┴──────────────┘
                       ▼
              RouterService (one tick, one decision)
                       ▼
        ┌──────────────┴───────────────┐
        ▼                              ▼
 PlaceMemorySkill              CaptureMapSkill
 (place memory)                (structure-map quality)
        └──────────────┬───────────────┘
                       ▼
         VLM (Qwen3.8-27B NVFP4, OpenAI API)
                       ▼
 remember_place · update_place · capture_map ·
   refine_map · reject_capture · skip · defer
                       ▼
              Executor ──► Places SQLite store
```

## Why a VLM, and why this one

Choosing *"is this worth remembering"* needs visual meaning the DINOv3
descriptor alone cannot give: it can say *"this looks different from every
place I know"*, not *"this is a new room"* vs *"a doorway glimpse"* vs *"a
different corner of the kitchen"*.

The router talks to any **OpenAI-compatible** `/v1/chat/completions` endpoint.
On the Thor the default is the voice-cascade **brain**,
`RadixArk/Qwen3.8-27B-NVFP4` (multimodal, 256k) on `:8012` — the same model
`docs/voice.md` already wires for the cascade. This is the prototype/teacher
model; a smaller router should be distilled from the decisions it logs (see
*Learning from the log*).

> The backend must be **resident** when the router tick runs. The cascade
> brain is an Edge-LLM child of `bebop-voice`; either keep the cascade up, or
> point `--router-url` at a supervisor-managed backend
> (`docs/models.md`) — the router only needs the URL.

## Files

| Path | Role |
|---|---|
| `bebop_vision/router/schema.py` | Decision vocabulary, observation bundle, JSON parsing |
| `bebop_vision/router/prompt.py` | The System 1 policy ("what is worth remembering") |
| `bebop_vision/router/vlm.py` | `ChatVLMClient` (OpenAI), `ScriptedVLMClient` (tests) |
| `bebop_vision/router/place_memory.py` | `PlaceMemorySkill` (decide) + `PlaceMemoryExecutor` (act) |
| `bebop_vision/router/capture_map.py` | `CaptureMapSkill` (decide + write gate) + `CaptureMapExecutor` (fetch + merge) |
| `bebop_vision/router/service.py` | `RouterService`: multi-skill registrations, one in-flight tick, event gating, voice guard, decision log |
| `bebop_vision/router/voice_busy.py` | Fail-open `VoiceBusyProbe` for `:9093/healthz` |
| `bebop_vision/places/scout.py` | Lofi pending-place capture (DINOv3 + thresholds, no LLM) |
| `bebop_vision/places/importer.py` | Import suggestions (one model call per place) and deterministic promotion/merge |
| `bebop_vision/places/server.py` | `/places/pending`, `/places/pending/{id}/suggest`, `/places/pending/import`, `/router/*` |
| `tests/test_router.py` | 44 tests, no torch / camera / VLM server needed |

## Decisions

| Decision | Skill | Meaning |
|---|---|---|
| `remember_place` | place memory | Create a new place (`arguments.name`) |
| `update_place` | place memory | Add this view as evidence for an existing place (`arguments.place_id`) |
| `capture_map` | map capture | Store the first accumulated structure map for the recognized place |
| `refine_map` | map capture | Add this capture's evidence to the place's existing accumulated map |
| `reject_capture` | map capture | Do not persist this capture (writes nothing) |
| `skip` | both | Already represented / nothing worth doing |
| `defer` | both | Evidence weak/ambiguous — look again later |

Reply shape (the model emits exactly this JSON):

```json
{
  "decision": "capture_map",
  "reason_code": "new_room_map",
  "confidence": 0.8,
  "arguments": {"place_id": "lab-3f9a1c"},
  "observations": ["a coherent lab bench and two walls"],
  "evidence": ["far"]
}
```

Aliases (`create_place`, `enrich`, `capture`, `refine`, `discard`, `wait`, …)
are normalised. Anything an executor could not safely apply is turned into a
`defer` with a reason: `unknown_decision`, `missing_name`, `unknown_place`,
`insufficient_evidence`, `unparseable_output`, `vlm_error`. The capture gate
adds `gate_gravity_invalid`, `gate_pose_invalid`, `gate_too_few_cells`,
`gate_moving`, `gate_not_recognized_here`, `gate_unknown_place`, and
`gate_no_structure`; the executor adds `cooldown`, `no_structure`, and
`invalid_structure`.

## Run it

The normal lofi path needs no router flag: the scout starts with the places
service. The VLM URL/model are only used by an operator-requested import
suggestion.

```bash
# on the Thor, from ~/bebop/bebop-vision
sudo .venv/bin/python -u main.py --record-navd /var/lib/bebop-captures --auto \
  --live-slam --live-slam-map
```

The legacy every-tick router is off unless `--place-router` is passed:

```bash
sudo .venv/bin/python -u main.py --record-navd /var/lib/bebop-captures --auto \
  --live-slam --live-slam-map --place-router --router-url http://127.0.0.1:8012
```

Legacy flags: `--router-interval` (default 20 s), `--router-cooldown`
(default 180 s), `--router-timeout` (default 120 s), `--router-model`,
`--router-voice-url` (default `http://127.0.0.1:9093/healthz`). Scout flags:
`--place-scout-interval` (default 1 s), `--place-pending-max` (default 50),
`--place-pending-ttl-days` (default 14).

With `--place-router`, `main.py` constructs the legacy router exactly as
follows (the lofi path instead constructs `PlaceScout` + `PlaceImporter` and
passes them to `PlacesService`):

```python
vlm = ChatVLMClient(args.router_url, args.router_model,
                    timeout=args.router_timeout)
place_skill = PlaceMemorySkill(pstore, penc, vlm)
place_exec = PlaceMemoryExecutor(pstore, penc, cooldown_s=args.router_cooldown)

structure_url = f"http://127.0.0.1:{args.live_slam_port}/structure"
capture_skill = CaptureMapSkill(pstore, penc, vlm,
                                structure_url=structure_url,
                                cooldown_s=args.router_cooldown)
capture_exec = CaptureMapExecutor(pstore, structure_url=structure_url,
                                  cooldown_s=args.router_cooldown)

router = RouterService(
    place_skill, place_exec, capture=places.capture,
    skills=[(capture_skill, capture_exec)],     # extra registrations
    voice_busy=VoiceBusyProbe(args.router_voice_url),
    interval_s=args.router_interval,
)
router.start()
```

`RouterService(skill, executor, ...)` remains the single-skill constructor;
`skills=[...]`, a bare list of pairs, or `SkillRegistration` objects add more.
Every registration has its own cooldown, while the tick thread, in-flight
guard, event history and decision log are shared.  Both skills are assessed
on every tick; an optional `event_trigger` hook remains for a future skill
that genuinely needs event gating.

`RouterService.tick()` performs one background evaluation of every
registration and is the deterministic entry point used by tests;
`assess_once()` remains the manual "decide now" path behind
`POST /router/assess` (primary skill unless `skill=` is passed).

### Inspect it

```bash
curl -s localhost:9095/router/status            # running, counters, per-skill gate
curl -s 'localhost:9095/router/decisions?limit=10'

# One decision now, primary skill (place memory):
curl -s -X POST localhost:9095/router/assess \
     -H 'Content-Type: application/json' \
     -d '{"instruction": "remember this as the workshop"}'

# Exercise the capture skill now instead of waiting for the next tick:
curl -s -X POST localhost:9095/router/assess \
     -H 'Content-Type: application/json' \
     -d '{"skill": "capture_map"}'
curl -s -X POST 'localhost:9095/router/assess?skill=capture_map'
```

`{"skill": "<name>"}` (or `?skill=<name>`) selects a registration for a
manual assessment; the body form wins when both are present. An unknown name
is a **400** whose message lists the registered names, e.g.
`unknown skill 'nope'; available: place_memory, capture_map`. Omitting
`skill` keeps the original behaviour: the primary (place memory) skill.

### Reading the capture gate

`GET /router/status` is a read-only view of the last gate evaluation for every
skill that exposes a gate. The gate is refreshed by **every assessment** (and
therefore every tick); it never triggers a probe or a brain call itself, so a
status read is cheap.

```json
{"skills": [
  {"name": "place_memory", "event_gated": false, "last_trigger": null},
  {"name": "capture_map", "event_gated": false, "last_trigger": null,
   "gate": {
     "evaluated": true,
     "gate_ok": true,
     "gate_reason": "",
     "place_id": "lab-3f9a1c",
     "stable_for_s": 40.2, "place_stable_s": 10.0,
     "place_changed": false,
     "cells_gained": 12, "min_cells_gained": 100,
     "in_cooldown": false, "cooldown_s": 180.0,
     "cooldown_remaining_s": 0.0,
     "age_s": 1.7
   }}
]}
```

Fields worth reading first:

- `gate_ok` / `gate_reason` — the deterministic write precondition
  (`gravity_invalid`, `pose_invalid`, `too_few_cells`, `robot_moving`,
  `not_recognized_here`, `unknown_place`, `no_structure`). When false the
  brain is still asked, but its decision is degraded to a `gate_*` defer and
  nothing is written.
- `place_id`, `stable_for_s` vs `place_stable_s` — telemetry: how long the
  current place has been continuously visible.
- `place_changed` — whether this place differs from the last recognized one.
- `cells_gained` vs `min_cells_gained` — telemetry: live known cells minus
  the stored map's known cells.
- `in_cooldown` / `cooldown_remaining_s` — the per-place write cooldown; a
  `capture_map` decision inside the window is refused by the executor.
- `age_s` — how stale this cached snapshot is since the last assessment.

`place_stable_s` and `min_cells_gained` are retained as tuning telemetry;
they no longer suppress the prompt. The exact thresholds in force are echoed
in `gate.thresholds` and in each skill's entries.

## Write guards (the executor)

- **Creation cooldown** — a second `remember_place` within the cooldown
  becomes an `update_place` of the place just created (`cooldown_dedupe`), so a
  slow walk through one room does not spawn a place per tick.
- **Duplicate name** — folds into the existing place (`deduped_by_name`).
- **Growth caps** — `max_views_per_call`, `max_keyframes_per_place`,
  `max_facts_per_call`.
- **Facts are attributed** — visual observations are stored with source
  `router`; a user instruction with source `operator`. An inferred name stays
  distinguishable from an operator-provided one.

The decision layer writes nothing; only the executor touches the store.

## Map-capture skill

`CaptureMapSkill` decides whether the live local structure map should be
persisted for the place the robot is recognized in. It is deliberately split
from place memory because its evidence is metric (gravity, pose, cell counts,
walls), not appearance.

### Input bundle (prompt-sized, never the raw payload)

- live `:9096/structure` summary: gravity valid?, pose valid?, pose, occupied
  and free cell counts, wall count, pose change while sampling;
- the `/whereami` recognition for this frame: place id/name, similarity and
  margin over the runner-up;
- the place's stored map: present?, `revision`, age, known cells, and cells
  gained since that stored map.

One camera frame is attached, downscaled to **640 px on the longest side**
(see the latency note below).

### Deterministic gate (write precondition)

The capture skill is **prompted every tick**, like place memory. The gate is
still evaluated first, but it no longer decides whether to ask the brain — it
decides whether a write is allowed. If the model chooses `capture_map` or
`refine_map` while the gate is closed, the decision is degraded to `defer`
with an explicit `gate_*` reason and the executor writes nothing:

| Gate check | Defer reason |
|---|---|
| gravity valid | `gate_gravity_invalid` |
| pose valid | `gate_pose_invalid` |
| minimum known cells (`min_cells`) | `gate_too_few_cells` |
| robot not moving (`pose_delta_m` / yaw, two samples `motion_window_s` apart) | `gate_moving` |
| target is the place recognized as "here" | `gate_not_recognized_here` |
| recognized place exists in the store | `gate_unknown_place` |
| a structure snapshot was available | `gate_no_structure` |

The model itself chooses among `capture_map | refine_map | reject_capture |
skip | defer`. The gate reason is recorded in `/router/decisions` as both the
`gate_*` `reason_code` and a separate `gate_reason` field.

Other `defer` reasons are not gate-related: `vlm_error`,
`unparseable_output`, `unknown_decision` (token outside the capture
vocabulary), and `unknown_place` (the model named a place other than the one
recognized as "here"). A validated capture can still be refused by the
executor with `cooldown`, `no_structure`, `invalid_structure`, or
`unknown_place`; none of those write.

### Executor: no model-authored geometry

`CaptureMapExecutor` fetches the live structure snapshot **itself** and calls
`store.upsert_place_map(place_id, snapshot)` — merge semantics, so a re-capture
adds evidence and never replaces or shrinks the stored map. Decision
`arguments` are ignored except `place_id`, which is forced to the recognized
place during sanitisation. `reject_capture`, `skip` and `defer` write nothing.
A per-place cooldown (`cooldown_s`, same `--router-cooldown` value) suppresses
a repeat capture; the outcome reports `written`, `created`, `revision` and the
stored cell/wall counts.

### Every tick, with a write cooldown

`RouterService` assesses `CaptureMapSkill` on **every** tick, alongside place
memory. The one-in-flight guard keeps the two calls sequential (never queued
or raced), so a tick now contains two brain calls; the measured cost is in the
latency section below. A per-place **180 s cooldown** in `CaptureMapExecutor`
means a repeated `capture_map` decision cannot re-write the map every tick:
the model is still asked, the decision is still logged, and the outcome shows
`reason_code: "cooldown"` with `written: false`. The gate state (including
`in_cooldown`) is visible in `/router/status`.

### Voice-turn guard (fail-open)

Background ticks are skipped entirely while a spoken turn is in flight. The
`bebop-voice` gateway on `:9093` currently exposes only build/readiness state
on `GET /healthz` — there is **no in-flight-turn signal**. `VoiceBusyProbe`
therefore:

- looks for an explicit signal when one exists (`turn_active`, `in_flight`,
  `busy`, `processing`, `speaking`, `utterance_active`, …);
- otherwise **fails open** (assumes no turn), logging that fact once;
- on any transport/protocol error also fails open, logs once, and returns
  `False`. It uses a short timeout and never raises, so a dead voice service
  can never crash or block the router thread.

Add the gateway's real turn counter to `BUSY_KEYS` when it grows one; no other
change is needed. `--router-voice-url` overrides the probe URL.

### Latency (measured on the Thor)

The brain (`RadixArk/Qwen3.8-27B-NVFP4`) is already resident at
`http://127.0.0.1:8012` as the voice cascade's middle stage:

| Prompt | Latency |
|---|---|
| text only | ~300 ms |
| one 640×400 image (273 prompt tokens) | ~834 ms |
| one 1280×800 image (509 prompt tokens) | ~1.0 s |

The answer was the same, so the capture prompt downscales images to ~640 px
before sending them. Calls are never queued behind a spoken turn, but with
both skills prompted **every tick** the tick now contains two brain calls.

Measured on the robot, a full place-memory decision is **6.0–9.5 s**
(`latency_ms` in the decision log); a capture decision is in the same
**5–10 s** class. That is the duty-cycle cost of asking every tick: at the
20 s interval the two calls consume most of the interval on the brain shared
with voice. The per-place 180 s capture cooldown keeps the *writes* bounded,
but not the prompts. `GET /router/status` reads a cached gate snapshot
instead of probing.

## Pending places and batch import

The default path is deliberately lofi:

```
PlaceScout (1 Hz, DINOv3, no LLM)
  → pending place (status=pending, keyframes + descriptors, invisible to nav)
  → operator selects pending places in the app
  → optional model suggestion per unnamed place
  → editable name → Confirm import
  → confirmed place (status=confirmed, recognized/navigable)
```

- The scout always runs whenever the places service is up. It matches a view
  against confirmed places first, then pending places, and only writes a
  pending row after ~3 sightings over ~2 s. Pending rows are capped (default
  50) and expire (default 14 days).
- Pending places are excluded from `/whereami`, `/places/graph`, navigation,
  and the default `/places` list. The app's Places screen shows them in a
  separate review queue.
- `POST /places/pending/{id}/suggest` runs the model over the pending
  place's stored keyframes (up to 3, 640 px) and stores
  `suggested_name` / `suggested_facts` on the row. It does **not** confirm the
  place. Suggestions are cached; `{"refresh": true}` regenerates one.
- `POST /places/pending/import` is deterministic and never calls the model. It
  promotes each pending row with the operator-edited name, records
  `name_source` (`llm` when unchanged from the suggestion, `operator` when
  edited or typed), and attaches suggested facts. A duplicate name or a
  missing name leaves the row pending.
- A close confirmed match is shown as a hint. Merging is an **explicit
  secondary action** (`merge_into` per place): it copies keyframes,
  descriptors and facts into the confirmed place and deletes the pending row.
  Importing as a new place is the default.
- Naming a place yourself skips the model entirely; import still works when
  no VLM is configured.

This is the battery fix: capture costs one DINOv3 embedding per second, and
the 27B only runs when the operator explicitly asks for a suggestion.

## Learning from the log

Every tick appends one record to a bounded in-memory log:

```json
{"ts": 1750000000.0, "source": "loop", "decision": "remember_place",
 "reason_code": "new_room", "latency_ms": 812, "applied": true,
 "place_id": "workshop-3f9a1c"}
```

That log — plus operator corrections in the app — is the seed of the training
set for the eventual **fast router**: the strong model is the teacher, the log
is the dataset, and the target is a small VLM that makes the same calls in
tens of milliseconds without reasoning.

## Known limits / next steps

- **Retrieve-then-reason only** — no metric localisation; remembering a place
  does not yet give a route back to it (see `bebop_vision/places/__init__.py`).
- **No persistence of the decision log yet** — it is in memory. Persisting it
  (SQLite next to the place DB) is the next step so training data survives a
  restart.
- **Two skills today** — place memory and map capture. The router is shaped to
  grow (`navigate_to`, `inspect_object`, `ask_user`) without changing the tick
  loop.
- **Every-tick prompting is expensive** — the legacy router asks both skills
  each tick; measured ~6–9.5 s for place memory and ~5–10 s for capture on the
  shared brain (~12 W average at a 45 s interval). The default lofi scout +
  operator import avoids this entirely; enable `--place-router` only for
  experiments.
