# System 1 place-memory router

A fast, bounded **visual decision layer** for the robot. It looks at a small
observation bundle — a couple of camera frames, a short task history, and the
places already in memory — and picks one next action.

It is **not** a planner and **not** a controller. It emits one structured
decision; an executor applies it. The first skill it owns is *place memory*:
**"should I remember this place?"**

```
 far/near JPEGs   task context   recent events   DINOv3 candidates
        └──────────────┬──────────────┴──────────────┘
                       ▼
              RouterService (one tick, one decision)
                       ▼
              PlaceMemorySkill  ──► VLM (Qwen3.8-27B NVFP4, OpenAI API)
                       ▼
        remember_place | update_place | skip | defer
                       ▼
              PlaceMemoryExecutor ──► Places SQLite store
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
| `bebop_vision/router/service.py` | `RouterService`: one in-flight tick, event history, decision log |
| `bebop_vision/places/server.py` | `/router/*` HTTP routes (proxied through the places service) |
| `tests/test_router.py` | 26 tests, no torch / camera / VLM server needed |

## Decisions

| Decision | Meaning |
|---|---|
| `remember_place` | Create a new place (`arguments.name`) |
| `update_place` | Add this view as evidence for an existing place (`arguments.place_id`) |
| `skip` | Already represented, or not a place |
| `defer` | Evidence weak/ambiguous — look again later |

Reply shape (the model emits exactly this JSON):

```json
{
  "decision": "remember_place",
  "reason_code": "new_room",
  "confidence": 0.8,
  "arguments": {"name": "Workshop"},
  "observations": ["a workbench under a wide window", "red storage cabinets"],
  "evidence": ["far"]
}
```

Aliases (`create_place`, `enrich`, `wait`, …) are normalised. Anything the
executor could not safely apply is turned into a `defer` with a reason:
`unknown_decision`, `missing_name`, `unknown_place`, `insufficient_evidence`,
`unparseable_output`, `vlm_error`.

## Run it

```bash
# on the Thor, from ~/bebop/bebop-vision
sudo .venv/bin/python -u main.py --record-navd /var/lib/bebop-captures --auto \
  --place-router --router-url http://127.0.0.1:8012
```

Flags: `--router-interval` (default 20 s), `--router-cooldown` (default 180 s,
a second *remember* inside the window folds into the place just created),
`--router-timeout` (default 120 s), `--router-model`.

The router is **off** unless `--place-router` is passed. It needs the places
service (do not combine with `--no-places`).

### Inspect it

```bash
curl -s localhost:9095/router/status            # running, counters, event list
curl -s 'localhost:9095/router/decisions?limit=10'
curl -s -X POST localhost:9095/router/assess \
     -H 'Content-Type: application/json' \
     -d '{"instruction": "remember this as the workshop"}'   # one decision now
```

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
- **Skill vocabulary is one skill** — the router is shaped to grow
  (`navigate_to`, `inspect_object`, `ask_user`) without changing the tick loop.
- **Latency budget unmeasured on-robot** — validate the 27B's image path and
  tick latency under real camera + navd load before trusting the cadence.
