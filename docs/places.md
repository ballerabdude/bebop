# Place memory (places service, `:9095`)

Appearance-based place recognition among rooms the operator has confirmed.
The service runs inside the `bebop-vision` process (`main.py --record-navd`,
which `bebop-vision.service` runs) and reads the Orbbec rig without opening a
camera. It is not a metric map: a recognised room has no pose, no global
frame and no route to it.

The every-tick place router (`--place-router`, `bebop_vision/router/`, its
`/router/*` routes on this port) and the per-place structure submaps that
live SLAM fed (`/places/{id}/map`, `places/submap.py`, `:9096`) were removed
on 2026-10-24; history:
`git log -- bebop-vision/bebop_vision/router bebop-vision/bebop_vision/places/submap.py bebop-vision/bebop_vision/slam`.
The router asked the 27B brain on every tick (measured ~12 W average at a
45 s interval, 2026-10-18); capture is now the lofi scout below.

```
far camera frame, every 1 s
  -> DINOv3 descriptor (places/encoder.py)
  -> PlaceScout (places/scout.py, no LLM)
       confident match to a confirmed place -> last_seen update; /place sighting in the open drive MCAP
       match to a pending place             -> more evidence for it
       novel, stable view                   -> new PENDING place
  -> operator review in the app (Memory > Places)
       optional name suggestion: one VLM call (bebop_vision/vlm.py)
       Import (no model call) | Merge into the matched place | Discard
  -> confirmed place: /whereami, the place graph, the voice place tools
```

## Files

| Path (`bebop-vision/`) | Role |
|---|---|
| `bebop_vision/places/server.py` | `PlacesService` (the operations) + the HTTP handler |
| `bebop_vision/places/store.py` | `PlaceStore`: SQLite + keyframe JPEGs, cosine matching, the pending queue, the place graph |
| `bebop_vision/places/scout.py` | `PlaceScout`, `ScoutConfig`: the pending-place capture loop |
| `bebop_vision/places/importer.py` | `PlaceImporter`: name suggestions and the deterministic import / merge |
| `bebop_vision/places/encoder.py` | `DinoV3Encoder`: JPEG -> L2-normalised 1024-d descriptor |
| `bebop_vision/places/capture.py` | `RigCapture`: the freshest color JPEG per camera role |
| `bebop_vision/vlm.py` | `ChatVLMClient`: OpenAI-compatible `/v1/chat/completions` client for suggestions |
| `bebop_vision/session_notes.py` | queues the scout's sightings for the recorder's `/place` channel |
| `tests/test_places_store.py`, `tests/test_place_scout.py`, `tests/test_place_import.py`, `tests/test_vlm.py` | store, scout, import and HTTP tests; no camera, GPU or VLM needed |

## Running it

`main.py --record-navd <dir>` starts the service after the robot runtime
answers and the cameras deliver frames, and stops it with the recorder. The
scout starts with the service.

| Flag | Default | Meaning |
|---|---|---|
| `--no-places` | off | do not start the service (or the scout) |
| `--places-port` | `9095` | HTTP port |
| `--places-root` | `/var/lib/bebop-places` | SQLite DB + keyframes |
| `--place-scout-interval` | `1.0` | seconds between scout scans |
| `--place-pending-max` | `50` | pending places kept for review |
| `--place-pending-ttl-days` | `14` | delete a pending place not seen for this long |
| `--router-url` | `http://127.0.0.1:8012` | OpenAI-compatible VLM for name suggestions: the voice cascade's brain stage |
| `--router-model` | `RadixArk/Qwen3.8-27B-NVFP4` | model name sent to it |
| `--router-timeout` | `120` | suggestion request timeout, seconds |

The `--router-*` names outlived the router; they configure only the
suggestion client. The default URL is served by `bebop-voice` while it runs
the cascade engine ([`voice.md`](voice.md)); with the brain down a
suggestion fails with `vlm_error` and the operator types the name.

Extra flags go in `/etc/bebop/vision.env` (`BEBOP_VISION_EXTRA_ARGS`).
`main.py` rejects unknown flags, so a leftover `--place-router`,
`--router-interval`, `--router-cooldown`, `--router-voice-url` or
`--live-slam*` keeps `bebop-vision.service` in a restart loop: remove them
before deploying.

## Encoder

`DinoV3Encoder` (`places/encoder.py`):

- Model: DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`). The
  architecture config is built in code (`DINOV3_CONFIG`), so loading needs
  no gated `config.json` fetch and no HF token.
- Weights: `bebop-vision/weights/dinov3-vitl16/`, else `bebop-vision/weights/`
  (`model.safetensors` or `pytorch_model.bin`). The catalog entry
  `dinov3-vitl16` (gated; Settings › Models) downloads `model.safetensors`
  into `weights/` ([`models.md`](models.md)).
- Preprocessing (numpy/cv2): resize to 224×224, BGR -> RGB, ImageNet
  mean/std. The descriptor is the CLS token of the last hidden state,
  L2-normalised, 1024-d float32.
- Loads lazily on first use (~1.2 GB); `main.py` warms it on a background
  thread, then freezes the heap (`gc_tuning.freeze_heap`, off with
  `--no-gc-freeze`), since the ViT-L is ~280 k tracked objects.
- CUDA when available, else CPU; fp32. One model behind one lock, shared by
  the scout, `/whereami`, teaching and view bursts.
- ~46 ms per embedding on the Thor (fp32, CUDA; measured 2026-10-16,
  `places/encoder.py`).

`RigCapture` returns, per camera role, the freshest frame that carries a
color JPEG and is at most 0.5 s old (the SDK also publishes depth-only frame
sets, so it scans the camera's recent history). Recognition (scout and
`/whereami`) encodes the far camera's view, falling back to the first view;
the near camera looks down at the floor. Teaching stores every camera's view.

## Scout: pending places (no LLM)

Every `--place-scout-interval` seconds `PlaceScout` encodes the far view and
classifies it (thresholds from `ScoutConfig`; cosine scores, best descriptor
per place):

| Case | Test | Action |
|---|---|---|
| known room | best confirmed score ≥ 0.80 and ≥ 0.15 above the runner-up | `last_seen_at` / `seen_count` updated (at most every 5 s per place); a `/place` sighting for the open drive segment |
| pending room | best pending score ≥ 0.82 | `seen_count` + 1; the view becomes a keyframe + descriptor when the place has < 8 and it is not a near-duplicate (cosine < 0.97) |
| novel view | otherwise | joins the in-memory candidate it matches (cosine ≥ 0.80 to its centroid) or starts one |

- A candidate becomes a pending place after ≥ 3 sightings spanning ≥ 2 s,
  at most one new pending place per 5 s. It keeps up to 3 mutually distinct
  frames (cosine < 0.97), which become the pending place's keyframes and
  descriptors (`dinov3:scout`). Candidates expire 6 s after their last
  sighting; at most 8 are held.
- The best confirmed match scoring ≥ 0.50 is stored on the pending place as
  the merge hint (`best_match_id`, `best_match_score`).
- Queue: at most `--place-pending-max` (50) pending places. When the queue
  is full the scout deletes expired rows and otherwise writes nothing new;
  it never evicts a fresh pending place to make room, so review the queue.
  Pending places not seen for `--place-pending-ttl-days` (14) are deleted on
  the first scan, hourly after, and whenever the queue is full. Each prune
  also trims a queue that is over the cap, oldest evidence first, which only
  happens after lowering `--place-pending-max`. Confirmed places are never
  pruned.
- A pending place has a placeholder name (`__pending__<hex>`;
  `display_name` shows the suggested name or "Unnamed place") and is
  excluded from `/whereami`, `GET /places`, `/places/graph` and the scout's
  known-room test until it is imported.
- Sightings reach the drive MCAP only while a recorder segment is open; the
  scout itself runs whenever the places service does, recording or not.

Cost: one DINOv3 embedding per second. The 27B runs only when the operator
asks for a suggestion.

## Review and import

The app's Memory › Places tab (`#/r/:host/autonomy/memory`, badge = pending
count) shows the queue newest evidence first, 12 cards per page: keyframes,
an editable name (prefilled with the suggestion), Import, Discard, and Merge
into the matched place when its score is ≥ 0.80 (the app's `MERGE_SCORE`).
Selecting cards opens a bar with Suggest names (for selected cards without a
name), Import N (once every selected card has a name) and Discard. Discard
is `DELETE /places/{id}`.

`POST /places/pending/{id}/suggest` `{refresh?}` (`PlaceImporter.suggest`):

- refreshes the merge hint, then returns a cached suggestion younger than
  1 h unless `refresh: true`;
- otherwise makes one VLM call: up to 3 keyframes (the far camera's when
  there are any, evenly spaced), each downscaled to ≤ 640 px (JPEG q80),
  `max_tokens` 160, temperature 0; the prompt asks for one JSON object
  `{"name", "facts" (3-5), "confidence"}`;
- stores `suggested_name` / `suggested_facts`; the place stays pending;
- latency, measured on the Thor's brain for the removed router (2026-10-18):
  text only ~300 ms, one 640×400 image ~834 ms, one 1280×800 image ~1.0 s
  with the same answer, hence the 640 px downscale;
- replies `{id, suggested_name, facts, confidence, cached, latency_ms,
  conflict, best_match}` (`conflict`: a place already has that name).
  Other failures are HTTP 200 with `error`: `vlm_error` (the request failed:
  backend down, an HTTP error, a timeout or a malformed completion),
  `unparseable_output`, `empty_name`, `invalid_name`, `no_images`, or
  `vlm_not_configured` (no client; `main.py` always configures one). An
  unknown id is 404, a confirmed place 400.

`POST /places/pending/import` `{ids, names?, merge_into?, facts?}`
(≤ 100 ids; the maps are keyed by pending id) never calls the model:

- `merge_into[id]` set: copy every descriptor (`dinov3:pending-merge`),
  keyframes until the target holds 32, and the pending place's own facts
  (source `pending-merge`; `facts[id]` and the suggested facts are not used)
  into that confirmed place, mark the target seen, then delete the pending
  row -> `merged`.
- Otherwise promote with `names[id]`, else the suggested name -> `imported`.
  `name_source` is `llm` when the final name equals the suggestion,
  `operator` otherwise. Facts: `facts[id]`, else the suggested facts, added
  with source `import`.
- A failure leaves the row pending: `not_found`, `already_confirmed`,
  `merge_failed`, `needs_name`, `name_conflict`, `invalid_name`.
- Reply: `{results: [...], imported, merged, failed}`.

## Recognition: `/whereami`

`GET /whereami?k=3` encodes the far view and ranks confirmed places by best
descriptor score. `matched` is true when the top score is ≥ `min_score`
(0.5) and ≥ `min_margin` (0.05) above the runner-up; a match updates that
place's `last_seen_at`. Reply: `matched, place, score, margin, min_score,
min_margin, view, top[{id, name, score}]`, or `{matched: false, reason}`
with `encoder not configured`, `no fresh camera frames`, `encode failed: ...`
or `no places taught`. These thresholds are separate from the scout's
known-room test (0.80 / 0.15), and neither set is a flag.

Each call is one encode. The app polls it every 3 s while a places consumer
(Memory, the Autonomy console's status panel) is mounted and the tab is
visible; the voice `where_am_i` tool calls it per request.

## Teaching and editing

The app's Memory › Places tab also lists the known places (tiles, then a
detail with views, facts, "Add current view", "Capture 10 s while turning"
and Forget) and a Teach card for the current view.

- `POST /places` `{name, facts?}` (app: Teach; voice: `remember_place`)
  creates a confirmed place from the current view of every camera: one
  keyframe and one descriptor (`dinov3:<role>`) each; facts get source
  `operator`. 201 `{place, views, descriptors}`. A place without a
  descriptor is listed but never recognised.
- `POST /places/{id}/views` `{seconds?, interval_s?}` (`interval_s`
  default 1.5 s, at least 0.2 s): a burst that captures until `seconds`
  elapse or 24 keyframes were added, skipping frames whose camera stamp did
  not change; `seconds` 0 (the default, the app's "Add current view") grabs
  once. 201 `{place, captured, keyframes, descriptors, views_total}`.
- `POST /places/{id}/facts` `{text, source?}` (default `operator`),
  `POST /places/{id}/rename` `{name}`, `DELETE /places/{id}` (descriptors,
  facts, keyframes, incident edges and the JPEGs go with it).
- Names are whitespace-collapsed, 1-64 characters and unique ignoring case.
  Ids are `<slug>-<6 hex>` and survive renames; every `{id}` path segment
  also accepts the place's name.

## Place graph

`place_edges` holds route knowledge ("got from A to B"), not a metric
transform. One directed edge per pair; A->B and B->A are separate.

| Column | Meaning |
|---|---|
| `from_place`, `to_place` | composite primary key; deleting either place deletes the edge |
| `traversals` | attempts recorded |
| `successes` | attempts with confirmed arrival |
| `last_attempt_at`, `last_success_at` | epoch seconds |
| `median_duration_s` | median of the last ≤ 100 successful durations that carried one |
| `guide_path_json` | `[[x, y], ...]` from the latest successful traversal that sent one: finite numbers, ≤ 10 000 points; nothing interprets them |
| `durations_json` | the durations behind the median |
| `created_at`, `updated_at` | epoch seconds |

- `POST /places/{from}/edges/{to}` `{success (bool, required), duration_s?,
  guide_path?}` -> 201 for a new edge, 200 after; `{edge, created}`. A
  failure counts as an attempt only; `success_rate` is computed on read.
- `GET /places/graph` -> `{nodes: [{id, name, created_at, last_seen_at}],
  edges: [...], node_count, edge_count}`; nodes are confirmed places.
- Nothing writes edges today. The writers, the route follower and route
  teach mode, were removed with the navigation stack on 2026-10-22
  (history: `git log -- bebop-vision/bebop_vision/nav`). The route and the
  store are covered only by unit tests (`tests/test_places_store.py`); edges
  recorded before then stay in the database. The app's Memory › Place graph
  tab (`?view=map`) draws the graph read-only, refreshed every 15 s.

## Storage

`--places-root` (default `/var/lib/bebop-places`, created on start;
`bebop-vision` runs as root):

```
/var/lib/bebop-places/
  places.db                                  SQLite (stdlib sqlite3, foreign keys on)
  keyframes/<place_id>/<stamp_us>_<role>.jpg
```

- Tables: `places` (`status` = `pending` | `confirmed`), `descriptors`
  (float32 blob, `dim`, `source`), `facts`, `keyframes`, `place_edges`.
- Databases created before pending places get the new `places` columns on
  open; existing rows stay confirmed.
- `odom_x`, `odom_y`, `odom_theta` stay in the schema; nothing reads or
  writes them, and the place JSON has no `odom`. Databases that predate
  2026-10-24 keep the unused `tags` and `place_maps` tables (no DROP); place
  tags and submaps were removed that day.
- Fact sources: `operator` (teach, `POST /facts`), `import` (confirmed on
  import), `pending-merge` (copied by a merge); rows written before
  2026-10-24 may say `router`.
- Descriptor sources: `dinov3:<role>` (teach, views), `dinov3:scout`,
  `dinov3:pending-merge`.

## HTTP API

JSON in and out, `Access-Control-Allow-Origin: *`, `OPTIONS` answers
preflights (GET, POST, DELETE). Errors are `{"error": "..."}`: 404 for an
unknown place or route, 400 for a bad body or value.

| Method | Path | Body / query | Reply |
|---|---|---|---|
| GET | `/healthz` | | `ok, root, places, pending, descriptors, encoder{configured, loaded, device, dim}, capture, min_score, min_margin, scout{running, interval_s, ticks, known, pending_hits, pending_created, pending_count, candidates, last_action, last_tick_ms, errors, last_error, thresholds}, importer{configured, vlm}` |
| GET | `/places` | | `{places: [confirmed], pending_count}` |
| POST | `/places` | `{name, facts?}` | 201 `{place, views, descriptors}` |
| GET | `/places/pending` | | `{places: [pending, each with best_match], count}`; up to 100, newest evidence first |
| POST | `/places/pending/{id}/suggest` | `{refresh?}` | the suggestion (above) |
| POST | `/places/pending/import` | `{ids, names?, merge_into?, facts?}` | `{results, imported, merged, failed}` |
| GET | `/places/graph` | | `{nodes, edges, node_count, edge_count}` |
| GET | `/whereami` | `?k=3` | the recognition (above) |
| GET | `/places/{id}` | | `{place}`, any status |
| DELETE | `/places/{id}` | | `{ok: true}` |
| POST | `/places/{id}/facts` | `{text, source?}` | 201 `{ok, fact}` |
| POST | `/places/{id}/rename` | `{name}` | `{place}` |
| POST | `/places/{id}/views` | `{seconds?, interval_s?}` | 201 burst result |
| GET | `/places/{id}/keyframes/{file}` | | `image/jpeg` (files of that place only) |
| POST | `/places/{from}/edges/{to}` | `{success, duration_s?, guide_path?}` | 201 / 200 `{edge, created}` |

Place JSON (`Place.to_dict`):

| Field | Meaning |
|---|---|
| `id`, `name`, `display_name` | `display_name` hides a pending placeholder name |
| `status` | `pending` or `confirmed` |
| `source` | `operator` (taught) or `scout` (captured) |
| `seen_count` | sightings: scout, `/whereami` matches, view bursts, merges |
| `best_match_id`, `best_match_score` | the merge hint of a pending place |
| `suggested_name`, `suggested_facts`, `suggested_at` | the last name suggestion |
| `name_source` | `operator`, `llm`; null while pending and on places taught before pending places existed |
| `confirmed_at`, `created_at`, `updated_at`, `last_seen_at` | epoch seconds |
| `facts` | `[{text, source, created_at}]` |
| `keyframes` | `[{path, role, stamp_us}]`, paths relative to the places root |

## Clients

| Client | Uses |
|---|---|
| App: `bebop-app/src/robot/placesStore.ts`, `bebop-app/src/screens/memory/` | polls `/places` (15 s), `/places/pending` (Memory only), `/whereami` (3 s), `/places/graph` (15 s, Place graph tab), `/healthz` (10 s, the status-bar dot); writes teach, views, facts, delete, suggest, import |
| Voice brain (`bebop_vision/voice_server.py`; cascade engine with live context and tools on) | `where_am_i` (`/whereami`), `list_places` (`/places`), `place_facts` (`/places/{name}`), `remember_place` (`POST /places`), `add_place_fact`, `rename_place`, `forget_place` |
| Recorder (`bebop_vision/recorder_mcap.py`) | writes the scout's known-room sightings as `/place` (`bebop.navd.PlaceSeen`: `place_id, name, score, margin, stamp_ns`); `tools/session_ledger.py` reads them |

## Limits

- Recognise only: no metric pose for a place and no route to it; the graph
  has no writer.
- Recognition compares one far-camera view against best-per-place cosine
  scores with fixed thresholds; descriptors of another dimension are
  skipped.
