"""Places HTTP service (default :9095).

Runs inside the `bebop-vision` process, sharing the Orbbec rig with the
recorder and the operator video server. It exposes the semantic place
memory over a small JSON API:

    GET    /healthz                      service + encoder status
    GET    /places                       list taught places
    POST   /places        {name, ...}    teach the current view as a place
    GET    /places/{id}                  one place (facts, tags, keyframes)
    DELETE /places/{id}                  forget a place
    POST   /places/{id}/facts {text}     operator-stated fact
    POST   /places/{id}/rename {name}    rename
    GET    /places/{id}/keyframes/{file} the keyframe JPEG
    GET    /whereami                     recognise the current room

    POST   /places/{id}/map              accumulate a :9096 /structure snapshot
                                         ({"replace": true} starts over)
    GET    /places/{id}/map              the place's local submap
    GET    /places/graph                 nodes (places) + route edges
    POST   /places/{from}/edges/{to}     record one A->B traversal

    POST   /routes/teach/start {from?, label?}   begin recording a trail
    POST   /routes/teach/stop  {to?, facts?}     finish: store route + edge
    POST   /routes/teach/cancel
    GET    /routes/teach                 teach status
    GET    /routes[?from=&to=]           taught routes
    GET    /routes/{id}                  one route with nodes
    GET    /routes/{id}/nodes/{seq}/keyframe?role=far
    DELETE /routes/{id}

When a System 1 router is attached (`router=`), it also proxies that:

    GET    /router/status                router state + rolling event list
    GET    /router/decisions?limit=N     recent routing decisions
    POST   /router/assess {instruction, skill}  one assessment now (blocking)

The same operations are public methods, so the LLM tool layer can call them
directly without a second HTTP hop.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .store import Place, PlaceStore

DEFAULT_PORT = 9095
DEFAULT_ROOT = Path("/var/lib/bebop-places")


def _log(msg: str) -> None:
    print(f"[places] {msg}", flush=True)


class PlacesService:
    """The memory operations, independent of the HTTP transport."""

    def __init__(
        self,
        store: PlaceStore,
        *,
        capture=None,
        encoder=None,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        min_score: float = 0.5,
        min_margin: float = 0.05,
        capture_roles=("near", "far"),
        router=None,
        scout=None,
        importer=None,
    ):
        self.store = store
        self.capture = capture
        self.encoder = encoder
        self.router = router
        self.scout = scout
        self.importer = importer
        # Route teach mode (set after construction; needs a pose source).
        self.teacher = None
        self.host = host
        self.port = int(port)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.capture_roles = tuple(capture_roles)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.bound_port: int | None = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        service = self

        class Handler(_PlacesHandler):
            svc = service

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.bound_port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="places-http", daemon=True
        )
        self._thread.start()
        if self.scout is not None:
            self.scout.start()
        _log(f"listening on :{self.bound_port} (root={self.store.root})")

    def stop(self) -> None:
        if self.teacher is not None:
            self.teacher.stop_thread()
        if self.scout is not None:
            self.scout.stop()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # -- memory operations ------------------------------------------------

    def list_places(self, *, status: str = "confirmed") -> list[Place]:
        return self.store.list_places(status=status)

    def list_pending(self, *, limit: int = 100) -> list[dict]:
        out: list[dict] = []
        for place in self.store.list_pending(limit=limit):
            item = place.to_dict(root=self.store.root)
            if place.best_match_id:
                target = self.store.get_place(place.best_match_id)
                item["best_match"] = (
                    {"id": target.id, "name": target.name,
                     "score": (round(float(place.best_match_score), 4)
                               if place.best_match_score is not None else None)}
                    if target is not None else None
                )
            else:
                item["best_match"] = None
            out.append(item)
        return out

    def suggest_pending(self, id_or_name: str, *, refresh: bool = False) -> dict:
        if self.importer is None:
            raise ValueError("name suggestions are not configured")
        return self.importer.suggest(id_or_name, refresh=refresh)

    def import_pending(
        self,
        ids,
        *,
        names=None,
        merge_into=None,
        facts=None,
    ) -> dict:
        if self.importer is None:
            raise ValueError("place import is not configured")
        return self.importer.import_pending(
            ids, names=names, merge_into=merge_into, facts=facts)

    def get_place(self, id_or_name: str) -> Place | None:
        return self.store.get_place(id_or_name)

    def teach(
        self,
        name: str,
        *,
        facts: list[str] | None = None,
        odom: tuple[float, float, float] | None = None,
        roles=None,
    ) -> dict:
        """Create a place from the current camera view(s).

        Every fresh view contributes a keyframe and (when an encoder is
        configured) a DINOv3 descriptor. A place with no descriptor can be
        listed and described but not recognised.
        """
        place = self.store.create_place(name, odom=odom)
        views = self._grab(roles)
        encoded = 0
        for v in views:
            self.store.add_keyframe(place.id, v.jpeg, role=v.role, stamp_us=v.stamp_us)
            if self.encoder is not None:
                try:
                    self.store.add_descriptor(
                        place.id, self.encoder.encode_jpeg(v.jpeg),
                        source=f"dinov3:{v.role}",
                    )
                    encoded += 1
                except Exception as exc:  # noqa: BLE001 - keep teaching
                    _log(f"encode failed for {v.role}: {exc}")
        for text in facts or []:
            self.store.add_fact(place.id, text, source="operator")
        result = self.store.get_place(place.id)
        _log(f"taught {result.name!r} ({len(views)} views, {encoded} descriptors)")
        return {
            "place": result.to_dict(root=self.store.root),
            "views": len(views),
            "descriptors": encoded,
        }

    def capture_views(
        self,
        id_or_name: str,
        *,
        seconds: float = 0.0,
        interval_s: float = 1.5,
        roles=None,
        max_views: int = 24,
    ) -> dict:
        """Append the current view(s) to an existing place.

        With `seconds > 0` this is a burst: it keeps capturing every
        `interval_s` until the window elapses, so the operator can pan or
        rotate the robot to cover the room from several angles. Frames whose
        camera stamp is unchanged since the last capture are skipped, so
        standing still does not pile up duplicate descriptors.
        """
        place = self.store.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        deadline = time.monotonic() + max(0.0, float(seconds))
        interval = max(0.2, float(interval_s))
        added_kf = added_desc = ticks = 0
        last_stamp: dict[str, int] = {}
        while True:
            fresh = [v for v in self._grab(roles)
                     if last_stamp.get(v.role) != v.stamp_us]
            for v in fresh:
                last_stamp[v.role] = v.stamp_us
                self.store.add_keyframe(place.id, v.jpeg, role=v.role,
                                        stamp_us=v.stamp_us)
                added_kf += 1
                if self.encoder is not None:
                    try:
                        self.store.add_descriptor(
                            place.id, self.encoder.encode_jpeg(v.jpeg),
                            source=f"dinov3:{v.role}",
                        )
                        added_desc += 1
                    except Exception as exc:  # noqa: BLE001 - keep capturing
                        _log(f"encode failed for {v.role}: {exc}")
            ticks += 1
            if time.monotonic() >= deadline or added_kf >= max_views:
                break
            time.sleep(interval)
        if added_kf:
            self.store.touch_seen(place.id)
        result = self.store.get_place(place.id)
        _log(f"captured {added_kf} view(s) into {result.name!r} "
             f"({added_desc} descriptors, {ticks} tick(s))")
        return {
            "place": result.to_dict(root=self.store.root),
            "captured": ticks,
            "keyframes": added_kf,
            "descriptors": added_desc,
            "views_total": len(result.keyframes),
        }

    # -- local maps + place graph ----------------------------------------

    def ingest_map(
        self, id_or_name: str, payload: dict, *, pose=None,
        replace: bool = False,
    ) -> dict:
        """Accumulate one place's submap from a structure snapshot.

        By default the capture is merged into the stored evidence (anchor
        pinned at the first capture, counts never shrink).  ``replace=True``
        deliberately resets the place to exactly this patch.
        """
        place_map, created = self.store.upsert_place_map(
            id_or_name, payload, pose=pose, replace=replace)
        action = "replaced" if replace else ("stored" if created else "merged")
        _log(f"{action} submap for {id_or_name!r} "
             f"(revision {place_map.revision}, "
             f"{place_map.width}x{place_map.height} cells, "
             f"{len(place_map.walls)} walls)")
        return {"map": place_map.to_dict(), "created": created,
                "replaced": bool(replace)}

    def get_map(self, id_or_name: str):
        """The stored submap, or None when the place has never been mapped."""
        place = self.store.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        return self.store.get_place_map(place.id)

    def graph(self) -> dict:
        return self.store.graph()

    def record_traversal(
        self,
        from_place: str,
        to_place: str,
        *,
        success: bool,
        duration_s=None,
        guide_path=None,
    ) -> dict:
        edge, created = self.store.record_traversal(
            from_place, to_place, success=success,
            duration_s=duration_s, guide_path=guide_path)
        _log(f"traversal {edge.from_name!r} -> {edge.to_name!r}: "
             f"{'success' if success else 'failure'} "
             f"({edge.successes}/{edge.traversals})")
        return {"edge": edge.to_dict(), "created": created}

    def whereami(self, *, k: int = 3, roles=None) -> dict:
        """Recognise the current room by appearance.

        Returns the best place only when it clears both an absolute score
        and a margin over the runner-up; otherwise `matched` is False and the
        raw ranking is still included so thresholds can be tuned in the
        field.
        """
        if self.encoder is None:
            return {"matched": False, "reason": "encoder not configured"}
        views = self._grab(roles)
        if not views:
            return {"matched": False, "reason": "no fresh camera frames"}
        # The far (forward-looking) camera sees the room; the near camera is
        # angled down at the floor. Prefer far for the recognition query.
        view = next((v for v in views if v.role == "far"), views[0])
        try:
            vec = self.encoder.encode_jpeg(view.jpeg)
        except Exception as exc:  # noqa: BLE001
            return {"matched": False, "reason": f"encode failed: {exc}"}
        matches = self.store.match(vec, k=k, status="confirmed")
        if not matches:
            return {"matched": False, "reason": "no places taught", "top": []}
        top = matches[0]
        runner = matches[1].score if len(matches) > 1 else 0.0
        margin = top.score - runner
        matched = top.score >= self.min_score and margin >= self.min_margin
        if matched:
            self.store.touch_seen(top.place.id)
        return {
            "matched": matched,
            "place": top.place.to_dict(root=self.store.root) if matched else None,
            "score": round(top.score, 4),
            "margin": round(margin, 4),
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "view": view.role,
            "top": [
                {"id": m.place.id, "name": m.place.name, "score": round(m.score, 4)}
                for m in matches
            ],
        }

    def _grab(self, roles=None):
        if self.capture is None:
            return []
        if roles is None:
            return self.capture.grab()
        # RigCapture honours an explicit role subset via a temporary wrapper.
        try:
            from .capture import RigCapture
            if isinstance(self.capture, RigCapture):
                return RigCapture(self.capture.rig, roles=roles,
                                  max_age_s=self.capture.max_age_s).grab()
        except Exception:  # noqa: BLE001
            pass
        return self.capture.grab()

    def status(self) -> dict:
        return {
            "ok": True,
            "root": str(self.store.root),
            "places": self.store.count(status="confirmed"),
            "pending": self.store.count(status="pending"),
            "descriptors": self.store.descriptor_count(),
            "encoder": {
                "configured": self.encoder is not None,
                "loaded": bool(getattr(self.encoder, "loaded", False)),
                "device": getattr(self.encoder, "device", None),
                "dim": getattr(self.encoder, "dim", None),
            },
            "capture": self.capture is not None,
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "scout": (self.scout.status() if self.scout is not None
                      else {"enabled": False}),
            "importer": {
                "configured": self.importer is not None,
                "vlm": (self.importer.describe
                        if self.importer is not None else None),
            },
            "router": self.router.status() if self.router is not None
            else {"enabled": False},
        }


class _PlacesHandler(BaseHTTPRequestHandler):
    svc: PlacesService

    def log_message(self, fmt, *args):  # quieter than the default
        pass

    # -- helpers ----------------------------------------------------------

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _path(self):
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
        query = parse_qs(parsed.query)
        return parts, query

    def _teacher(self):
        teacher = self.svc.teacher
        if teacher is None:
            self._err(503, "route teach mode not configured (needs --nav "
                           "for the gyro pose)")
            return None
        return teacher

    def _router(self):
        """The attached System 1 router, or None after sending a 503."""
        router = self.svc.router
        if router is None:
            self._err(503, "router not configured")
            return None
        return router

    # -- routes -----------------------------------------------------------

    def do_GET(self):
        self.close_connection = True
        parts, query = self._path()
        svc = self.svc
        try:
            if parts == ["healthz"]:
                return self._json(svc.status())
            if parts == ["routes", "teach"]:
                teacher = self._teacher()
                if teacher is None:
                    return
                return self._json(teacher.status())
            if parts == ["routes"]:
                routes = svc.store.list_routes(
                    from_place=(query.get("from") or [None])[0],
                    to_place=(query.get("to") or [None])[0])
                return self._json({"routes": [
                    r.to_dict(root=svc.store.root, nodes=False)
                    for r in routes]})
            if len(parts) == 2 and parts[0] == "routes":
                route = svc.store.get_route(parts[1])
                if route is None:
                    return self._err(404, "no such route")
                return self._json({"route": route.to_dict(root=svc.store.root)})
            if (len(parts) == 5 and parts[0] == "routes"
                    and parts[2] == "nodes" and parts[4] == "keyframe"):
                role = (query.get("role") or ["far"])[0]
                try:
                    seq = int(parts[3])
                except ValueError:
                    return self._err(400, "bad node seq")
                path = svc.store.route_keyframe_path(parts[1], seq, role)
                if path is None or not path.is_file():
                    return self._err(404, "no such keyframe")
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
                return
            if parts == ["places"]:
                return self._json({
                    "places": [p.to_dict(root=svc.store.root)
                               for p in svc.list_places()],
                    "pending_count": svc.store.count(status="pending"),
                })
            if parts == ["places", "pending"]:
                return self._json({
                    "places": svc.list_pending(),
                    "count": svc.store.count(status="pending"),
                })
            if parts == ["places", "graph"]:
                return self._json(svc.graph())
            if parts == ["whereami"]:
                k = int((query.get("k") or ["3"])[0])
                return self._json(svc.whereami(k=k))
            if parts == ["router", "status"]:
                router = self._router()
                if router is None:
                    return
                return self._json(router.status())
            if parts == ["router", "decisions"]:
                router = self._router()
                if router is None:
                    return
                limit = int((query.get("limit") or ["20"])[0])
                return self._json({
                    "decisions": router.decisions(limit=limit),
                    "events": router.events(),
                })
            if len(parts) == 2 and parts[0] == "places":
                place = svc.get_place(parts[1])
                if place is None:
                    return self._err(404, "no such place")
                return self._json({"place": place.to_dict(root=svc.store.root)})
            if (len(parts) == 3 and parts[0] == "places"
                    and parts[2] == "map"):
                place_map = svc.get_map(parts[1])
                if place_map is None:
                    return self._err(404, "no map captured for place")
                return self._json({"map": place_map.to_dict()})
            if (len(parts) == 4 and parts[0] == "places"
                    and parts[2] == "keyframes"):
                return self._serve_keyframe(parts[1], parts[3])
            return self._err(404, "not found")
        except KeyError:
            return self._err(404, "no such place")
        except ValueError as exc:
            return self._err(400, str(exc))

    def do_POST(self):
        self.close_connection = True
        parts, query = self._path()
        svc = self.svc
        try:
            body = self._body()
            if len(parts) == 3 and parts[:2] == ["routes", "teach"]:
                teacher = self._teacher()
                if teacher is None:
                    return
                action = parts[2]
                try:
                    if action == "start":
                        return self._json(teacher.start(
                            body.get("from") or body.get("from_place"),
                            label=str(body.get("label") or "")), status=201)
                    if action == "stop":
                        return self._json(teacher.stop(
                            body.get("to") or body.get("to_place"),
                            facts=body.get("facts")), status=201)
                    if action == "cancel":
                        return self._json(teacher.cancel())
                except KeyError as exc:
                    return self._err(404, f"no such place: {exc.args[0]}")
                except RuntimeError as exc:
                    return self._err(409, str(exc))
                return self._err(404, "not found")
            if parts == ["router", "assess"]:
                router = self._router()
                if router is None:
                    return
                skill = body.get("skill")
                if skill is None:
                    skill = (query.get("skill") or [None])[0]
                if skill is not None:
                    skill = str(skill).strip()
                    names = router.skill_names()
                    if skill not in names:
                        return self._err(
                            400,
                            f"unknown skill {skill!r}; available: "
                            f"{', '.join(names)}")
                instruction = body.get("instruction") or None
                return self._json(router.assess_once(
                    instruction=instruction, source="http", skill=skill))
            if parts == ["places", "pending", "import"]:
                ids = body.get("ids")
                if not isinstance(ids, list) or not ids:
                    return self._err(400, "ids must be a non-empty list")
                names = body.get("names") or {}
                merge_into = body.get("merge_into") or {}
                facts = body.get("facts") or {}
                if not isinstance(names, dict):
                    return self._err(400, "names must be an object")
                if not isinstance(merge_into, dict):
                    return self._err(400, "merge_into must be an object")
                if not isinstance(facts, dict):
                    return self._err(400, "facts must be an object")
                result = svc.import_pending(
                    ids, names=names, merge_into=merge_into, facts=facts)
                return self._json(result)
            if (len(parts) == 4 and parts[0] == "places"
                    and parts[1] == "pending" and parts[3] == "suggest"):
                refresh = bool(body.get("refresh", False))
                return self._json(
                    svc.suggest_pending(parts[2], refresh=refresh))
            if parts == ["places"]:
                name = body.get("name")
                if not name:
                    return self._err(400, "name is required")
                facts = body.get("facts") or []
                if isinstance(facts, str):
                    facts = [facts]
                result = svc.teach(str(name), facts=list(facts))
                return self._json(result, status=201)
            if (len(parts) == 3 and parts[0] == "places"
                    and parts[2] == "views"):
                seconds = float(body.get("seconds", 0) or 0)
                interval = float(body.get("interval_s", 1.5) or 1.5)
                result = svc.capture_views(
                    parts[1], seconds=seconds, interval_s=interval
                )
                return self._json(result, status=201)
            if (len(parts) == 3 and parts[0] == "places"
                    and parts[2] == "facts"):
                text = body.get("text")
                if not text:
                    return self._err(400, "text is required")
                fact = svc.store.add_fact(
                    parts[1], str(text), source=str(body.get("source", "operator"))
                )
                return self._json(
                    {"ok": True, "fact": {"text": fact.text,
                                          "source": fact.source}},
                    status=201,
                )
            if (len(parts) == 3 and parts[0] == "places"
                    and parts[2] == "rename"):
                name = body.get("name")
                if not name:
                    return self._err(400, "name is required")
                place = svc.store.rename_place(parts[1], str(name))
                return self._json({"place": place.to_dict(root=svc.store.root)})
            if (len(parts) == 3 and parts[0] == "places"
                    and parts[2] == "map"):
                replace = body.get("replace", False)
                if not isinstance(replace, bool):
                    raise ValueError("replace must be a boolean")
                result = svc.ingest_map(parts[1], body, replace=replace)
                return self._json(result,
                                  status=201 if result["created"] else 200)
            if (len(parts) == 4 and parts[0] == "places"
                    and parts[2] == "edges"):
                if "success" not in body:
                    return self._err(400, "success is required")
                result = svc.record_traversal(
                    parts[1], parts[3],
                    success=body["success"],
                    duration_s=body.get("duration_s"),
                    guide_path=body.get("guide_path"),
                )
                return self._json(result,
                                  status=201 if result["created"] else 200)
            return self._err(404, "not found")
        except KeyError:
            return self._err(404, "no such place")
        except ValueError as exc:
            return self._err(400, str(exc))

    def do_DELETE(self):
        self.close_connection = True
        parts, _ = self._path()
        svc = self.svc
        if len(parts) == 2 and parts[0] == "routes":
            if not svc.store.delete_route(parts[1]):
                return self._err(404, "no such route")
            return self._json({"ok": True})
        if len(parts) == 2 and parts[0] == "places":
            ok = svc.store.delete_place(parts[1])
            if not ok:
                return self._err(404, "no such place")
            return self._json({"ok": True})
        return self._err(404, "not found")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_keyframe(self, id_or_name: str, filename: str) -> None:
        place = self.svc.get_place(id_or_name)
        if place is None:
            return self._err(404, "no such place")
        # Only serve files that belong to this place's keyframe dir.
        safe = Path(filename).name
        base = (self.svc.store.keyframe_dir / place.id).resolve()
        target = (base / safe).resolve()
        if base not in target.parents or not target.is_file():
            return self._err(404, "no such keyframe")
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
