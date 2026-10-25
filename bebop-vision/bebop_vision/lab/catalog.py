"""Sessions on disk, on the robot and in the ledger.

The lab reads the recorder's raw MCAPs wherever they already are: the
mirror ``datasets/sessions`` that ``tools/pull_captures.py`` fills, and the
six 2026-10-18 drives that still live in ``datasets/bev-student/raw/mcap``
on the workstation. :class:`Catalog` lists both, joins the session ledger
(``tools/session_ledger.py``, ``sessions.jsonl``) when one exists, asks the
firmware's ``/captures`` route what the robot holds, pulls by name or glob
through the same code the mirror script uses, and turns a name, stem, glob
or path into the one file it means. Nothing here decodes images.

polars is the dataframe library (the venv has no pandas / pyarrow). The
robot-facing tools are not a package (``tools/`` has no ``__init__``), so
they are loaded by path on first use.
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.util
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import polars as pl

from bebop_vision.lab.types import file_identity
from bebop_vision.mcap_tools import read_info

#: ``bebop-vision/``: relative roots and the default cache resolve against it,
#: wherever the process was started (the package location is the anchor, not
#: the cwd, so ``python -m bebop_vision.lab ls`` works from any directory).
VISION_DIR = Path(__file__).resolve().parents[2]
TOOLS_DIR = VISION_DIR / "tools"

#: Where drive MCAPs live on this workstation, relative to :data:`VISION_DIR`.
#: The first root is also where ``pull()`` lands files by default.
DEFAULT_ROOTS: Tuple[str, ...] = ("datasets/sessions", "datasets/bev-student/raw/mcap")
DEFAULT_ROBOT = "http://bebop.local:9090"
#: The robot's log of every closed run (bebop-vision ``GET :9092/runs``),
#: mirrored next to the sessions by :meth:`Catalog.pull_runs`.
RUN_LOG_NAME = "runs.jsonl"
VIDEO_PORT = 9092
DEFAULT_CACHE = "datasets/lab"
CACHE_ENV = "BEBOP_LAB_CACHE"
LEDGER_NAME = "sessions.jsonl"
SESSION_PREFIX = "navd_session_"
SUFFIX = ".mcap"

#: Scalar ledger fields that make sense as columns next to the file facts.
#: Nested fields (``topics``, ``intents``, ``events``, ``health``) stay in the
#: full row that :meth:`Catalog.ledger_row` / :meth:`Catalog.info` return.
LEDGER_SCHEMA: Dict[str, pl.DataType] = {
    "footerless": pl.Boolean,
    "start_ns": pl.Int64,
    "end_ns": pl.Int64,
    "duration_s": pl.Float64,
    "messages": pl.Int64,
    "ticks": pl.Int64,
    "tick_hz": pl.Float64,
    "frame_gaps": pl.Int64,
    "frame_gap_max_s": pl.Float64,
    "gyro_hz": pl.Float64,
    "gyro_gaps": pl.Int64,
    "gyro_gap_max_ms": pl.Float64,
    "moving_frac": pl.Float64,
    "reverse_frac": pl.Float64,
    "turn_events": pl.Int64,
    "turns_left": pl.Int64,
    "turns_right": pl.Int64,
    "odom_path_m": pl.Float64,
    "runtime_messages": pl.Int64,
    "intent": pl.Utf8,
    "places": pl.List(pl.Utf8),
}

LOCAL_SCHEMA: Dict[str, pl.DataType] = {
    "name": pl.Utf8,
    "path": pl.Utf8,
    "size_bytes": pl.Int64,
    "mtime": pl.Datetime("ms", "UTC"),
    "has_cache": pl.Boolean,
}

REMOTE_SCHEMA: Dict[str, pl.DataType] = {
    "name": pl.Utf8,
    "file": pl.Utf8,
    "size_bytes": pl.Int64,
    "modified_ms": pl.Int64,
    "active": pl.Boolean,
    "eligible": pl.Boolean,
    "local": pl.Boolean,
}

#: One row per drive run (:meth:`Catalog.runs`).
RUNS_SCHEMA: Dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "task": pl.Utf8,
    "target": pl.Utf8,
    "text": pl.Utf8,
    "from": pl.Utf8,
    "to": pl.Utf8,
    "sessions": pl.List(pl.Utf8),
    "start_ns": pl.Int64,
    "end_ns": pl.Int64,
    "n_sessions": pl.Int64,
    "outcome": pl.Utf8,
    "outcome_source": pl.Utf8,
    "interruptions": pl.Int64,
    "stops": pl.Int64,
    "stop_causes": pl.List(pl.Utf8),
    "resumes": pl.Int64,
    "resume_notes": pl.List(pl.Utf8),
}

#: The outcome of a run closed without arrived / found / gave_up: the
#: operator declined to continue it, cleared it, or started another run over
#: it (``session_notes.UNFINISHED``). Recordings made before 2026-10-25 closed
#: such runs with no outcome at all; :meth:`Catalog.runs` reads that "" as
#: this too, and so does a run whose recordings show no ending (its
#: ``outcome_source`` then says which case it is, see :data:`OUTCOME_SOURCES`).
UNFINISHED = "unfinished"
#: Where :meth:`Catalog.runs` got a run's ``outcome`` from, so a training
#: consumer can tell an operator's decision from a default: ``event`` (an
#: ending ``/event`` in the recordings), ``intent`` (no event, but the
#: recordings show the run cleared or replaced: a close written before
#: 2026-10-25, or a clear / new run over it; reads as UNFINISHED), ``log``
#: (the robot's run log passed as ``run_log``: the run ended while nothing
#: recorded, e.g. Arrived on the end-of-run sheet after Stop driving
#: disarmed and closed the segment), ``none`` (no ending seen anywhere: the
#: run was still open where its recordings end, or it ended unrecorded and
#: no log covers it; reads as UNFINISHED).
OUTCOME_SOURCES = ("event", "intent", "log", "none")
#: ``/event`` kinds that end a run (``SessionNotes.end_run``); ``interrupted``
#: is a marker inside a run and is counted instead.
RUN_END_OUTCOMES = ("arrived", "found", "gave_up", UNFINISHED)
RUN_MARKER = "interrupted"
#: ``/event`` kinds that mark a break in a run: ``stop`` is written into the
#: closing segment with note ``"<cause>"`` or ``"<cause>: <detail>"``
#: (``SessionNotes.note_stop``), ``resume`` is the operator choosing to go
#: on with the open run, written as the first event of the next segment with
#: the operator's note (``SessionNotes.resume_run``).
RUN_STOP = "stop"
RUN_RESUME = "resume"
#: Stop causes as the recorder writes them (``session_notes.STOP_CAUSES``);
#: listed for reference, a cause outside it is kept as written.
STOP_CAUSES = ("disarmed", "wheel_disarmed", "idle", "estop_operator", "estop_auto",
               "link_lost", "rolled", "recorder_stop")
#: How long after a run's span ends an outcome / marker event in no span
#: still counts as that run's (see run_spans; a stop in no span is no run's,
#: since ``note_stop`` adds a stop only to a run still open): the recorder ticks at ~30 Hz,
#: so the race is a few ms; 1 s also covers a tick delayed by a GC pause or a
#: slow MCAP write, and is well below the time an operator takes to start the
#: next run after ending one.
RUN_EVENT_SLACK_NS = 1_000_000_000

GLOB_CHARS = "*?["

log = logging.getLogger(__name__)


def _tool(name: str):
    """``tools/<name>.py`` as a module, by package import when the caller
    already put ``bebop-vision`` on ``sys.path`` (the tests do), else by
    path. Registered under ``tools.<name>`` so both routes share one module."""
    qualified = f"tools.{name}"
    mod = sys.modules.get(qualified)
    if mod is not None:
        return mod
    try:
        return importlib.import_module(qualified)
    except ImportError:
        pass
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        # Otherwise the next call returns this empty module and the caller
        # fails on an AttributeError instead of the real import error.
        sys.modules.pop(qualified, None)
        raise
    return mod


def stem_of(name_or_file: str) -> str:
    """``navd_session_x.mcap`` and ``navd_session_x`` both name the same session."""
    s = str(name_or_file)
    return s[:-len(SUFFIX)] if s.endswith(SUFFIX) else s


def cache_root(cache_dir: Union[str, Path, None] = None) -> Path:
    """The lab cache root: explicit, else ``$BEBOP_LAB_CACHE``, else
    ``bebop-vision/datasets/lab``. Not created here."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return VISION_DIR / DEFAULT_CACHE


def _is_glob(s: str) -> bool:
    return any(c in s for c in GLOB_CHARS)


class Catalog:
    """Drive sessions on this machine, on the robot, and what the ledger says.

    ``roots``: directories holding ``navd_session_*.mcap``; relative paths
    resolve against ``base_dir`` (default :data:`VISION_DIR`). Missing roots
    are skipped, not errors, so one Catalog works on the workstation and on
    the robot. A stem present in two roots is taken from the earlier root.
    ``ledger``: a ``sessions.jsonl`` path; ``None`` reads every
    ``<root>/sessions.jsonl`` that exists. ``cache_dir``: the lab cache root
    (see :func:`cache_root`); ``has_cache`` means ``<cache>/<stem>`` exists.
    ``min_age_s`` is the mirror script's "finished capture" rule for
    ``remote()`` / ``pull()``.
    """

    def __init__(self, roots: Sequence[Union[str, Path]] = DEFAULT_ROOTS,
                 robot: str = DEFAULT_ROBOT, ledger: Union[str, Path, None] = None,
                 cache_dir: Union[str, Path, None] = None,
                 base_dir: Union[str, Path, None] = None, min_age_s: float = 30.0):
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else VISION_DIR
        if isinstance(roots, (str, Path)):
            roots = (roots,)
        self.roots: List[Path] = [self._resolve_root(r) for r in roots]
        self.robot = robot.rstrip("/")
        self.ledger_path: Optional[Path] = (Path(ledger).expanduser().resolve()
                                            if ledger is not None else None)
        self.cache_dir = cache_root(cache_dir)
        self.min_age_s = float(min_age_s)
        #: file name -> transfer error text for the ``failed`` rows of the last
        #: :meth:`pull`; the state tuple stays ``(name, state)`` per the contract.
        self.pull_errors: Dict[str, str] = {}
        #: stems the last :meth:`runs` skipped because they had no valid index.
        self.runs_unindexed: List[str] = []

    def _resolve_root(self, r: Union[str, Path]) -> Path:
        p = Path(r).expanduser()
        return (p if p.is_absolute() else self.base_dir / p).resolve()

    # -- disk ----------------------------------------------------------------

    def files(self) -> Dict[str, Path]:
        """stem -> path over the roots, first root wins, sorted by stem (the
        stem carries the start time, so this is chronological)."""
        found: Dict[str, Path] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            for p in sorted(root.glob(f"{SESSION_PREFIX}*{SUFFIX}")):
                if p.is_file() and p.stem not in found:
                    found[p.stem] = p
        return dict(sorted(found.items()))

    def names(self) -> List[str]:
        return list(self.files())

    def cache_dir_for(self, name: str) -> Path:
        return self.cache_dir / stem_of(name)

    def has_cache(self, name: str) -> bool:
        return self.cache_dir_for(name).is_dir()

    def local(self) -> pl.DataFrame:
        """One row per session on disk: ``name`` (stem), ``path``,
        ``size_bytes``, ``mtime`` (UTC), ``has_cache``, plus the scalar ledger
        columns (:data:`LEDGER_SCHEMA`) when a ledger was found; sessions the
        ledger does not know get nulls there."""
        rows = []
        for stem, p in self.files().items():
            st = p.stat()
            rows.append({
                "name": stem,
                "path": str(p),
                "size_bytes": int(st.st_size),
                "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                "has_cache": self.has_cache(stem),
            })
        df = pl.DataFrame(rows, schema=LOCAL_SCHEMA)
        ledger = self.ledger()
        if ledger is not None:
            df = df.join(ledger, on="name", how="left", maintain_order="left")
        return df

    # -- ledger --------------------------------------------------------------

    def ledger_paths(self) -> List[Path]:
        if self.ledger_path is not None:
            return [self.ledger_path] if self.ledger_path.is_file() else []
        return [root / LEDGER_NAME for root in self.roots if (root / LEDGER_NAME).is_file()]

    def ledger_rows(self) -> Dict[str, dict]:
        """stem -> the full ledger row (nested fields included). A malformed
        line is skipped; a later row for the same session replaces an earlier
        one, so appending a re-analysis to the file is enough."""
        rows: Dict[str, dict] = {}
        for path in self.ledger_paths():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict) and row.get("name"):
                        rows[stem_of(row["name"])] = row
        return rows

    def ledger_row(self, name: str) -> Optional[dict]:
        return self.ledger_rows().get(stem_of(name))

    def ledger(self) -> Optional[pl.DataFrame]:
        """The scalar ledger columns keyed by ``name`` (stem), or ``None``
        when no ledger file exists."""
        if not self.ledger_paths():
            return None
        return ledger_frame(self.ledger_rows().values())

    # -- robot ---------------------------------------------------------------

    def remote(self, timeout: float = 10.0) -> pl.DataFrame:
        """What the firmware's ``GET /captures`` lists, filtered to the mirror
        script's prefixes (drive sessions and the firmware health log):
        ``name`` (stem), ``file``, ``size_bytes``, ``modified_ms``, ``active``,
        ``eligible`` (what ``pull()`` would take: not active, older than
        ``min_age_s`` on the robot's clock) and ``local`` (a file of that
        name and size sits in one of the roots). Never call this in a test
        without a stand-in server."""
        files, eligible = self._listing(timeout)
        rows = []
        for fname, f in files.items():
            size = int(f.get("sizeBytes", 0))
            rows.append({
                "name": stem_of(fname),
                "file": fname,
                "size_bytes": size,
                "modified_ms": int(float(f.get("modifiedMs", 0))),
                "active": bool(f.get("active", False)),
                "eligible": fname in eligible,
                "local": self._have_local(fname, size),
            })
        rows.sort(key=lambda r: r["name"])
        return pl.DataFrame(rows, schema=REMOTE_SCHEMA)

    def _listing(self, timeout: float) -> Tuple[Dict[str, dict], set]:
        """``GET /captures`` reduced to what the catalog handles: file name ->
        entry for the mirror script's prefixes (no path separators, so a
        hostile listing cannot name a file outside ``dst``), plus the set of
        names ``select`` considers finished."""
        pc = _tool("pull_captures")
        files, now_ms = pc.list_remote(self.robot, timeout=timeout)
        listed = {f["name"]: f for f in files
                  if f.get("name") and f["name"].startswith(pc.PREFIXES) and "/" not in f["name"]}
        eligible = {f["name"] for f in pc.select(files, now_ms, min_age_s=self.min_age_s)}
        return listed, eligible

    def _have_local(self, fname: str, size: int, extra: Optional[Path] = None) -> bool:
        """A copy of ``fname`` at the remote size sits in a root (or ``extra``)."""
        dirs = list(self.roots) + ([extra] if extra is not None else [])
        for d in dirs:
            p = d / fname
            if p.is_file() and p.stat().st_size == size:
                return True
        return False

    def run_log_path(self) -> Optional[Path]:
        """The mirrored robot run log (``<root>/runs.jsonl`` in the first
        root that has one), or None before :meth:`pull_runs` ran."""
        for root in self.roots:
            if (root / RUN_LOG_NAME).is_file():
                return root / RUN_LOG_NAME
        return None

    def pull_runs(self, dst: Union[str, Path, None] = None, timeout: float = 10.0,
                  url: Optional[str] = None) -> Path:
        """Mirror the robot's run log (``GET :9092/runs``, every closed run
        with its outcome, end note, stops and resumes) to
        ``<dst or first root>/runs.jsonl``. The outcome picked on the app's
        end-of-run sheet after Stop driving is in no drive file (disarming
        closed it), so :meth:`runs` needs this log to know how such a run
        ended. The whole log is rewritten each time (it is small)."""
        if url is None:
            host = urllib.parse.urlparse(self.robot).hostname or "bebop.local"
            url = f"http://{host}:{VIDEO_PORT}/runs"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = json.load(resp)
        if not isinstance(body, Mapping) or not isinstance(body.get("runs", []), list):
            raise ValueError(f"{url}: expected {{\"runs\": [...]}}")
        dst_dir = Path(dst) if dst is not None else (self.roots[0] if self.roots else Path("."))
        dst_dir.mkdir(parents=True, exist_ok=True)
        out = dst_dir / RUN_LOG_NAME
        # Merge, never replace: the mirror is the only copy off the robot,
        # and an empty or partial answer (persistence off, a wiped robot, a
        # read error there) must not erase history here. Per run id the row
        # that ended last wins (see :func:`run_log_outcomes`).
        merged = _latest_rows(_read_jsonl(out) + [r for r in body.get("runs") or []
                                                 if isinstance(r, Mapping)])
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in merged), encoding="utf-8")
        os.replace(tmp, out)
        return out

    def pull(self, names_or_glob: Union[str, Iterable[str]], dst: Union[str, Path, None] = None,
             finalize: bool = True, timeout: float = 30.0) -> List[Tuple[str, str]]:
        """Fetch captures from the robot into ``dst`` (default: the first
        root). Each pattern is a stem, a file name or a glob matched against
        both, over the drive sessions and health logs the robot lists (other
        files there are invisible here, as in ``remote()``). Returns
        ``(file name, state)`` per matched or named file: ``have`` (a copy at
        the remote size already sits in ``dst`` or in any root, so the six
        2026-10-18 drives in the second default root are never re-pulled into
        the first), ``pulled`` / ``resumed`` from ``pull_captures.fetch``,
        ``busy`` (active or younger than ``min_age_s``: the recorder may still
        be writing it), ``failed`` (a transfer error; the ``.part`` stays for a
        resume and the message is in :attr:`pull_errors` and the log), or
        ``absent`` for an explicit name the robot does not list.
        ``finalize`` posts ``/captures/finalize`` first so the live health log
        rolls over and the copy has a footer; a failed finalize is not fatal.
        """
        pc = _tool("pull_captures")
        patterns = [names_or_glob] if isinstance(names_or_glob, str) else list(names_or_glob)
        if dst is not None:
            dst_dir = Path(dst).expanduser().resolve()
        elif self.roots:
            dst_dir = self.roots[0]
        else:
            raise ValueError("pull() needs a dst: this Catalog has no roots")
        dst_dir.mkdir(parents=True, exist_ok=True)
        self.pull_errors = {}
        if finalize:
            try:
                req = urllib.request.Request(f"{self.robot}/captures/finalize", method="POST")
                urllib.request.urlopen(req, timeout=10.0).close()
            except (urllib.error.URLError, OSError) as exc:
                log.warning("finalize on %s failed (%s); continuing", self.robot, exc)
        listed, eligible = self._listing(timeout)
        chosen: List[str] = []
        results: List[Tuple[str, str]] = []
        for pat in patterns:
            matches = [n for n in listed if _match(pat, n)]
            if not matches:
                if not _is_glob(pat):
                    results.append((pat if pat.endswith(SUFFIX) else pat + SUFFIX, "absent"))
                continue
            for n in sorted(matches):
                if n not in chosen:
                    chosen.append(n)
        for fname in chosen:
            size = int(listed[fname].get("sizeBytes", 0))
            if self._have_local(fname, size, dst_dir):
                results.append((fname, "have"))
                continue
            if fname not in eligible:
                results.append((fname, "busy"))
                continue
            try:
                state = pc.fetch(self.robot, fname, size, dst_dir, timeout=timeout)
            except (urllib.error.URLError, OSError) as exc:
                state = "failed"
                self.pull_errors[fname] = str(exc)
                log.warning("pull %s failed: %s", fname, exc)
            results.append((fname, state))
        return results

    # -- runs ------------------------------------------------------------------

    def runs(self, index: bool = False, run_log: Any = None) -> pl.DataFrame:
        """One row per drive run (``run_id``) over the local sessions, in
        order of ``start_ns``; columns in :data:`RUNS_SCHEMA`.

        A run is what the operator started with ``POST /intent`` (bebop-vision
        ``session_notes``); the recorder writes the intent in force at the
        start of every segment and again on every change, so one run usually
        spans several consecutive session files, and one file can hold the
        end of one run and the start of the next. Within a session a run is
        in force from its first ``/intent`` row until the next row with a
        different ``run_id`` (another run, or a clear) or the end of the file.

        ``text``, ``task``, ``target``, ``from``, ``to`` are the run's last
        recorded values (``mode: "continue"`` may update them mid-run).
        ``sessions``: the stems the run is in force in, in time order.
        ``start_ns``: the run's ``run_started_ns`` when recorded (the run may
        have started before the first recorded segment), else its first
        ``/intent`` row; ``end_ns``: where it stops being in force in its
        last session. ``outcome``: the last ``arrived`` / ``found`` /
        ``gave_up`` / ``unfinished`` ``/event`` inside the run's span, else
        the robot's outcome from ``run_log`` for a run it shows closed, else
        :data:`UNFINISHED` (a run cleared or replaced before 2026-10-25, when
        such closes wrote no event; a run still open where its recordings
        end; one ended while nothing was recording); ``outcome_source``:
        which of these it is (:data:`OUTCOME_SOURCES`).

        ``run_log``: the robot's own record of how runs ended, needed because
        a run is often ended while no segment records (Stop driving disarms
        first, which closes the segment, then the operator picks Arrived on
        the end-of-run sheet), and such an outcome is in no file. Default:
        the mirrored robot run log (:meth:`run_log_path`, written by
        :meth:`pull_runs` / ``lab pull``), which holds every closed run.
        Also accepted: a ``runs.jsonl`` path, a path to bebop-vision's
        ``--intent-state`` JSON or its parsed dict (only the last
        ``session_notes.MAX_RECENT`` runs, plus runs closed before the run
        log existed), or rows (``run_id``, ``ended_ns``, ``outcome``). A
        recorded ending event wins over it.
        ``interruptions``: its ``interrupted`` events. ``stops`` /
        ``stop_causes``: its ``stop`` events and their causes, in time order
        (why each of its segments ended, e.g. ``disarmed``, ``estop_auto``);
        ``resumes`` / ``resume_notes``: its ``resume`` events and the
        operator's notes (the operator chose to go on after a break). Intent
        rows without a ``run_id`` (recordings made before drive runs) are
        not runs.

        Cost: the intent and event series come from each session's lab index
        (``series.npz``, milliseconds per session) plus ``read_info`` for the
        file end (the summary; a footerless file is scanned). A session
        without a valid index is skipped and listed in
        :attr:`runs_unindexed`, unless ``index=True``, which builds the
        missing indexes first: one streaming pass per MCAP that also copies
        its colour and depth payloads into the cache (seconds per GB, about
        the MCAP's size on disk).

        Labels are still per session (``labels.py`` does not stitch a run's
        segments together); following a run across segment boundaries is
        future work, and so is a split that is not by run (see
        ``dataset.make_run_splits``).
        """
        from bebop_vision.lab.session import Session   # lazy: the catalog stays light

        self.runs_unindexed = []
        acc: Dict[str, Dict[str, Any]] = {}
        for stem, path in self.files().items():
            sess = Session(path, cache_dir=self.cache_dir_for(stem))
            if not sess.indexed and not index:
                self.runs_unindexed.append(stem)
                continue
            spans = run_spans(sess.series("intent").data, sess.series("event").data, sess.end_ns)
            for span in spans:
                rid = span["run_id"]
                run = acc.get(rid)
                if run is None:
                    run = acc[rid] = {"run_id": rid, "sessions": [], "first_ns": span["start_ns"],
                                      "started_ns": 0, "end_ns": span["end_ns"], "last_ns": -1,
                                      "outcome": "", "outcome_ns": -1, "interruptions": 0,
                                      "stops": [], "resumes": [], "closed": False,
                                      "closed_end_ns": -1}
                if stem not in run["sessions"]:
                    run["sessions"].append(stem)
                run["first_ns"] = min(run["first_ns"], span["start_ns"])
                run["end_ns"] = max(run["end_ns"], span["end_ns"])
                if span["end_ns"] >= run["closed_end_ns"]:
                    # the latest span decides: one that runs to its file's
                    # end leaves the run open there, even after an earlier
                    # span was replaced (a resumed run)
                    run["closed_end_ns"], run["closed"] = span["end_ns"], span["closed"]
                if span["run_started_ns"] > 0 and not run["started_ns"]:
                    run["started_ns"] = span["run_started_ns"]
                if span["last_ns"] >= run["last_ns"]:
                    run["last_ns"] = span["last_ns"]
                    for k in ("task", "target", "text", "from", "to"):
                        run[k] = span[k]
                if span["outcome"] and span["outcome_ns"] >= run["outcome_ns"]:
                    run["outcome"], run["outcome_ns"] = span["outcome"], span["outcome_ns"]
                run["interruptions"] += span["interruptions"]
                run["stops"].extend(span["stops"])
                run["resumes"].extend(span["resumes"])
        if run_log is None:
            run_log = self.run_log_path()
        logged = run_log_outcomes(run_log)
        rows = []
        for run in acc.values():
            if run["outcome"]:
                outcome, source = run["outcome"], "event"
            elif run["run_id"] in logged:
                outcome, source = logged[run["run_id"]], "log"
            elif run["closed"]:
                outcome, source = UNFINISHED, "intent"
            else:
                outcome, source = UNFINISHED, "none"
            stops = sorted(run["stops"], key=lambda b: b["t_ns"])
            resumes = sorted(run["resumes"], key=lambda b: b["t_ns"])
            rows.append({
                "run_id": run["run_id"], "task": run["task"], "target": run["target"],
                "text": run["text"], "from": run["from"], "to": run["to"],
                "sessions": list(run["sessions"]),
                "start_ns": int(run["started_ns"] or run["first_ns"]), "end_ns": int(run["end_ns"]),
                "n_sessions": len(run["sessions"]), "outcome": outcome, "outcome_source": source,
                "interruptions": int(run["interruptions"]),
                "stops": len(stops), "stop_causes": [b["cause"] for b in stops],
                "resumes": len(resumes), "resume_notes": [b["note"] for b in resumes],
            })
        rows.sort(key=lambda r: (r["start_ns"], r["run_id"]))
        return pl.DataFrame(rows, schema=RUNS_SCHEMA)

    # -- one session -----------------------------------------------------------

    def resolve(self, name_or_path: Union[str, Path]) -> Path:
        """The MCAP a user means: an existing path (as given), a stem or file
        name in the roots, a unique fragment of a stem (``20261018_191941``),
        or a glob over the roots' stems (also a path with
        glob characters, matched against its own directory, relative to the
        cwd like a plain path; only ``.mcap`` files count, so a ``.part`` left
        by an interrupted pull does not make the pattern ambiguous). Exactly
        one file must match; several raise ``ValueError`` naming them, none
        raise ``FileNotFoundError``."""
        if isinstance(name_or_path, Path):
            p = name_or_path.expanduser()
            if p.is_file():
                return p.resolve()
            s = str(name_or_path)
        else:
            s = str(name_or_path)
            p = Path(s).expanduser()
            if not _is_glob(s) and p.is_file():
                return p.resolve()
        if _is_glob(s) and ("/" in s or os.sep in s):
            # a path pattern: glob inside its own directory, not the roots
            base = p if p.is_absolute() else Path.cwd() / p
            matches = sorted(q for q in base.parent.glob(base.name)
                             if q.is_file() and q.suffix == SUFFIX)
            return _single(s, matches)
        files = self.files()
        key = stem_of(s)
        if key in files:
            return files[key]
        if _is_glob(s):
            matches = [files[k] for k in files if _match(s, k) or _match(s, k + SUFFIX)]
            return _single(s, matches)
        # a fragment of the stem ("20261018_191941"): unique substring match
        matches = [files[k] for k in files if key and key in k]
        if matches:
            return _single(s, matches)
        raise FileNotFoundError(f"no session {s!r} in {[str(r) for r in self.roots]}")

    def info(self, name: Union[str, Path], analyse: bool = False) -> dict:
        """The light facts about one session: ``read_info`` (span, counts,
        footerless), the file identity, the cache dir and whether it exists,
        and the ledger row under ``"ledger"`` when there is one. ``analyse``
        runs ``session_ledger.analyse_session`` (one pass over the small
        channels, seconds per GB) and puts that row under ``"ledger"``
        instead; the ledger file is not updated."""
        path = self.resolve(name)
        mi = read_info(path)
        stem = path.stem
        out: Dict[str, Any] = {
            "name": stem,
            "path": str(path),
            "size_bytes": mi.size_bytes,
            "identity": file_identity(path),
            "footerless": mi.footerless,
            "start_ns": mi.start_ns,
            "end_ns": mi.end_ns,
            "duration_s": round(mi.duration_s, 3),
            "messages": mi.message_count,
            "topics": dict(sorted(mi.message_counts.items())),
            "cache_dir": str(self.cache_dir_for(stem)),
            "has_cache": self.has_cache(stem),
            "ledger": self.ledger_row(stem),
        }
        if analyse:
            out["ledger"] = _tool("session_ledger").analyse_session(path)
        return out


# -- helpers -----------------------------------------------------------------

def _match(pattern: str, name: str) -> bool:
    """A pattern names ``name`` when it equals its stem or file name, or
    globs either; a bare stem also matches the ``.mcap`` form."""
    stem = stem_of(name)
    if _is_glob(pattern):
        return fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(stem, pattern)
    return stem_of(pattern) == stem


def _single(what: str, matches: Sequence[Path]) -> Path:
    if len(matches) == 1:
        return Path(matches[0]).resolve()
    if not matches:
        raise FileNotFoundError(f"nothing matches {what!r}")
    raise ValueError(f"{what!r} matches {len(matches)} files: "
                     + ", ".join(m.name for m in matches))


def run_spans(intent: Any, events: Any = None, end_ns: int = 0) -> List[Dict[str, Any]]:
    """The runs in force in one session, in time order.

    ``intent`` / ``events``: the session's ``intent`` and ``event`` series
    (structured arrays sorted by ``t_ns``, :data:`types.SERIES_DTYPES`);
    ``end_ns``: the session's last log time. A run's span starts at its
    first row and ends at the next row with another ``run_id`` (the clear
    that ``end_run`` writes, or the next run) or at ``end_ns``. Consecutive
    rows of one run (a ``continue`` update) extend one span, whose fields
    are the last row's. Rows without a ``run_id`` open no span.

    Each span: ``run_id, task, target, text, from, to, run_started_ns,
    start_ns, end_ns, last_ns`` (the stamp of its last row), ``outcome`` /
    ``outcome_ns`` (the last run-ending event inside it, "" / -1 for none;
    ``runs()`` reads "" as :data:`UNFINISHED`), ``interruptions``, and
    ``stops`` / ``resumes``: the span's breaks as :func:`run_breaks` dicts,
    and ``closed``: True when a row with another ``run_id`` ends the span
    (the run was cleared or replaced while recording), False when it runs to
    ``end_ns``.
    An event inside a span (ends inclusive, so one stamped exactly on a
    boundary belongs to the span that ends there) is that span's.

    The recorder stamps the two sides of an ``end_run`` differently: the
    clear row gets the tick's ``log_ns``, taken at the start of the tick,
    while the outcome event gets ``end_run``'s own clock reading. When the
    operator ends a run after the tick took ``log_ns`` but before it drained
    the notes, the event is a few ms *after* the clear and falls in the gap
    a clear leaves. Such an event (outside every span) goes to the latest
    span that ended at most :data:`RUN_EVENT_SLACK_NS` before it, so the
    outcome is not lost. An event in no span and later than that is not
    any run's (``end_run`` with no run open, or a stalled recorder). A
    ``stop`` event in no span is no run's at all: ``note_stop`` adds a stop
    only to a run still open, and that run's span covers the stop.

    A ``resume`` event in no span goes to the next span instead: it is
    written when the segment opens, so it can be stamped just before the
    segment's first ``/intent`` row, and it always means the run that row
    carries on."""
    spans: List[Dict[str, Any]] = []
    n = len(intent) if intent is not None else 0
    t_rows = [int(t) for t in intent["t_ns"]] if n else []
    for i in range(n):
        r = intent[i]
        t0 = t_rows[i]
        t1 = t_rows[i + 1] if i + 1 < n else max(int(end_ns), t0)
        rid = str(r["run_id"])
        if not rid:
            continue
        fields = {"task": str(r["task"]), "target": str(r["target"]), "text": str(r["text"]),
                  "from": str(r["from_place"]), "to": str(r["to_place"])}
        if spans and spans[-1]["run_id"] == rid and spans[-1]["end_ns"] == t0:
            spans[-1].update(fields, end_ns=t1, last_ns=t0, closed=i + 1 < n)
            if not spans[-1]["run_started_ns"]:
                spans[-1]["run_started_ns"] = int(r["run_started_ns"])
            continue
        spans.append({"run_id": rid, **fields, "run_started_ns": int(r["run_started_ns"]),
                      "start_ns": t0, "end_ns": t1, "last_ns": t0,
                      "outcome": "", "outcome_ns": -1, "interruptions": 0,
                      "stops": [], "resumes": [], "closed": i + 1 < n})
    if events is not None and len(events) and spans:
        for ev in events:
            kind = str(ev["kind"])
            if kind not in RUN_END_OUTCOMES and kind not in (RUN_MARKER, RUN_STOP, RUN_RESUME):
                continue
            t = int(ev["t_ns"])
            span = next((sp for sp in spans if sp["start_ns"] <= t <= sp["end_ns"]), None)
            if span is None and kind == RUN_RESUME:
                span = next((sp for sp in spans if sp["start_ns"] > t), None)
            if span is None and kind == RUN_STOP:
                continue
            if span is None:
                before = [sp for sp in spans if sp["end_ns"] < t]
                if not before or t - before[-1]["end_ns"] > RUN_EVENT_SLACK_NS:
                    continue
                span = before[-1]
            if kind == RUN_MARKER:
                span["interruptions"] += 1
            elif kind == RUN_STOP:
                span["stops"].append(_break(t, kind, str(ev["note"])))
            elif kind == RUN_RESUME:
                span["resumes"].append(_break(t, kind, str(ev["note"])))
            elif t >= span["outcome_ns"]:
                span["outcome"], span["outcome_ns"] = kind, t
    return spans


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Rows of a JSON-lines file, skipping lines that do not parse (a line
    cut by a power loss on the robot); ``[]`` when the file is missing."""
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _latest_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One row per ``run_id``: the one with the largest ``ended_ns`` (a crash
    between the robot's log append and its state save can log a run twice;
    the later close is the operator's last word), in ``ended_ns`` order.
    Rows without a run id are dropped."""
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = str(row.get("run_id") or "")
        if not rid:
            continue
        if rid not in best or int(row.get("ended_ns") or 0) >= int(best[rid].get("ended_ns") or 0):
            best[rid] = dict(row)
    return sorted(best.values(), key=lambda r: int(r.get("ended_ns") or 0))


def run_log_outcomes(run_log: Any) -> Dict[str, str]:
    """``{run_id: outcome}`` for the runs the robot's run log shows closed
    (``ended_ns`` > 0), from any ``run_log`` form :meth:`Catalog.runs`
    takes: the mirrored ``runs.jsonl``, a state file (``intent.json``), its
    dict, or a list of rows in any order. Per run id the row that ended last
    wins; an older log's "" outcome reads as :data:`UNFINISHED`. An open run
    (no ``ended_ns``) is left out: it may still be resumed. ``None`` gives
    ``{}``; an unreadable state file raises."""
    if run_log is None:
        return {}
    if isinstance(run_log, (str, Path)):
        if str(run_log).endswith(".jsonl"):
            run_log = _read_jsonl(Path(run_log))
        else:
            run_log = json.loads(Path(run_log).read_text(encoding="utf-8"))
    if isinstance(run_log, Mapping):
        run_log = run_log.get("recent") or []
    rows = [r for r in run_log if isinstance(r, Mapping) and int(r.get("ended_ns") or 0)]
    return {r["run_id"]: str(r.get("outcome") or "") or UNFINISHED for r in _latest_rows(rows)}


def _break(t_ns: int, kind: str, note: str) -> Dict[str, Any]:
    """One break dict (see :func:`run_breaks`) from an event's kind and note."""
    if kind == RUN_STOP:
        cause, _, detail = note.partition(":")
        return {"t_ns": int(t_ns), "kind": kind, "cause": cause.strip(), "note": detail.strip()}
    return {"t_ns": int(t_ns), "kind": kind, "cause": "", "note": note}


def run_breaks(session: Any) -> List[Dict[str, Any]]:
    """The breaks in one session, in time order: ``{t_ns, kind, cause,
    note}`` per ``stop`` / ``resume`` event.

    ``session``: a :class:`~bebop_vision.lab.session.Session` (its ``event``
    series is read, so it must be indexed or indexable) or the event series
    itself (a structured array with ``t_ns``, ``kind``, ``note``). For a
    ``stop`` the event note ``"<cause>: <detail>"`` splits into ``cause``
    and ``note`` = the detail (the firmware's E-STOP reason, or the wheels
    that were disarmed; "" when there is none). For a ``resume``, ``cause``
    is "" and ``note`` is the operator's note. Recordings made before
    2026-10-25 have no such events and give ``[]``. Which run a break
    belongs to is :func:`run_spans`' business (its ``stops`` / ``resumes``).
    """
    events = session.series("event").data if hasattr(session, "series") else session
    if events is None or not len(events):
        return []
    out = [_break(int(ev["t_ns"]), str(ev["kind"]), str(ev["note"]))
           for ev in events if str(ev["kind"]) in (RUN_STOP, RUN_RESUME)]
    out.sort(key=lambda b: b["t_ns"])
    return out


def ledger_frame(rows: Iterable[dict]) -> pl.DataFrame:
    """Ledger rows (``session_ledger.analyse_session`` dicts) as the scalar
    columns in :data:`LEDGER_SCHEMA` keyed by ``name`` (stem). ``intent`` is
    the text of the row's last non-empty intent; ``places`` the place names
    in order of first sighting."""
    out = []
    for r in rows:
        rec: Dict[str, Any] = {"name": stem_of(r.get("name", ""))}
        for col in LEDGER_SCHEMA:
            if col == "intent":
                intent = r.get("intent") or {}
                rec[col] = (intent.get("text") or None) if isinstance(intent, dict) else None
            elif col == "places":
                places = r.get("places") or []
                rec[col] = [str(p.get("name", "")) for p in places if isinstance(p, dict)]
            else:
                rec[col] = r.get(col)
        out.append(rec)
    schema = {"name": pl.Utf8, **LEDGER_SCHEMA}
    return pl.DataFrame(out, schema=schema)
