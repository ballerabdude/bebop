#!/usr/bin/env python3
"""Companion dashboard for rsl_rl training runs.

TensorBoard has no hook for custom "copy play command" buttons on runs, so
this serves a small local page alongside TensorBoard that scans a log root
and exposes one-click copy for play, resume-train, and export commands.

Usage (from ``sim/`` on the host)::

    python scripts/training_runs_server.py
    python scripts/training_runs_server.py --port 6007 --logdir logs/rsl_rl

Open http://localhost:6007 while TensorBoard runs on :6006.
"""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple


DEFAULT_LOGDIR = "logs/rsl_rl"
DEFAULT_PORT = 6007
DEFAULT_NUM_ENVS = 20
ISAACLAB_SH = "/workspace/isaaclab/isaaclab.sh"
EXPORT_SCRIPT = "bebop_training/export_bebop_model.py"


class RunInfo(NamedTuple):
    task: str
    run_id: str
    run_dir: Path
    latest_checkpoint: str | None
    mtime: float


def _find_sim_cwd(log_root: Path) -> Path:
    """Locate the sim package root (directory containing ``play_bebop.py``)."""
    candidate = log_root.resolve()
    for _ in range(6):
        if (candidate / "play_bebop.py").is_file():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return log_root.resolve().parent


def path_relative_to_sim(path: Path, sim_cwd: Path) -> str:
    """Return ``path`` relative to the sim CWD used inside the container."""
    path = path.resolve()
    sim_cwd = sim_cwd.resolve()
    try:
        return path.relative_to(sim_cwd).as_posix()
    except ValueError:
        return path.as_posix()


def _latest_checkpoint(run_dir: Path) -> str | None:
    ckpts = sorted(
        p.name for p in run_dir.iterdir() if p.is_file() and p.name.startswith("model_") and p.suffix == ".pt"
    )
    return ckpts[-1] if ckpts else None


def discover_runs(log_root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    if not log_root.is_dir():
        return runs

    for task_dir in sorted(log_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            runs.append(
                RunInfo(
                    task=task_dir.name,
                    run_id=run_dir.name,
                    run_dir=run_dir.resolve(),
                    latest_checkpoint=_latest_checkpoint(run_dir),
                    mtime=run_dir.stat().st_mtime,
                )
            )

    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


def format_play_command(
    *,
    task: str,
    resume_path: str,
    num_envs: int = DEFAULT_NUM_ENVS,
    visualizer: str = "kit",
) -> str:
    return (
        f"{ISAACLAB_SH} -p play_bebop.py "
        f"--task {task} "
        f"--num_envs {num_envs} "
        f"--resume {resume_path} "
        f"--visualizer {visualizer}"
    )


def format_train_resume_command(*, task: str, resume_path: str) -> str:
    return f"{ISAACLAB_SH} -p train_bebop.py --task {task} --resume {resume_path}"


def format_export_command(*, checkpoint_path: str) -> str:
    return f"{ISAACLAB_SH} -p {EXPORT_SCRIPT} --checkpoint {checkpoint_path}"


def _render_page(runs: list[RunInfo], log_root: Path, sim_cwd: Path, num_envs: int) -> str:
    rows: list[str] = []
    for run in runs:
        resume_path = path_relative_to_sim(run.run_dir, sim_cwd)
        play_cmd = format_play_command(task=run.task, resume_path=resume_path, num_envs=num_envs)
        train_cmd = format_train_resume_command(task=run.task, resume_path=resume_path)

        export_cmd = ""
        export_btn = ""
        if run.latest_checkpoint:
            checkpoint_path = path_relative_to_sim(run.run_dir / run.latest_checkpoint, sim_cwd)
            export_cmd = format_export_command(checkpoint_path=checkpoint_path)
            export_btn = (
                f'<button type="button" class="copy-btn secondary" '
                f'data-copy="{html.escape(export_cmd, quote=True)}">Copy export</button>'
            )

        when = datetime.fromtimestamp(run.mtime).strftime("%Y-%m-%d %H:%M")
        ckpt = run.latest_checkpoint or "—"
        rows.append(
            f"""
            <tr>
              <td class="mono">{html.escape(run.task)}</td>
              <td class="mono">{html.escape(run.run_id)}</td>
              <td>{html.escape(when)}</td>
              <td class="mono">{html.escape(ckpt)}</td>
              <td class="actions">
                <button type="button" class="copy-btn" data-copy="{html.escape(play_cmd, quote=True)}">
                  Copy play
                </button>
                <button type="button" class="copy-btn secondary" data-copy="{html.escape(train_cmd, quote=True)}">
                  Copy resume train
                </button>
                {export_btn}
              </td>
            </tr>
            <tr class="cmd-row">
              <td colspan="5">
                <code>{html.escape(play_cmd)}</code>
                {f'<code class="cmd-secondary">{html.escape(export_cmd)}</code>' if export_cmd else ""}
              </td>
            </tr>
            """
        )

    body = "\n".join(rows) if rows else '<tr><td colspan="5" class="empty">No runs found.</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bebop training runs</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --accent: #ea580c;
      --accent-hover: #c2410c;
      --code-bg: #f1f5f9;
      --row-alt: #fafafa;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0b1220;
        --panel: #111827;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --border: #1f2937;
        --accent: #fb923c;
        --accent-hover: #fdba74;
        --code-bg: #0f172a;
        --row-alt: #0d1526;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 1.5rem;
    }}
    h1 {{
      margin: 0 0 0.25rem;
      font-size: 1.5rem;
    }}
    .sub {{
      color: var(--muted);
      margin: 0 0 1.25rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      overflow: hidden;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      background: var(--code-bg);
    }}
    tr:not(.cmd-row):hover {{
      background: var(--row-alt);
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85rem;
    }}
    .cmd-row td {{
      padding-top: 0;
      border-bottom: 1px solid var(--border);
    }}
    .cmd-row code {{
      display: block;
      padding: 0.55rem 0.65rem;
      background: var(--code-bg);
      border-radius: 0.45rem;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.82rem;
      margin-bottom: 0.45rem;
    }}
    .cmd-row code:last-child {{
      margin-bottom: 0;
    }}
    .cmd-secondary {{
      opacity: 0.85;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
    }}
    .copy-btn {{
      border: none;
      border-radius: 0.45rem;
      padding: 0.35rem 0.65rem;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-size: 0.82rem;
    }}
    .copy-btn.secondary {{
      background: transparent;
      color: var(--text);
      border: 1px solid var(--border);
    }}
    .copy-btn:hover {{
      filter: brightness(1.05);
    }}
    .copy-btn.secondary:hover {{
      border-color: var(--accent);
      color: var(--accent-hover);
    }}
    .empty {{
      text-align: center;
      color: var(--muted);
      padding: 2rem;
    }}
    .toast {{
      position: fixed;
      right: 1rem;
      bottom: 1rem;
      background: var(--panel);
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      opacity: 0;
      transform: translateY(0.5rem);
      transition: opacity 0.15s ease, transform 0.15s ease;
      pointer-events: none;
    }}
    .toast.show {{
      opacity: 1;
      transform: translateY(0);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Bebop training runs</h1>
    <p class="sub">
      Companion to TensorBoard. Log root: <span class="mono">{html.escape(str(log_root.resolve()))}</span>.
      Paths are relative to sim CWD <span class="mono">{html.escape(str(sim_cwd.resolve()))}</span>
      (container: <span class="mono">/workspace/bebop_bot/sim</span>).
      Export uses the latest checkpoint in each run.
    </p>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>Task</th>
            <th>Run</th>
            <th>Modified</th>
            <th>Latest ckpt</th>
            <th>Commands</th>
          </tr>
        </thead>
        <tbody>
          {body}
        </tbody>
      </table>
    </div>
  </div>
  <div id="toast" class="toast">Copied to clipboard</div>
  <script>
    const toast = document.getElementById("toast");
    let toastTimer = null;

    function showToast() {{
      toast.classList.add("show");
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => toast.classList.remove("show"), 1400);
    }}

    async function copyText(text) {{
      try {{
        await navigator.clipboard.writeText(text);
        showToast();
      }} catch (err) {{
        window.prompt("Copy command:", text);
      }}
    }}

    document.querySelectorAll(".copy-btn").forEach((btn) => {{
      btn.addEventListener("click", () => copyText(btn.dataset.copy));
    }});
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    log_root: Path
    sim_cwd: Path
    num_envs: int

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return

        runs = discover_runs(self.log_root)
        page = _render_page(runs, self.log_root, self.sim_cwd, self.num_envs)
        encoded = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print(f"[training_runs] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a copy-friendly training run dashboard.")
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path(DEFAULT_LOGDIR),
        help=f"Root log directory relative to CWD (default: {DEFAULT_LOGDIR})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help=f"Default --num_envs in generated play commands (default: {DEFAULT_NUM_ENVS})",
    )
    parser.add_argument(
        "--sim-cwd",
        type=Path,
        default=None,
        help="Sim working directory for generated paths (default: auto-detect from logdir)",
    )
    args = parser.parse_args()

    log_root = args.logdir.resolve()
    sim_cwd = args.sim_cwd.resolve() if args.sim_cwd else _find_sim_cwd(log_root)
    Handler.log_root = log_root
    Handler.sim_cwd = sim_cwd
    Handler.num_envs = args.num_envs

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[training_runs] logdir={log_root}")
    print(f"[training_runs] sim_cwd={sim_cwd}")
    print(f"[training_runs] open http://{args.host}:{args.port}/  (TensorBoard usually :6006)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[training_runs] stopped")


if __name__ == "__main__":
    main()
