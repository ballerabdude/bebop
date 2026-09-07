"""main() mode gating (§7.3): driving is model-only.

`--goal-drive` must refuse to start without `--navd-model` — the student
model is the only drive-time BEV source; the geometric drive path was
removed, so a bare `--goal-drive` is a usage error, not a silent
geometric session.
"""

import pytest


def test_goal_drive_requires_navd_model(monkeypatch, capsys):
    from main import main
    monkeypatch.setattr("sys.argv", ["main.py", "--goal-drive"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--navd-model" in capsys.readouterr().err


def test_record_navd_goal_drive_requires_navd_model(monkeypatch, capsys):
    from main import main
    monkeypatch.setattr(
        "sys.argv", ["main.py", "--record-navd", "/tmp/opencode/navd", "--goal-drive"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--navd-model" in capsys.readouterr().err
