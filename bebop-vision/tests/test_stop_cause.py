"""main._stop_cause: why the --auto loop closes a recording segment."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import _drive_active, _settled_stop_cause, _stop_cause  # noqa: E402
from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb  # noqa: E402
from bebop_vision.session_notes import STOP_CAUSES  # noqa: E402


class FakeState:
    def __init__(self, **kw):
        self.connected = True
        self.mode = pb.MODE_DIAL_IN
        self.estop_latched = False
        self.estop_reason = ""
        self.wheel_armed = {"left": True, "right": True}
        self.__dict__.update(kw)


class FakeRobot:
    def __init__(self, **kw):
        self.state = FakeState(**kw)


@pytest.mark.parametrize("state, rolled, expected", [
    ({}, False, (None, "")),
    ({}, True, ("rolled", "")),
    ({"wheel_armed": {"left": False, "right": False}}, False, ("disarmed", "")),
    ({"wheel_armed": {"left": True, "right": False}}, False, ("wheel_disarmed", "right")),
    ({"wheel_armed": {"right": False, "left": False, "aux": True}}, False,
     ("wheel_disarmed", "left, right")),
    ({"mode": pb.MODE_IDLE}, False, ("idle", "")),
    ({"estop_latched": True, "estop_reason": "operator: app button"}, False,
     ("estop_operator", "operator: app button")),
    ({"estop_latched": True, "estop_reason": "Operator"}, False,
     ("estop_operator", "Operator")),
    ({"estop_latched": True, "estop_reason": "CAN bus can4: bus off"}, False,
     ("estop_auto", "CAN bus can4: bus off")),
    ({"estop_latched": True, "estop_reason": ""}, False, ("estop_auto", "")),
    ({"connected": False}, False, ("link_lost", "")),
])
def test_stop_cause(state, rolled, expected):
    assert _stop_cause(FakeRobot(**state), rolled) == expected


def test_stop_cause_order():
    """The firmware's IDLE after an E-STOP with wheels disarmed is an E-STOP;
    a lost link wins over whatever stale state it left; a roll only counts
    while the drive state is fine."""
    robot = FakeRobot(estop_latched=True, estop_reason="left: motor fault",
                      mode=pb.MODE_IDLE,
                      wheel_armed={"left": False, "right": False})
    assert _stop_cause(robot, True) == ("estop_auto", "left: motor fault")
    robot.state.connected = False
    assert _stop_cause(robot, True) == ("link_lost", "")
    robot = FakeRobot(mode=pb.MODE_IDLE, wheel_armed={"left": False, "right": False})
    assert _stop_cause(robot, True) == ("idle", "")
    robot = FakeRobot(wheel_armed={"left": False, "right": False})
    assert _stop_cause(robot, True) == ("disarmed", "")


def test_every_cause_is_known_and_agrees_with_drive_active():
    for state, rolled in (({}, True), ({"connected": False}, False),
                          ({"mode": pb.MODE_IDLE}, False),
                          ({"estop_latched": True, "estop_reason": "operator"}, False),
                          ({"estop_latched": True}, False),
                          ({"wheel_armed": {"left": False, "right": True}}, False),
                          ({"wheel_armed": {"left": False, "right": False}}, False)):
        robot = FakeRobot(**state)
        cause, _ = _stop_cause(robot, rolled)
        assert cause in STOP_CAUSES
        # anything but a roll is a state the recorder does not record in
        assert _drive_active(robot) == (cause == "rolled")


class ScriptedRobot:
    """A robot whose state steps through ``states`` (dicts of FakeState
    fields), one per read, then stays on the last."""

    def __init__(self, *states):
        self._states = [FakeState(**s) for s in states]
        self._i = 0

    @property
    def state(self):
        st = self._states[min(self._i, len(self._states) - 1)]
        self._i += 1
        return st


def _fake_time():
    now = [0.0]

    def sleep(s):
        now[0] += s
    return (lambda: now[0]), sleep


def _settle(robot, rolled=False):
    clock, sleep = _fake_time()
    return _settled_stop_cause(robot, rolled, settle_s=0.15, max_s=0.6,
                               poll_s=0.03, clock=clock, sleep=sleep)


def test_settle_waits_out_a_wheel_by_wheel_disarm():
    """disarm_all_wheels disarms one wheel at a time: the first frame reads
    one wheel off, which is the normal stop, not a wheel fault."""
    robot = ScriptedRobot({"wheel_armed": {"left": False, "right": True}},
                          {"wheel_armed": {"left": False, "right": False}})
    assert _settle(robot) == ("disarmed", "")


def test_settle_waits_out_set_mode_idle():
    """set_mode(Idle) disarms every wheel before it stores the mode."""
    off = {"left": False, "right": False}
    robot = ScriptedRobot({"wheel_armed": off}, {"wheel_armed": off},
                          {"wheel_armed": off, "mode": pb.MODE_IDLE})
    assert _settle(robot) == ("idle", "")


def test_settle_keeps_a_real_single_wheel_drop_and_reports_recovery():
    one = {"wheel_armed": {"left": True, "right": False}}
    assert _settle(ScriptedRobot(one)) == ("wheel_disarmed", "right")
    # a transient that recovers: no stop at all
    assert _settle(ScriptedRobot(one, {})) == (None, "")


def test_settle_gives_up_on_a_flapping_state():
    a = {"wheel_armed": {"left": True, "right": False}}
    b = {"wheel_armed": {"left": False, "right": True}}
    robot = ScriptedRobot(*([a, b] * 50))
    cause, _ = _settle(robot)
    assert cause == "wheel_disarmed"
