#!/usr/bin/env python3
"""Quantify a Bebop V2 standing-policy ROS2 MCAP capture.

Decodes /joint_states, /imu, /policy/action, /policy/observation, /policy/status
(CDR) and reports sanity, pipeline consistency, oscillation, balance, gains and
posture metrics for validating the learned standing policy.
"""
import sys
import numpy as np

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

JOINT_NAMES = [
    "hip_flexion_left", "hip_flexion_right",
    "hip_abduction_left", "hip_abduction_right",
    "knee_flexion_left", "knee_flexion_right",
    "foot_left", "foot_right",
]
SLEW = 0.020  # rad/tick


def quat_to_projgrav(x, y, z, w):
    g_x = -2.0 * (x * z - w * y)
    g_y = -2.0 * (y * z + w * x)
    g_z = -(1.0 - 2.0 * (x * x + y * y))
    return g_x, g_y, g_z


def load(path):
    rows = {k: [] for k in
            ["js_t", "js_pos", "js_vel", "js_eff", "js_names",
             "imu_t", "quat", "gyro",
             "act_t", "raw", "tgt", "kp", "kd",
             "obs_t", "obs",
             "status"]}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            t = message.log_time * 1e-9
            top = channel.topic
            if top == "/joint_states":
                rows["js_t"].append(t)
                rows["js_pos"].append(list(ros_msg.position))
                rows["js_vel"].append(list(ros_msg.velocity))
                rows["js_eff"].append(list(ros_msg.effort))
                if not rows["js_names"]:
                    rows["js_names"] = list(ros_msg.name)
            elif top == "/imu":
                rows["imu_t"].append(t)
                o = ros_msg.orientation
                rows["quat"].append([o.x, o.y, o.z, o.w])
                a = ros_msg.angular_velocity
                rows["gyro"].append([a.x, a.y, a.z])
            elif top == "/policy/action":
                rows["act_t"].append(t)
                rows["raw"].append(list(ros_msg.raw_action))
                rows["tgt"].append(list(ros_msg.position_targets_rad))
                rows["kp"].append(list(ros_msg.kp))
                rows["kd"].append(list(ros_msg.kd))
            elif top == "/policy/observation":
                rows["obs_t"].append(t)
                rows["obs"].append(list(ros_msg.data))
            elif top == "/policy/status":
                d = {}
                for fld in ("mode", "dry_run", "imu_live"):
                    if hasattr(ros_msg, fld):
                        d[fld] = getattr(ros_msg, fld)
                rows["status"].append(d)
    for k in rows:
        if k in ("js_names", "status"):
            continue
        rows[k] = np.array(rows[k], dtype=float) if rows[k] else np.array([])
    return rows


def st(label, a):
    a = np.asarray(a, dtype=float)
    return f"{label:22s} mean={a.mean():+.4f} std={a.std():.4f} min={a.min():+.4f} max={a.max():+.4f}"


def main(path):
    R = load(path)
    print("=" * 78)
    print("FILE:", path.split("/")[-1])
    print("=" * 78)

    # 1. Sanity / context
    if R["status"]:
        s0 = R["status"][0]
        print("\n[1] SANITY")
        print("  status sample:", s0)
        modes = set(d.get("mode") for d in R["status"])
        drys = set(d.get("dry_run") for d in R["status"])
        lives = set(d.get("imu_live") for d in R["status"])
        print(f"  modes={modes} dry_run={drys} imu_live={lives}")
    else:
        print("\n[1] SANITY: no /policy/status")

    for nm, key in [("joint_states", "js_t"), ("imu", "imu_t"),
                    ("action", "act_t"), ("observation", "obs_t")]:
        t = R[key]
        if len(t):
            dur = t[-1] - t[0]
            dt = np.diff(t)
            print(f"  {nm:12s} n={len(t):5d} dur={dur:6.2f}s "
                  f"dt_mean={dt.mean()*1000:6.2f}ms dt_std={dt.std()*1000:5.2f}ms")

    print("  joint order (capture):", R["js_names"])

    # 2. Pipeline consistency
    print("\n[2] PIPELINE CONSISTENCY")
    if len(R["raw"]) and len(R["tgt"]):
        raw_pos = np.clip(R["raw"][:, :8], -1, 1)
        decoded = 0.5 * raw_pos  # target - default ; default=0 nominal
        # position_targets are absolute (default + 0.5*clip). default nominal 0.
        resid = np.abs(R["tgt"] - decoded)
        print(f"  max|tgt - 0.5*clip(raw_pos)| = {resid.max():.5f} "
              f"(expect ~0 if default==0)")
    if len(R["obs"]):
        obs = R["obs"]
        print(f"  obs dim = {obs.shape[1]}")
        # layout: [0:3]ang_vel [3:6]projgrav [6:14]jpos_rel [14:22]jvel
        #         [22:46]last_action [46:49]cmd
        if obs.shape[1] >= 46 and len(R["raw"]) > 1:
            # obs[22:46] at tick i should equal raw_action at tick i-1
            n = min(len(obs), len(R["raw"]))
            la = obs[1:n, 22:46]
            prev = R["raw"][0:n - 1, :]
            d = np.abs(la - prev)
            print(f"  max|obs[22:46]_t - raw_{{t-1}}| = {d.max():.5f}")

    # 3. Oscillation
    print("\n[3] OSCILLATION (per joint)")
    vel = R["js_vel"]
    tgt = R["tgt"]
    print(f"  {'joint':20s} {'vel_std':>8s} {'vel_max':>8s} {'lag1ac':>7s} {'fft_hz':>7s}")
    js_dt = np.median(np.diff(R["js_t"])) if len(R["js_t"]) > 1 else 0.01
    for j in range(8):
        v = vel[:, j]
        vstd = v.std()
        vmax = np.abs(v).max()
        vc = v - v.mean()
        ac = (np.sum(vc[1:] * vc[:-1]) / np.sum(vc * vc)) if np.sum(vc * vc) > 0 else 0
        # dominant fft
        sp = np.abs(np.fft.rfft(vc))
        fr = np.fft.rfftfreq(len(vc), d=js_dt)
        fpk = fr[1 + np.argmax(sp[1:])] if len(sp) > 2 else 0
        print(f"  {JOINT_NAMES[j]:20s} {vstd:8.3f} {vmax:8.3f} {ac:+7.3f} {fpk:7.2f}")

    # slew exceedance on commanded targets
    if len(tgt) > 1:
        dtg = np.abs(np.diff(tgt, axis=0))
        print("\n  commanded |Δtarget/tick| (rad) and slew-exceedance fraction "
              f"(limit {SLEW}):")
        print(f"  {'joint':20s} {'mean':>8s} {'p95':>8s} {'frac>lim':>9s}")
        for j in range(8):
            d = dtg[:, j]
            frac = float(np.mean(d > SLEW))
            print(f"  {JOINT_NAMES[j]:20s} {d.mean():8.4f} "
                  f"{np.percentile(d,95):8.4f} {frac:9.3f}")
        alld = dtg.flatten()
        print(f"  ALL JOINTS slew-exceedance fraction = {np.mean(alld>SLEW):.3f}")

    # 3b. Torque (motor feedback; empty on captures predating effort logging)
    print("\n[3b] TORQUE (Nm, motor feedback in /joint_states.effort)")
    eff = R["js_eff"]
    if eff.ndim == 2 and eff.shape[1] == 8:
        print(f"  {'joint':20s} {'mean':>8s} {'std':>7s} {'p95|':>8s} {'max|':>8s}")
        for j in range(8):
            e = eff[:, j]
            print(f"  {JOINT_NAMES[j]:20s} {e.mean():+8.3f} {e.std():7.3f} "
                  f"{np.percentile(np.abs(e),95):8.3f} {np.abs(e).max():8.3f}")
        el, er = eff[:, 6], eff[:, 7]
        print(f"\n  ankles: |L| p95={np.percentile(np.abs(el),95):.2f} max={np.abs(el).max():.2f}  "
              f"|R| p95={np.percentile(np.abs(er),95):.2f} max={np.abs(er).max():.2f} Nm")
        print(f"  refs: old sim cap 6.0 Nm, firmware tau_max 17.0 Nm")
    else:
        print("  effort[] empty (capture predates torque logging)")

    # 4. Balance
    print("\n[4] BALANCE")
    if len(R["quat"]):
        q = R["quat"]
        gx, gy, gz = quat_to_projgrav(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
        print("  " + st("proj_grav g_x", gx))
        print("  " + st("proj_grav g_y", gy))
        print("  " + st("proj_grav g_z", gz))
        # quartiles
        print("  quartiles (g_x mean/std, g_z mean):")
        qsz = len(gx) // 4
        for i in range(4):
            sl = slice(i * qsz, (i + 1) * qsz if i < 3 else len(gx))
            print(f"    Q{i+1}: g_x mean={gx[sl].mean():+.4f} std={gx[sl].std():.4f}"
                  f"  g_z mean={gz[sl].mean():+.4f}")
    if len(R["gyro"]):
        g = R["gyro"]
        print(f"  base gyro xy std: wx={g[:,0].std():.4f} wy={g[:,1].std():.4f} rad/s")

    # 5. Gains
    print("\n[5] GAINS (decoded kp/kd per joint)")
    if len(R["kp"]):
        kp, kd = R["kp"], R["kd"]
        print(f"  {'joint':20s} {'kp_mean':>8s} {'kp_std':>7s} {'kp_min':>7s} {'kp_max':>7s}"
              f" | {'kd_mean':>8s} {'kd_std':>7s} {'kd_min':>7s} {'kd_max':>7s}")
        for j in range(8):
            print(f"  {JOINT_NAMES[j]:20s} "
                  f"{kp[:,j].mean():8.2f} {kp[:,j].std():7.2f} {kp[:,j].min():7.2f} {kp[:,j].max():7.2f}"
                  f" | {kd[:,j].mean():8.3f} {kd[:,j].std():7.3f} {kd[:,j].min():7.3f} {kd[:,j].max():7.3f}")

    # 6. Posture
    print("\n[6] POSTURE (joint_pos mean, rad)")
    pos = R["js_pos"]
    if len(pos):
        for j in range(8):
            print(f"  {JOINT_NAMES[j]:20s} mean={pos[:,j].mean():+.4f} "
                  f"std={pos[:,j].std():.4f} range=[{pos[:,j].min():+.3f},{pos[:,j].max():+.3f}]")
        # symmetry pairs
        print("\n  L/R symmetry (mean):")
        pairs = [("hip_flexion", 0, 1), ("hip_abduction", 2, 3),
                 ("knee_flexion", 4, 5), ("foot", 6, 7)]
        for nm, l, r in pairs:
            print(f"    {nm:14s} L={pos[:,l].mean():+.4f}  R={pos[:,r].mean():+.4f}"
                  f"  L+R={pos[:,l].mean()+pos[:,r].mean():+.4f}  L-R={pos[:,l].mean()-pos[:,r].mean():+.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
