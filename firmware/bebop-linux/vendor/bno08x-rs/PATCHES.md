# bno08x-rs patches (Bebop fork of 2.0.1)

This directory is a **fork** of [bno08x-rs 2.0.1] (`crates.io`,
`EdgeFirstAI/bno08x-rs`). Every Bebop-specific change is annotated
inline with a `BEBOP-PATCH:` comment so a future re-vendor can be done
mechanically (copy upstream over the directory, then re-apply the
hunks below).

We carry this fork because the production `firmware/bebop-linux`
runtime needs SH-2 sensor report **0x28** ("AR/VR-Stabilized Rotation
Vector"). This report is the EMI-hardened replacement for the
magnetometer-fused 0x05 report — the old Teensy firmware in
`firmware/bebop-locomotion/include/BNO085_IMU.h` already uses it for
exactly this reason (motor currents on a humanoid biases the BNO's
magnetometer enough that 0x05's fusion filter rejects all subsequent
mag updates, freezing yaw indefinitely until a hard reset).

The upstream crate physically cannot enable 0x28: it indexes a
`[bool; 16]` array with the raw report ID, which panics for any
ID ≥ 16. There is also no parser, no accessor, and no Q-point entry
for the AR/VR-Stabilized reports anywhere in the upstream code.

## The patches

Each patch carries a `BEBOP-PATCH [N/4]` marker in the source; the
table below maps them back to the underlying issue.

### `[1/4]`  Bump report tracking arrays from `[_; 16]` → `[_; 256]`

* `src/driver.rs`, the `BNO08x` struct (around the `report_enabled` /
  `report_update_time` / `report_update_callbacks` fields), and the
  matching `new_with_interface` initializer.

These three arrays are indexed directly by raw SH-2 report ID in
several hot paths (e.g. `handle_sensor_report_update`,
`enable_report`). Size-16 was a hard limit: any report ID ≥ 16 caused
a runtime panic on the first SHTP message we received that mentioned
it. Bumping to 256 covers the full `u8` ID space.

Memory cost is ~10 KB of HashMaps and 2 KB of `u128`s, which is
negligible for a userspace driver. No public API change.

**Upstream-as-a-PR plausibility:** high. This is a straight bug fix —
any user enabling, say, the GyroRV (0x2A) or Step Counter (0x11)
report from the SH-2 spec hits the same panic.

### `[2/4]`  Add SH-2 reports 0x28 / 0x29 (AR/VR-Stabilized RV / Game RV)

Spread across:

* `src/constants.rs` — new `SENSOR_REPORTID_ARVR_STABILIZED_RV`
  (0x28) and `SENSOR_REPORTID_ARVR_STABILIZED_GAME_RV` (0x29)
  constants. The `Q_POINTS` and `Q_POINTS2` arrays are regenerated as
  `[usize; 256]` via a `const fn build_q_table` helper, with the
  proper Q14 / Q12 entries for the new reports.
* `src/driver.rs` — `handle_sensor_reports` now has two new match
  arms that route 0x28 / 0x29 packets into
  `update_rotation_quaternion_arvr` / `update_rotation_quaternion_arvr_game`,
  which are wire-format-identical to `update_rotation_quaternion`
  / `update_rotation_quaternion_game` (the upstream's plain-RV /
  Game-RV decoders) but write into separate cache fields so the
  caller's choice of report doesn't get clobbered.
* `src/driver.rs` — three new public accessors on `BNO08x`:
  `arvr_stabilized_rotation_quaternion()`,
  `arvr_stabilized_rotation_acc()`,
  `arvr_stabilized_game_rotation_quaternion()`.
* `src/lib.rs` — re-exports the two new constants from the crate
  root for convenience.

Wire formats are taken from CEVA's SH-2 Reference Manual §6.5.18
("AR/VR-Stabilized Rotation Vector") and §6.5.19 ("AR/VR-Stabilized
Game Rotation Vector"). 0x28 carries five `i16`s (quaternion +
accuracy), 0x29 four (quaternion only) — identical to 0x05 and 0x08
respectively.

**Upstream-as-a-PR plausibility:** high. These are first-class SH-2
reports defined in the public spec, and the wire format is a near
copy of two reports the crate already supports. Bundling this with
patch `[1/4]` in the same PR makes sense — they're both prerequisites
for `enable_report(0x28, …)` to actually work end to end.

### `[3/4]`  Add `+ Send` to the report-callback trait object

* `src/driver.rs` — `type ReportCallbackMap<'a, SI> = HashMap<…, Box<dyn Fn(…) + Send + 'a>>;`
* `src/driver.rs` — `add_sensor_report_callback`'s `func: impl Fn(&Self) + Send + 'a` parameter bound.
* `src/reports.rs` — `pub type ReportCallback<'a, T> = Box<dyn Fn(&T) + Send + 'a>;`

The `BNO08x` struct contains a `HashMap<String, Box<dyn Fn(&Self)>>`
per report ID. Because the upstream trait object isn't `Send`, the
whole struct is non-`Send`, which means it cannot be moved into a
`std::thread::spawn` closure. Bebop's `firmware/bebop-linux/src/imu.rs`
does exactly that — the IMU reader runs on its own OS thread so a
busy SHTP bus never stalls the runtime's other I/O — so we tighten the
bound.

This is the kind of bound that, once relaxed, can never be tightened
back without an API break, which is presumably why the upstream
defaults to the more permissive (non-`Send`) form. For a
single-threaded user the cost is zero; for a multi-threaded user the
cost was "you can't use this crate".

**Upstream-as-a-PR plausibility:** medium. It's a strictly tightening
change to a public type alias and a public method's bound. Some
existing downstreams that capture `Rc<…>` / `RefCell<…>` in their
callbacks would have to re-architect. A reasonable upstream patch
would split the trait alias into `ReportCallback` (Send) and
`LocalReportCallback` (no Send) — but that's a bigger refactor than
just bolting on `+ Send`. We took the simplest path for the fork.

### `[4/4]`  Test fallout from patch `[3/4]`

* `src/driver.rs::tests::test_add_sensor_report_callback`
* `src/reports.rs::tests::test_report_state_callback_management`

Both upstream tests captured an `Rc<Cell<bool>>` to assert "the
callback fired". `Rc<Cell<_>>` is not `Send`, so they no longer
satisfy the tightened trait bound from patch `[3/4]`. Swapped both
for `Arc<AtomicBool>`, which preserves the intent (race-free
"callback fired" flag) under the new bound.

This patch is purely a test-suite repair and would land in the same
upstream PR as `[3/4]` — different rationale for a reviewer, same
file change.

## Re-vendoring procedure

When `bno08x-rs` ships a 2.0.2 / 2.1.0 / etc. that we want to pull in:

```bash
# 1. Pull the new release into Cargo's registry cache.
cargo update -p bno08x-rs --precise <new-version>

# 2. Mirror the new source over this directory (deletes Bebop changes!).
rm -rf firmware/bebop-linux/vendor/bno08x-rs
cp -r ~/.cargo/registry/src/index.crates.io-*/bno08x-rs-<new-version> \
      firmware/bebop-linux/vendor/bno08x-rs
rm firmware/bebop-linux/vendor/bno08x-rs/.cargo-ok \
   firmware/bebop-linux/vendor/bno08x-rs/.cargo_vcs_info.json
mv firmware/bebop-linux/vendor/bno08x-rs/Cargo.toml.orig \
   firmware/bebop-linux/vendor/bno08x-rs/Cargo.toml

# 3. Re-apply the four hunks above (grep history for `BEBOP-PATCH` in
#    the previous tree to see the exact edits).

# 4. Rerun the smoke tests:
( cd firmware/bebop-linux/vendor/bno08x-rs && cargo test --lib )
( cd firmware/bebop-linux && cargo test --lib --bins )
( cd firmware/bebop-linux && cargo run --bin imu-probe -- \
      --spi /dev/spidev0.0 \
      --int-chip gpiochip0 --int-line 144 \
      --rst-chip gpiochip0 --rst-line 106 \
      --report-id 0x28 \
      --period-ms 50 --duration-s 10 )

# 5. If the upstream has merged any of our patches in the meantime,
#    delete the matching `BEBOP-PATCH` markers (and this section of
#    PATCHES.md) instead of re-applying.
```

## Upstream tracking

Once we have the bandwidth to open the upstream PRs, fill in the
issue / PR URLs here so the patch markers can be deleted in the
matching order.

| Patch | Status            | Upstream issue / PR |
|-------|-------------------|---------------------|
| `[1/4]` array-size bump | Not filed yet     | — |
| `[2/4]` 0x28 / 0x29 add | Not filed yet     | — |
| `[3/4]` `+ Send` bound  | Not filed yet     | — |
| `[4/4]` test repair     | Not filed yet     | (bundled with [3/4]) |

[bno08x-rs 2.0.1]: https://crates.io/crates/bno08x-rs/2.0.1
