//! Standalone BNO080/BNO085 SPI smoke test.
//!
//! Run this whenever you bring up an IMU on a new robot — it validates
//! wiring + SPI-mode jumpers + INT/RST GPIO assignments end-to-end,
//! without touching anything else in the runtime. The bin pulls in the
//! vendored [`bno08x_rs`] (`firmware/bebop-linux/vendor/bno08x-rs`) and
//! prints quaternions until a deadline; the exit code tells you which
//! physical layer was wrong if anything fails.
//!
//! Note: this probe deliberately runs the SHTP boot handshake (`init`)
//! and the `SET_FEATURE` for the requested report **once**, with no
//! retry. The production `imu.rs::spawn_imu_thread` path retries the
//! same sequence up to four times with a 250 ms backoff, because in
//! production we'd rather absorb a transient handshake glitch than
//! refuse to start the IMU thread for the rest of the run. The
//! diagnostic value of this probe depends on it failing loudly and
//! immediately on real wiring/jumper problems instead of papering
//! over them with retries — so once `imu-probe` passes the production
//! path is *at least as likely* to come up cleanly, but a single
//! `imu-probe` failure does not on its own prove the production path
//! is broken (re-run it a couple of times before suspecting hardware).
//!
//! # Example (Bebop V2 reference wiring; see `config/bebop_v2.yaml`)
//!
//! On a Jetson Orin Nano dev kit:
//!
//! ```text
//! cargo run --bin imu-probe -- \
//!     --spi /dev/spidev0.0 \
//!     --int-chip gpiochip0 --int-line 144 \
//!     --rst-chip gpiochip0 --rst-line 106 \
//!     --report-id 0x28 \
//!     --period-ms 50 \
//!     --duration-s 10
//! ```
//!
//! Header pin → gpiochip line for Orin Nano (cross-check with
//! `sudo gpioinfo gpiochip0`): pin 7 → 144 (PAC.06, used as INT),
//! pin 31 → 106 (PQ.06, used as RST). All 40-pin-header GPIOs live on
//! `gpiochip0`; `gpiochip1` (the AON controller) is NOT routed to the
//! header.
//!
//! # Exit codes
//!
//! | Code | Meaning                                                |
//! |------|--------------------------------------------------------|
//! | 0    | Got at least one valid quaternion                      |
//! | 1    | Failed to open the SPI device or GPIO lines            |
//! | 2    | `init()` failed — chip didn't ACK the SHTP advertisement (most likely the SPI-mode jumpers aren't bridged, or RST/INT are wired to the wrong GPIOs) |
//! | 3    | `enable_report()` failed — chip is alive but won't take the SET_FEATURE command |
//! | 4    | Init/enable succeeded but no quaternion arrived within the deadline (check the report period, or that the SPI clock isn't being clobbered by motor harness noise) |

use std::process::ExitCode;
use std::time::{Duration, Instant};

use bno08x_rs::BNO08x;
use clap::Parser;

/// CLI arguments for the BNO08x SPI smoke test.
#[derive(Debug, Parser)]
#[command(
    version,
    about = "BNO080/BNO085 SPI smoke test (bypasses the production imu.rs path)"
)]
struct Args {
    /// SPI character device. `spi1` in jetson-io maps to `/dev/spidev0.0`
    /// on Jetson Orin Nano.
    #[arg(long, default_value = "/dev/spidev0.0")]
    spi: String,

    /// GPIO chip hosting the BNO `INT` (HINTN) line — usually `gpiochip1`
    /// on Orin Nano because header pin 7 (`PBB.00`) lives on the AON
    /// controller.
    #[arg(long)]
    int_chip: String,

    /// Line offset within `--int-chip` for `INT`.
    #[arg(long)]
    int_line: u32,

    /// GPIO chip hosting the BNO `RST` (RSTN) line.
    #[arg(long)]
    rst_chip: String,

    /// Line offset within `--rst-chip` for `RST`.
    #[arg(long)]
    rst_line: u32,

    /// SH-2 sensor report ID to enable. The default `0x28` is the
    /// **AR/VR-stabilized rotation vector** — the same report the
    /// previous Teensy firmware enabled in
    /// `firmware/bebop-locomotion/include/BNO085_IMU.h`. Common values:
    ///
    /// | Hex   | Name                              | Magnetometer? |
    /// |-------|-----------------------------------|---------------|
    /// | 0x05  | Rotation Vector                   | yes (locks near motors) |
    /// | 0x08  | Game Rotation Vector              | no |
    /// | 0x28  | AR/VR Stabilized Rotation Vector  | yes, EMI-hardened |
    /// | 0x29  | AR/VR Stabilized Game Rotation V. | no |
    #[arg(long, value_parser = parse_u8_maybe_hex, default_value = "0x28")]
    report_id: u8,

    /// Report cadence hint sent to the chip in milliseconds. Lower =
    /// more samples/sec. Bounded by the chip's gyro rate (1 kHz). Anything
    /// >= 5 ms is comfortably safe.
    #[arg(long, default_value_t = 50)]
    period_ms: u16,

    /// How long to stream before exiting.
    #[arg(long, default_value_t = 10)]
    duration_s: u64,

    /// Print every Nth sample instead of every sample. The default
    /// throttles to ~4 lines/sec so the terminal stays readable while
    /// still proving fresh data is arriving.
    #[arg(long, default_value_t = 250)]
    print_every_ms: u64,
}

/// Parse a decimal or hex (`0x`-prefixed) `u8` for clap.
fn parse_u8_maybe_hex(s: &str) -> Result<u8, std::num::ParseIntError> {
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u8::from_str_radix(rest, 16)
    } else {
        s.parse::<u8>()
    }
}

fn main() -> ExitCode {
    let args = Args::parse();

    println!(
        "imu-probe: spi={} INT={}:{} RST={}:{} report=0x{:02x} period={}ms duration={}s",
        args.spi,
        args.int_chip,
        args.int_line,
        args.rst_chip,
        args.rst_line,
        args.report_id,
        args.period_ms,
        args.duration_s,
    );

    // ---------------------------------------------------------------------
    // Stage 1: open SPI + GPIOs. Failure here means wiring/permissions,
    // not anything BNO-specific.
    // ---------------------------------------------------------------------
    let mut imu = match BNO08x::new_spi(
        &args.spi,
        &args.int_chip,
        args.int_line,
        &args.rst_chip,
        args.rst_line,
    ) {
        Ok(imu) => imu,
        Err(e) => {
            eprintln!(
                "FAIL [stage 1: open]  could not open SPI/GPIO: {e:?}\n\
                 hint: check `ls -l /dev/spidev*` and `sudo gpioinfo {} {}`.\n\
                 also try running with `sudo` — spidev needs `gpio` group access.",
                args.int_chip, args.rst_chip,
            );
            return ExitCode::from(1);
        }
    };
    println!("✓ stage 1 — SPI device + INT/RST GPIO lines opened");

    // ---------------------------------------------------------------------
    // Stage 2: SHTP init handshake. Toggles RST, reads the advertisement
    // and product-ID frames, verifies the chip identifies as BNO08x.
    // Failure here almost always means the chip is in I²C/UART mode (the
    // PS0/PS1 solder jumpers on the back of the Adafruit board aren't
    // bridged) or RST isn't wired to the GPIO we think it is.
    // ---------------------------------------------------------------------
    if let Err(e) = imu.init() {
        eprintln!(
            "FAIL [stage 2: init]  SHTP handshake failed: {e:?}\n\
             hint: did you bridge BOTH SPI-enable jumpers on the back of the BNO board?\n\
             also verify RST is wired to header pin 31 and INT to header pin 7."
        );
        return ExitCode::from(2);
    }
    println!("✓ stage 2 — chip booted, SHTP advertisement + product ID parsed");

    // ---------------------------------------------------------------------
    // Stage 3: enable the requested sensor report. Most callers pick
    // 0x28 (AR/VR-stabilized RV) to match the previous C++ firmware.
    // ---------------------------------------------------------------------
    // NOTE: `bno08x-rs::enable_report` returns `Ok(false)` if it never sees
    // a `GET_FEATURE_RESP` confirming the new subscription. Per the CEVA
    // SH-2 spec the chip is supposed to auto-send that response after
    // `SET_FEATURE`, but on some BNO085 firmware revisions (and/or due to a
    // crate-side SHTP demux bug) it doesn't surface here. The SET_FEATURE
    // command itself still goes out on the wire, so input reports may
    // start flowing regardless. To diagnose, we treat `Ok(false)` as a
    // soft warning and let stage 4 be the source of truth.
    match imu.enable_report(args.report_id, args.period_ms) {
        Ok(true) => println!(
            "✓ stage 3 — report 0x{:02x} accepted (period {} ms)",
            args.report_id, args.period_ms
        ),
        Ok(false) => {
            eprintln!(
                "WARN [stage 3: enable]  no GET_FEATURE_RESP for 0x{:02x} \
                 within 2 s; proceeding to stage 4 to see whether the \
                 chip is actually streaming or actually refusing.",
                args.report_id
            );
        }
        Err(e) => {
            eprintln!("FAIL [stage 3: enable]  SET_FEATURE error: {e:?}");
            return ExitCode::from(3);
        }
    }

    // ---------------------------------------------------------------------
    // Stage 4: stream samples until the deadline.
    // ---------------------------------------------------------------------
    let deadline = Instant::now() + Duration::from_secs(args.duration_s);
    let print_period = Duration::from_millis(args.print_every_ms.max(1));
    let mut last_print = Instant::now()
        .checked_sub(print_period)
        .unwrap_or_else(Instant::now);
    let mut samples: u64 = 0;

    // `bno08x-rs` parses each rotation-vector flavour into its own
    // field (so 0x08 lands in `game_rotation_quaternion`, not the 0x05
    // slot read by `rotation_quaternion()`). Validate the report ID
    // up-front so stage 4 only runs for reports we actually know how
    // to read. 0x28 / 0x29 require the local vendored crate
    // (`firmware/bebop-linux/vendor/bno08x-rs`) — the upstream crate
    // would have panicked in stage 3 before getting here.
    if !matches!(args.report_id, 0x05 | 0x08 | 0x09 | 0x28 | 0x29) {
        eprintln!(
            "FAIL [stage 4: stream]  report 0x{:02x} has no quaternion \
             accessor available. Use one of: 0x05 (Rotation Vector), \
             0x08 (Game RV), 0x09 (Geomag RV), 0x28 (AR/VR-Stabilized \
             RV — production default), 0x29 (AR/VR-Stabilized Game RV).",
            args.report_id
        );
        return ExitCode::from(4);
    }
    // Heading accuracy is only meaningful for the magnetometer-fused
    // reports — 0x05, 0x09, and 0x28. Pure game-rotation variants
    // (0x08, 0x29) don't report it.
    let report_has_heading_acc = matches!(args.report_id, 0x05 | 0x09 | 0x28);

    while Instant::now() < deadline {
        // Pump the SHTP read path; the crate decodes input reports and
        // updates its internal cache. We pass a short per-message timeout
        // so a quiet bus doesn't stall the loop.
        let _ = imu.handle_all_messages(25);
        let quat = match args.report_id {
            0x05 => imu.rotation_quaternion().ok(),
            0x08 => imu.game_rotation_quaternion().ok(),
            0x09 => imu.geomag_rotation_quaternion().ok(),
            0x28 => imu.arvr_stabilized_rotation_quaternion().ok(),
            0x29 => imu.arvr_stabilized_game_rotation_quaternion().ok(),
            _ => unreachable!("guarded above"),
        };
        if let Some([qx, qy, qz, qw]) = quat {
            samples += 1;
            if last_print.elapsed() >= print_period {
                let norm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
                let norm = norm_sq.sqrt();
                if report_has_heading_acc {
                    // 0x05 / 0x09 store their accuracy in
                    // `rotation_acc()`, 0x28 in
                    // `arvr_stabilized_rotation_acc()`. Pick the
                    // matching one so the printed `heading_acc` is the
                    // real number for *this* report, not a stale 0
                    // from a different cache.
                    let acc_rad = match args.report_id {
                        0x28 => imu.arvr_stabilized_rotation_acc(),
                        _ => imu.rotation_acc(),
                    };
                    let acc_deg = acc_rad.to_degrees();
                    println!(
                        "  #{samples:>6}  q=[{qx:+.3}, {qy:+.3}, {qz:+.3}, {qw:+.3}]  |q|={norm:.4}  heading_acc=±{acc_deg:.1}°",
                    );
                } else {
                    println!(
                        "  #{samples:>6}  q=[{qx:+.3}, {qy:+.3}, {qz:+.3}, {qw:+.3}]  |q|={norm:.4}",
                    );
                }
                last_print = Instant::now();
            }
        }
        // Lighter spin than the production loop — we're not under any
        // policy-rate pressure here, just smoke-testing.
        std::thread::sleep(Duration::from_millis(2));
    }

    let hz = samples as f64 / args.duration_s as f64;
    println!(
        "\nstage 4 — captured {} samples in {} s ({:.1} Hz)",
        samples, args.duration_s, hz
    );

    // ---------------------------------------------------------------------
    // Graceful disable.
    //
    // Critical for the production runtime's startup reliability: if we
    // exit here with a live subscription, the chip keeps streaming on
    // the data channel into nobody. The next time `bebop-linux`
    // (or this probe) opens the bus it sees those stale reports collide
    // with the SHTP control channel during `verify_product_id`, and
    // burns through the bring-up retry loop in `imu.rs` before settling.
    //
    // Period = 0 µs is the SH-2 spec way of disabling a report. We do
    // NOT also `soft_reset()` here — see the equivalent comment in
    // `imu.rs::spawn_imu_thread`'s graceful-disable epilogue for the
    // detailed rationale; short version is "redundant with the next
    // bring-up's RST GPIO pulse, and the chip's advert reply was
    // tripping an upstream `bno08x-rs` parser bug (now fixed by
    // BEBOP-PATCH [5/5])".
    let _ = imu.enable_report(args.report_id, 0);

    if samples == 0 {
        eprintln!(
            "FAIL [stage 4: stream]  zero samples received.\n\
             hint: SPI is clocking and the chip ACKed enable, but no input reports arrived.\n\
             check the INT line — is it actually connected, and falling-edge when data is ready?"
        );
        ExitCode::from(4)
    } else {
        println!("PASS — IMU is alive on SPI. Yaw should now respond to rotations.");
        ExitCode::SUCCESS
    }
}
