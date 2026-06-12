//! Real-time scheduling + absolute-deadline timing helpers for the
//! 100 Hz control loop.
//!
//! The control loop in `main.rs` used to be a `tokio::time::interval`
//! task on the shared async worker pool. That timer is serviced by the
//! runtime's timer wheel and competes with the WS server / telemetry
//! pump for worker threads, so under any contention (or when the Jetson
//! governor leaves the loop's core at a low clock) the tick slips late —
//! we measured a 100 Hz loop running at ~89 Hz with ~20 ms wake gaps.
//!
//! A control loop wants the opposite of "best effort": a precise,
//! drift-free wake at a fixed period, scheduled ahead of background
//! work. This module provides the two POSIX primitives that buy us that
//! on a dedicated OS thread:
//!
//! 1. [`set_current_thread_fifo`] — raise the calling thread to
//!    `SCHED_FIFO` so the kernel preempts timeshare work to wake it.
//! 2. [`Deadline`] — an absolute `CLOCK_MONOTONIC` deadline advanced by
//!    a fixed period and slept on with `clock_nanosleep(TIMER_ABSTIME)`.
//!    Absolute deadlines don't accumulate drift the way "sleep for
//!    `period - work`" loops do, and `TIMER_ABSTIME` means a late wake
//!    doesn't shift all future deadlines.

use std::io;
use std::time::Duration;

const NANOS_PER_SEC: i64 = 1_000_000_000;

/// Raise the **calling** thread to `SCHED_FIFO` at `priority`.
///
/// `SCHED_FIFO` is a real-time policy: a runnable FIFO thread preempts
/// every `SCHED_OTHER` (timeshare) thread and runs until it blocks. Our
/// control loop blocks on `clock_nanosleep` every ~10 ms, so it yields
/// the core for the rest of the period — it does not starve the system.
///
/// `priority` must be in `1..=99`; a mid-range value (e.g. 80) sits
/// above typical kernel threads' default RT bands without crowding out
/// the highest-priority IRQ threads. Requires `CAP_SYS_NICE` (run as
/// root or grant the capability); returns the OS error otherwise so the
/// caller can fall back to timeshare scheduling with a warning rather
/// than aborting.
pub fn set_current_thread_fifo(priority: i32) -> io::Result<()> {
    // SAFETY: `sched_param` is a plain repr(C) struct; passing a pointer
    // to a local that outlives the call is sound. Thread id 0 targets
    // the calling thread.
    let param = libc::sched_param {
        sched_priority: priority,
    };
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// An absolute `CLOCK_MONOTONIC` wake target advanced by a fixed period.
///
/// Usage:
/// ```ignore
/// let mut d = Deadline::start(Duration::from_millis(10));
/// loop {
///     d.advance();        // next = prev + period (resync if we fell a full period behind)
///     d.sleep_until();    // block until the absolute deadline
///     // ... do tick work ...
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Deadline {
    period_ns: i64,
    next: libc::timespec,
}

impl Deadline {
    /// Create a deadline anchored at "now", with the given tick period.
    pub fn start(period: Duration) -> Self {
        Self {
            period_ns: period.as_nanos() as i64,
            next: now_monotonic(),
        }
    }

    /// Advance the deadline by one period.
    ///
    /// If the new target is already in the past (the loop fell at least a
    /// full period behind — e.g. a long stall), resync to `now + period`
    /// instead of letting the deadline chase a backlog. This mirrors the
    /// old `MissedTickBehavior::Delay`: a missed tick is dropped rather
    /// than replayed as a burst of back-to-back ticks (which would dump a
    /// flurry of motor commands after a hiccup).
    pub fn advance(&mut self) {
        add_nanos(&mut self.next, self.period_ns);
        let now = now_monotonic();
        if timespec_lt(&self.next, &now) {
            self.next = now;
            add_nanos(&mut self.next, self.period_ns);
        }
    }

    /// Block until the absolute deadline. Restarts on `EINTR`.
    pub fn sleep_until(&self) {
        loop {
            // SAFETY: `next` is a valid initialized timespec; null remainder
            // is allowed for TIMER_ABSTIME (the kernel ignores it).
            let rc = unsafe {
                libc::clock_nanosleep(
                    libc::CLOCK_MONOTONIC,
                    libc::TIMER_ABSTIME,
                    &self.next,
                    std::ptr::null_mut(),
                )
            };
            // clock_nanosleep returns the error number directly (0 on
            // success) rather than setting errno. Only EINTR warrants a
            // retry; any other error (shouldn't happen for a valid abs
            // monotonic deadline) we treat as "deadline reached".
            if rc != libc::EINTR {
                break;
            }
        }
    }
}

/// Read `CLOCK_MONOTONIC` into a `timespec`.
fn now_monotonic() -> libc::timespec {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: writing through a pointer to a local, valid timespec.
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    ts
}

/// Add `nanos` (>= 0) to `ts`, normalizing the carry into `tv_sec`.
/// `tv_sec`/`tv_nsec` are `c_long`/`time_t` (i64 on our aarch64 target).
fn add_nanos(ts: &mut libc::timespec, nanos: i64) {
    let total = ts.tv_nsec + nanos;
    ts.tv_sec += (total / NANOS_PER_SEC) as libc::time_t;
    ts.tv_nsec = total % NANOS_PER_SEC;
}

/// `a < b` for two normalized timespecs.
fn timespec_lt(a: &libc::timespec, b: &libc::timespec) -> bool {
    (a.tv_sec, a.tv_nsec) < (b.tv_sec, b.tv_nsec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_nanos_carries_into_seconds() {
        let mut ts = libc::timespec {
            tv_sec: 5,
            tv_nsec: 800_000_000,
        };
        add_nanos(&mut ts, 300_000_000); // 0.3 s -> rolls over a second
        assert_eq!(ts.tv_sec, 6);
        assert_eq!(ts.tv_nsec, 100_000_000);
    }

    #[test]
    fn add_nanos_multi_second_carry() {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        add_nanos(&mut ts, 2_500_000_000); // 2.5 s
        assert_eq!(ts.tv_sec, 2);
        assert_eq!(ts.tv_nsec, 500_000_000);
    }

    #[test]
    fn timespec_lt_orders_by_sec_then_nsec() {
        let a = libc::timespec {
            tv_sec: 1,
            tv_nsec: 999_999_999,
        };
        let b = libc::timespec {
            tv_sec: 2,
            tv_nsec: 0,
        };
        assert!(timespec_lt(&a, &b));
        assert!(!timespec_lt(&b, &a));
        assert!(!timespec_lt(&a, &a));
    }

    #[test]
    fn deadline_advance_is_monotonic_and_period_spaced_when_keeping_up() {
        // Two advances with no intervening sleep: since "now" hasn't moved
        // a full period, the deadline should step by exactly one period
        // each time (no resync), proving we don't drift.
        let mut d = Deadline::start(Duration::from_millis(10));
        let t0 = d.next;
        d.advance();
        let t1 = d.next;
        let step = (t1.tv_sec - t0.tv_sec) * NANOS_PER_SEC + (t1.tv_nsec - t0.tv_nsec);
        assert_eq!(step, 10_000_000, "first advance steps exactly one period");
    }

    #[test]
    fn deadline_sleep_until_blocks_about_one_period() {
        // Integration-ish: a single short sleep should take roughly the
        // period (allow generous slack for CI timeshare scheduling).
        let mut d = Deadline::start(Duration::from_millis(20));
        let start = std::time::Instant::now();
        d.advance();
        d.sleep_until();
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(15),
            "slept too little: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(500),
            "slept way too long: {elapsed:?}"
        );
    }
}
