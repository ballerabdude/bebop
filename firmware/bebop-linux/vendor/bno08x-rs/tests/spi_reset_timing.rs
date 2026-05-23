//! Regression guard for BEBOP-PATCH [6/6] SPI reset timing in
//! `interface/spi.rs::setup`. A future upstream re-vendor must not
//! shrink these values back to the broken upstream defaults (2 ms RST
//! hold, 200 ms HINTN wait, no pre/post settle).

use bno08x_rs::interface::spi::{
    SPI_SETUP_HINTN_WAIT_MS, SPI_SETUP_POST_AWAKE_SETTLE_MS, SPI_SETUP_PRE_RESET_DRAIN_MS,
    SPI_SETUP_RST_LOW_HOLD_MS,
};

/// Upstream `setup()` used 2 ms RST low — must stay at Hillcrest 10 ms.
const UPSTREAM_RST_LOW_HOLD_MS: usize = 2;
/// Upstream `wait_for_sensor_awake(200)` — must stay at 500 ms.
const UPSTREAM_HINTN_WAIT_MS: usize = 200;

#[test]
fn spi_setup_pre_reset_drain_at_least_50ms() {
    assert!(
        SPI_SETUP_PRE_RESET_DRAIN_MS >= 50,
        "pre-reset drain was {} ms (need >= 50 for stale SHTP flush)",
        SPI_SETUP_PRE_RESET_DRAIN_MS
    );
}

#[test]
fn spi_setup_rst_low_hold_at_least_10ms() {
    assert!(
        SPI_SETUP_RST_LOW_HOLD_MS >= 10,
        "RST low hold was {} ms (upstream {} ms; Hillcrest RESET_DELAY is 10 ms)",
        SPI_SETUP_RST_LOW_HOLD_MS,
        UPSTREAM_RST_LOW_HOLD_MS
    );
}

#[test]
fn spi_setup_hintn_wait_at_least_500ms() {
    assert!(
        SPI_SETUP_HINTN_WAIT_MS >= 500,
        "HINTN wait was {} ms (upstream {} ms; warm restart needs >= 500)",
        SPI_SETUP_HINTN_WAIT_MS,
        UPSTREAM_HINTN_WAIT_MS
    );
}

#[test]
fn spi_setup_post_awake_settle_at_least_50ms() {
    assert!(
        SPI_SETUP_POST_AWAKE_SETTLE_MS >= 50,
        "post-awake settle was {} ms (need >= 50 to avoid product-ID race)",
        SPI_SETUP_POST_AWAKE_SETTLE_MS
    );
}
