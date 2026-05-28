// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

extern crate spidev;

use log::trace;
use spidev::{SpiModeFlags, Spidev, SpidevOptions, SpidevTransfer};

use std::{io, path::Path, vec};
/// Blocking transfer
pub trait Transfer {
    /// Error type
    type Error;

    /// Sends `words` to the slave. Returns the `words` received from the slave
    fn transfer<'a>(&'a mut self, words: &'a mut [u8]) -> Result<&'a [u8], Self::Error>;
}

/// Blocking write
pub trait Write {
    /// Error type
    type Error;

    /// Sends `words` to the slave, ignoring all the incoming words
    fn write(&mut self, words: &[u8]) -> Result<(), Self::Error>;
}

/// Default SPI clock used by [`SpiDevice::new`] when no explicit speed is
/// given. Kept at the upstream-conservative 80 kHz so the symbol-lookup
/// constructor, examples, and hardware tests behave exactly as before.
/// The Bebop runtime instead passes an explicit (higher) speed via
/// [`SpiDevice::new_with_speed`] so it can sustain higher report rates.
pub const DEFAULT_SPI_MAX_SPEED_HZ: u32 = 80_000;

pub struct SpiDevice {
    spi: Spidev,
}
impl SpiDevice {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<SpiDevice> {
        Self::new_with_speed(path, DEFAULT_SPI_MAX_SPEED_HZ)
    }

    /// Open the spidev with an explicit max clock (Hz).
    ///
    /// The BNO08x SPI slave is rated to 3 MHz; 80 kHz (the historical
    /// default) is far below that and cannot carry high-rate report
    /// streams — e.g. a 200 Hz rotation-vector + 200 Hz gyro pair needs
    /// ~hundreds of kbit/s once SHTP framing is included. Raise this to
    /// 1 MHz+ for high report rates, but keep margin on long / unshielded
    /// jumper wiring where signal integrity degrades at higher clocks.
    pub fn new_with_speed<P: AsRef<Path>>(path: P, max_speed_hz: u32) -> io::Result<SpiDevice> {
        let mut spi = Spidev::open(path)?;
        let options = SpidevOptions::new()
            .bits_per_word(8)
            .max_speed_hz(max_speed_hz)
            .mode(SpiModeFlags::SPI_MODE_3)
            .lsb_first(false)
            .build();
        spi.configure(&options)?;

        Ok(SpiDevice { spi })
    }
}

impl Transfer for SpiDevice {
    type Error = io::Error;

    fn transfer<'a>(&'a mut self, words: &'a mut [u8]) -> Result<&'a [u8], Self::Error> {
        let mut rx_buf = vec![0_u8; words.len()];
        let buf = rx_buf.as_mut();
        trace!("Transfer write: {:?}", words);
        let mut transfer = SpidevTransfer::read_write(words, buf);
        self.spi.transfer(&mut transfer)?;
        words.clone_from_slice(buf);
        trace!("Transfer read: {:?}", words);
        Ok(words)
    }
}

impl Write for SpiDevice {
    type Error = io::Error;

    fn write(&mut self, words: &[u8]) -> Result<(), Self::Error> {
        trace!("Write: {:?}", words);
        let mut rx_buf = vec![0_u8; words.len()];
        let buf = rx_buf.as_mut();
        let mut transfer = SpidevTransfer::read_write(words, buf);
        self.spi.transfer(&mut transfer)?;
        trace!("Write read: {:?}", buf);
        Ok(())
    }
}
