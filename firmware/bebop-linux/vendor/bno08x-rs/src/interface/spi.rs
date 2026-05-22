// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use log::{error, trace};

use super::SensorInterface;
use crate::{
    interface::{
        delay::delay_ms,
        gpio::{InputPin, OutputPin},
        spidev::{Transfer, Write},
        SensorCommon, PACKET_HEADER_LENGTH,
    },
    Error,
    Error::{BufferOverflow, NoDataAvailable, SensorUnresponsive},
};
use std::fmt::Debug;

// BEBOP-PATCH [6/6]: minimum SPI reset-sequence timing (ms). Used by
// `setup()` and regression-tested in `tests/spi_reset_timing.rs`.
pub const SPI_SETUP_PRE_RESET_DRAIN_MS: usize = 50;
pub const SPI_SETUP_RST_LOW_HOLD_MS: usize = 10;
pub const SPI_SETUP_HINTN_WAIT_MS: usize = 500;
pub const SPI_SETUP_POST_AWAKE_SETTLE_MS: usize = 50;

/// Encapsulates all the lines required to operate this sensor
/// - SCK: clock line from master
/// - MISO: Data input from the sensor to the master
/// - MOSI: Output from the master to the sensor
/// - CSN: chip select line that selects the device on the shared SPI bus
/// - HINTN: Hardware Interrupt. Sensor uses this to indicate it had data
///   available for read
/// - RSTN: Reset the device
pub struct SpiControlLines<SPI, /* CSN, */ IN, RSTN> {
    pub spi: SPI, // the spidev read/write
    // pub csn: CSN,    // chip select pin, SPI_CS
    pub hintn: IN,   // interrupt, IMU_INT
    pub reset: RSTN, // reset, IMU_RST
}

/// This combines the SPI peripheral and associated control pins
pub struct SpiInterface<SPI, /* CSN, */ IN, RSTN> {
    spi: SPI,
    // csn: CSN,
    hintn: IN,
    reset: RSTN,
    received_packet_count: usize,
}

impl<SPI, /* CSN, */ IN, RSTN, CommE, PinE> SpiInterface<SPI, /* CSN, */ IN, RSTN>
where
    SPI: Write<Error = CommE> + Transfer<Error = CommE>,
    // CSN: OutputPin<Error = PinE>,
    IN: InputPin<Error = PinE>,
    RSTN: OutputPin<Error = PinE>,
    CommE: core::fmt::Debug,
    PinE: core::fmt::Debug,
{
    pub fn new(lines: SpiControlLines<SPI, /* CSN, */ IN, RSTN>) -> Self {
        Self {
            spi: lines.spi,
            // csn: lines.csn,
            hintn: lines.hintn,
            reset: lines.reset,
            received_packet_count: 0,
        }
    }

    /// Is the sensor indicating it has data available
    /// "In SPI and I2C mode the HOST_INTN signal is used by the BNO080 to
    /// indicate to the application processor that the BNO080 needs attention."
    fn hintn_signaled(&self) -> bool {
        self.hintn.is_low().unwrap_or(false)
    }

    /// Wait for sensor to be ready.
    /// After reset this can take around 120 ms
    /// Return true if the sensor is awake, false if it doesn't wake up
    /// `max_ms` maximum milliseconds to await for HINTN change
    fn wait_for_sensor_awake(&mut self, max_ms: usize) -> bool {
        for _ in 0..max_ms {
            if self.hintn_signaled() {
                return true;
            }
            delay_ms(1);
        }

        false
    }

    /// block on HINTN for n cycles
    fn block_on_hintn(&mut self, max_cycles: usize) -> bool {
        for _ in 0..max_cycles {
            if self.hintn_signaled() {
                return true;
            }
            delay_ms(1);
        }

        trace!("no hintn??");

        false
    }
}

impl<SPI, /* CSN, */ IN, RS, CommE, PinE> SensorInterface
    for SpiInterface<SPI, /* CSN, */ IN, RS>
where
    SPI: Write<Error = CommE> + Transfer<Error = CommE>,
    // CSN: OutputPin<Error = PinE>,
    IN: InputPin<Error = PinE>,
    RS: OutputPin<Error = PinE>,
    CommE: Debug,
    PinE: Debug,
{
    type SensorError = Error<CommE, PinE>;

    fn requires_soft_reset(&self) -> bool {
        false
    }

    fn setup(&mut self) -> Result<(), Self::SensorError> {
        // BEBOP-PATCH [6/6]: robust SPI reset sequence.
        //
        // Upstream pulses RST low for 2 ms and waits 200 ms for HINTN. Those
        // values meet the BNO085 datasheet minimums for a *cold* chip but
        // are unreliable on warm restarts: if a previous host process
        // exited mid-stream (panic / OOM-kill / `kill -9`), the chip is
        // still pumping SHTP packets onto MISO when we open the bus, and
        // the first post-reset read latches the tail of a stale packet
        // header. Symptoms observed before this patch:
        //
        //   * `CommError(SensorUnresponsive)` — HINTN didn't fall within
        //     the upstream 200 ms window (warm boot is slower than cold).
        //   * `InvalidChipId(0)` — verify_product_id read zeros from the
        //     not-yet-flushed FIFO.
        //   * Stage 3 OK but stage 4 streams all-zero quaternions — the
        //     new enable_report got buried under undrained old packets.
        //
        // All three were "fix it by power-cycling the breakout", which is
        // unacceptable in the field. The values below match Hillcrest's
        // reference sh2_hal C driver (10 ms RST hold, 500 ms HINTN wait)
        // plus a 50 ms pre-reset drain that we add for our specific
        // "previous process was streaming" case.
        self.reset.set_high().map_err(Error::Pin)?;
        delay_ms(SPI_SETUP_PRE_RESET_DRAIN_MS);

        trace!("reset cycle... ");
        self.reset.set_low().map_err(Error::Pin)?;
        delay_ms(SPI_SETUP_RST_LOW_HOLD_MS);
        self.reset.set_high().map_err(Error::Pin)?;

        // BNO firmware boot + SHTP advertisement: cold boot is ~120 ms,
        // warm restart can hit 300 ms+ because the chip aborts its prior
        // stream before booting. 500 ms matches Hillcrest sh2_hal.
        let ready = self.wait_for_sensor_awake(SPI_SETUP_HINTN_WAIT_MS);
        if !ready {
            return Err(SensorUnresponsive);
        }

        // Give the chip a beat to finish publishing its advertisement
        // response before the driver's caller starts verify_product_id;
        // without this, verify_product_id can race the advertisement and
        // misread the product-ID response.
        delay_ms(SPI_SETUP_POST_AWAKE_SETTLE_MS);

        Ok(())
    }

    fn send_and_receive_packet(
        &mut self,
        send_buf: &[u8],
        recv_buf: &mut [u8],
    ) -> Result<usize, Self::SensorError> {
        //zero the receive buffer
        for i in recv_buf[..].iter_mut() {
            *i = 0;
        }

        let tmp = &mut [0u8; PACKET_HEADER_LENGTH];
        // check how long the message to read is
        let mut read_packet_len = 0;
        let rc = self.spi.transfer(&mut tmp[..]);
        if rc.is_ok() {
            read_packet_len = SensorCommon::parse_packet_header(&tmp[..PACKET_HEADER_LENGTH]);
        }

        // Copy the write message into the buffer
        recv_buf[..send_buf.len()].copy_from_slice(send_buf);
        let total_packet_len = std::cmp::max(read_packet_len, send_buf.len());
        if total_packet_len > recv_buf.len() {
            error!(
                "Total packet length ({}) greater than recv buffer size ({})",
                total_packet_len,
                recv_buf.len()
            );
            return Err(BufferOverflow {
                packet_size: total_packet_len,
                buffer_size: recv_buf.len(),
            });
        }
        delay_ms(5);
        let rc = self.spi.transfer(&mut recv_buf[..total_packet_len]);
        if rc.is_ok() {
            read_packet_len = SensorCommon::parse_packet_header(&recv_buf[..PACKET_HEADER_LENGTH]);
        }

        if read_packet_len > 0 {
            self.received_packet_count += 1;
        }
        Ok(read_packet_len)
    }

    fn write_packet(&mut self, packet: &[u8]) -> Result<(), Self::SensorError> {
        self.spi.write(packet).map_err(Error::Comm)?;
        Ok(())
    }

    /// Read a complete packet from the sensor
    fn read_packet(&mut self, recv_buf: &mut [u8]) -> Result<usize, Self::SensorError> {
        if !self.block_on_hintn(1000) {
            error!("No message to read - HINTN timeout");
            return Err(NoDataAvailable);
        }
        // As soon as host selects CSN, HINTN resets

        // check how long the message to read is
        let mut read_packet_len = 0;
        for i in recv_buf[..PACKET_HEADER_LENGTH].iter_mut() {
            *i = 0;
        }
        let rc = self.spi.transfer(&mut recv_buf[..PACKET_HEADER_LENGTH]);
        if rc.is_ok() {
            read_packet_len = SensorCommon::parse_packet_header(&recv_buf[..PACKET_HEADER_LENGTH]);
        }

        //zero the receive buffer
        for i in recv_buf[..read_packet_len].iter_mut() {
            *i = 0;
        }
        delay_ms(5);
        let rc = self.spi.transfer(&mut recv_buf[..read_packet_len]);
        if rc.is_ok() {
            read_packet_len = SensorCommon::parse_packet_header(&recv_buf[..PACKET_HEADER_LENGTH]);
        }

        if read_packet_len > 0 {
            self.received_packet_count += 1;
        }

        Ok(read_packet_len)
    }

    fn read_with_timeout(
        &mut self,
        recv_buf: &mut [u8],
        max_ms: usize,
    ) -> Result<usize, Self::SensorError> {
        if self.wait_for_sensor_awake(max_ms) {
            return self.read_packet(recv_buf);
        }
        // trace!("Sensor did not wake for read");
        Ok(0)
    }
}
