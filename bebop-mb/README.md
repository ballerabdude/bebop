# Bebop Motherboard (`bebop-mb`)

KiCad 10 project for the **Bebop Connector / motherboard** board. It is the interconnect
between an **NVIDIA Jetson Orin Nano** carrier board, a **Teensy 4.1** microcontroller, a
**Bosch BNO085 9‑DoF IMU** (Adafruit breakout), and a **CAN bus** front end. The board mostly
provides headers/breakouts and routes shared buses between these modules.

> Silkscreen title block: *"Bebop Connector"*.

## Schematic sheets

| Sheet file | Contents |
|---|---|
| `bebop-mb.kicad_sch` | Root sheet: BNO085 IMU header (`J8`) + IMU breakouts (`J11`/`J12`), CAN connectors (`J1`–`J4`), mounting holes, power |
| `teensy-mcu.kicad_sch` | Teensy 4.1 (`U1`) and its two breakout headers (`J6`/`J7`) |
| `jetson-interface.kicad_sch` | Jetson 40‑pin IDC header (`J5`) and its two breakout headers (`J9`/`J10`) |
| `can-transceiver.kicad_sch` | SN65HVD230 CAN transceiver (`U2`) + decoupling (`C14`) |

## Component / connector reference

| Ref | Part / type | Footprint | Purpose |
|---|---|---|---|
| `U1` | Teensy 4.1 (48‑pin THT, 2×24) | `bebop-mb:Teensy41_Simple` | Main MCU |
| `J6` | 1×24 socket | `PinSocket_1x24` | Teensy left row breakout (Teensy pins 1–24) |
| `J7` | 1×24 socket | `PinSocket_1x24` | Teensy right row breakout (Teensy pins 25–48) |
| `J5` | 2×20 IDC box header | `Connector_IDC:IDC-Header_2x20_P2.54mm_Vertical` | Ribbon to Jetson 40‑pin header |
| `J9` | 1×20 socket | `PinSocket_1x20` | Jetson **odd** pins (1,3,…,39) breakout |
| `J10` | 1×20 socket | `PinSocket_1x20` | Jetson **even** pins (2,4,…,40) breakout |
| `J8` | BNO085 breakout header (2×6) | `bebop-mb:BNO08x_Breakout` | Adafruit BNO085 IMU |
| `J11` | 1×6 socket | `PinSocket_1x6` | BNO085 primary‑pin breakout (VIN/3Vo/GND/SCL/SDA/INT) |
| `J12` | 1×6 socket | `PinSocket_1x6` | BNO085 secondary‑pin breakout (BT/P0/P1/RST/DI/CS) |
| `U2` | SN65HVD230 (SOIC‑8) | — | 3.3 V CAN transceiver |
| `C14` | 100 nF | `C_0603` | `U2` decoupling |
| `J1` | 3‑pin screw terminal | `TerminalBlock_Xinya XY308 3P` | CAN bus (CANH/CANL/GND) |
| `J2`,`J3`,`J4` | JST‑GH 2‑pin | `JST_GH_SM02B-GHS-TB` | CAN bus daisy‑chain (CANH/CANL) |
| `H1`–`H4` | M3 mounting holes | `MountingHole_3.2mm_M3` | Mechanical |

## Power rails

| Net | Notes |
|---|---|
| `+3.3V` | Main logic rail. Feeds BNO085 `VIN`, CAN transceiver `U2`, and the BNO `P0`/`P1` mode straps |
| `GND` | Common ground |
| `Vin` | Teensy `Vin` (broken out on `J7` pin 1) |

The Jetson 40‑pin header also exposes its own `3.3V` (pins 1, 17) and `5.0V` (pins 2, 4) on
`J9`/`J10`; these are **not** tied to the board `+3.3V` rail unless you bridge them.

---

## Teensy 4.1 (`U1`) → breakouts `J6` / `J7`

`J6`/`J7` are 1×24 socket breakouts; **`J6` pad N = Teensy pin N**, **`J7` pad N = Teensy pin N+24**.
Silkscreen on `J6`/`J7` shows the Teensy pin name.

### Active Teensy connections

| Function | Teensy pin (GPIO) | Breakout | Net |
|---|---|---|---|
| IMU SPI clock | 13 (SCK/LED) | `J7` pad 14 | `IMU_SCK` |
| IMU SPI MISO | 12 | `J6` pad 14 | `IMU_MISO` |
| IMU SPI MOSI | 11 | `J6` pad 13 | `IMU_MOSI` |
| IMU SPI CS | 10 | `J6` pad 12 | `IMU_CS` |
| IMU interrupt | 37 | `J7` pad 20 | `IMU_INT` |
| IMU reset | 36 | `J7` pad 21 | `IMU_RST` |
| CAN1 TX (CTX1) | 22 | `J7` pad 5 | `CAN1_TX` |
| CAN1 RX (CRX1) | 23 | `J7` pad 4 | `CAN1_RX` |
| I²C SCL (spare) | 19 (A5/SCL) | `J7` pad 8 | `IMU_SCL` |
| I²C SDA (spare) | 18 (A4/SDA) | `J7` pad 9 | `IMU_SDA` |

> `IMU_SCL`/`IMU_SDA` are the legacy I²C pins. The IMU was migrated to **SPI**, so these are
> now just an unused I²C bus broken out on `J7`.

---

## BNO085 IMU (`J8`) — wired for SPI

The BNO085 is configured for **SPI** (not I²C). Protocol is selected by tying `P0`/`P1` high.

| `J8` pin | BNO pad | SPI role | Net |
|---|---|---|---|
| 1 | VIN | power | `+3.3V` |
| 3 | GND | ground | `GND` |
| 4 | SCL | SCK (clock) | `IMU_SCK` |
| 5 | SDA | MISO (sensor → host) | `IMU_MISO` |
| 11 | DI | MOSI (host → sensor) | `IMU_MOSI` |
| 12 | CS | chip select | `IMU_CS` |
| 6 | INT | data‑ready | `IMU_INT` |
| 10 | RST | reset | `IMU_RST` |
| 8 | P0 (PS0) | mode select → **high** | `+3.3V` |
| 9 | P1 (PS1) | mode select → **high** | `+3.3V` |
| 2 | 3Vo | BNO 3.3 V out (breakout only) | — |
| 7 | BT | boot (breakout only) | `J12` pin 1 |

`J11` breaks out the primary row (VIN/3Vo/GND/SCL/SDA/INT) and `J12` the secondary row
(BT/P0/P1/RST/DI/CS); both simply follow whatever `J8` carries.

---

## Shared IMU SPI bus (Teensy **and** Jetson)

The BNO085 SPI bus is wired **in parallel** to both the Teensy and the Jetson so either can act
as master. They must **not** be active at the same time.

```
            IMU_SCK / IMU_MISO / IMU_MOSI / IMU_CS / IMU_INT / IMU_RST
                                  |
        +---------------+---------+---------+----------------+
        |               |                   |                |
   BNO085 (J8)     Teensy U1           Jetson J5         Breakouts
                (13/12/11/10,        (SPI0 + GPIO,      (J6/J7, J9/J10,
                 37/36)               see below)         J11/J12)
```

| Signal | BNO085 | Teensy 4.1 | Jetson 40‑pin |
|---|---|---|---|
| SCK | SCL | GPIO13 | pin 23 (SPI0_SCK) |
| MISO | SDA | GPIO12 | pin 21 (SPI0_MISO) |
| MOSI | DI | GPIO11 | pin 19 (SPI0_MOSI) |
| CS | CS | GPIO10 | pin 24 (SPI0_CS0\*) |
| INT | INT | GPIO37 | pin 7 (GPIO09) |
| RST | RST | GPIO36 | pin 15 (GPIO12) |

> ⚠️ **Bus contention:** there is no hardware isolation (no mux/buffer) between the two masters.
> Firmware must keep the **inactive** master's SCK/MOSI/CS (and INT/RST) pins as **high‑impedance
> inputs** so the two masters never drive the bus simultaneously.

---

## Jetson 40‑pin header (`J5`) → breakouts `J9` / `J10`

`J5` is a 2×20 IDC box header (ribbon to the Jetson carrier 40‑pin expansion header).
`J9` breaks out the **odd** column, `J10` the **even** column. Silkscreen on `J9`/`J10` is
`<Jetson pin #> <signal>`.

| Pin | Signal | | Pin | Signal |
|---:|---|---|---:|---|
| 1 | 3.3V | | 2 | 5.0V |
| 3 | I2C1_SDA | | 4 | 5.0V |
| 5 | I2C1_SCL | | 6 | GND |
| 7 | GPIO09 → **IMU_INT** | | 8 | UART1_TXD |
| 9 | GND | | 10 | UART1_RXD |
| 11 | UART1_RTS\* | | 12 | I2S0_SCLK |
| 13 | SPI1_SCK | | 14 | GND |
| 15 | GPIO12 → **IMU_RST** | | 16 | SPI1_CS1\* |
| 17 | 3.3V | | 18 | SPI1_CS0\* |
| 19 | SPI0_MOSI → **IMU_MOSI** | | 20 | GND |
| 21 | SPI0_MISO → **IMU_MISO** | | 22 | SPI1_MISO |
| 23 | SPI0_SCK → **IMU_SCK** | | 24 | SPI0_CS0\* → **IMU_CS** |
| 25 | GND | | 26 | SPI0_CS1\* |
| 27 | I2C0_SDA | | 28 | I2C0_SCL |
| 29 | GPIO01 | | 30 | GND |
| 31 | GPIO11 | | 32 | GPIO07 |
| 33 | GPIO13 | | 34 | GND |
| 35 | I2S0_FS | | 36 | UART1_CTS\* |
| 37 | SPI1_MOSI | | 38 | I2S0_DIN |
| 39 | GND | | 40 | I2S0_DOUT |

Pins in **bold** are routed to the shared IMU bus; all other pins are simply broken out on
`J9`/`J10` for access.

---

## CAN bus (`U2` + `J1`–`J4`)

`U2` (SN65HVD230, 3.3 V) bridges the Teensy CAN1 controller to the physical bus.

| `U2` side | Connects to |
|---|---|
| `D` (driver in) | `CAN1_TX` ← Teensy GPIO22 (CTX1) |
| `R` (receiver out) | `CAN1_RX` → Teensy GPIO23 (CRX1) |
| `CANH` / `CANL` | CAN connectors `J1`–`J4` |
| `VCC` / `GND` | `+3.3V` / `GND` |

- `J1` — 3‑pin screw terminal: **CANH / CANL / GND**.
- `J2`, `J3`, `J4` — JST‑GH 2‑pin: **CANH / CANL** (for daisy‑chaining nodes).

Add a 120 Ω termination resistor at the bus ends as needed (not currently populated on‑board).

---

## Notes

- This board is KiCad 10 (name‑based nets; no numbered net table).
- After editing any schematic sheet, run **Tools → Update PCB from Schematic** and re‑run **DRC**.
- The IMU SPI bus sharing relies on firmware discipline (see the contention warning above); if you
  want hardware‑guaranteed isolation, add series resistors or a GPIO/jumper‑selected SPI bus switch.
