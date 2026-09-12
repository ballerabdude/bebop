# Bebop

The Bebop companion app. One customer-facing app for every interaction
with a Bebop robot — first-time setup, ongoing Wi-Fi configuration, motor
bench, and driving.

Built with [Tauri 2](https://tauri.app) + React + TypeScript + Vite.
Runs on desktop today and targets iOS and Android.

## Features

* **Setup wizard** — the robot broadcasts a `Bebop-XXXX` hotspot when it has
  no Wi-Fi. Join it, connect the app to the robot's setup server
  (`192.168.42.1:9091`), then scan/join your Wi-Fi and name the robot.
* **Dashboard** — live Wi-Fi state and a network-mode switch
  (`auto` / `client` / `ap`), plus access to the operator screens.
* **Connect by IP** — talk to the runtime controls server on a robot that is
  already on the network.
* **Motor bench** — live per-joint telemetry, dial-in slider with re-zero
  affordance, power-board card, sticky toolbar with E-STOP.
* **Live video** — the bebop-vision process's MJPEG streams
  (`:9092/video?stream=...`): color + depth for both Orbbec cameras, each a
  toggleable tile (`VideoScreen`).
* **Teleop** — live video and driving in one screen (`TeleopScreen`): the
  primary stream front and center with other open streams docked as
  clickable thumbnails, a sticky HUD (link / mode / wheels / battery /
  E-STOP), a one-tap "Start driving" quick-start, and every input path side
  by side — on-screen joystick, WASD / arrows, or a gamepad connected to the
  device running the app.
* **App-side gamepad dial-in & drive** — connect an 8BitDo / DualSense /
  Xbox / Switch Pro pad to your phone or laptop and drive the active joint's
  target (legged) or the wheeled chassis directly. See
  [Gamepad controls](#gamepad-controls).

## Architecture

Two independent transports, split by purpose:

| Purpose      | Talks to                    | Transport                                          | Default port |
|--------------|-----------------------------|----------------------------------------------------|--------------|
| Provisioning | `bebop-agent` (setup server)| `SetupTransport` — protobuf over binary WebSocket  | 9091         |
| Operating    | `bebop-linux` (runtime)     | `RuntimeTransport` — protobuf over binary WebSocket| 9090         |

`SetupTransport` (`src/ble/setupTransport.ts`) implements the
`BebopTransport` interface (`src/ble/transport.ts`) and provides Wi-Fi
scan/join, robot naming, and network-mode selection. It is reached either
over the robot's setup hotspot or over the LAN.

`RuntimeTransport` (`src/runtime/wsTransport.ts`) carries the high-rate
operator API — telemetry, modes, motor/wheel control, twists — using the
`bebop.runtime.v1.*` envelope and pushes telemetry at ~30 Hz.

Both envelopes are shared with the robot. TypeScript bindings are generated
by [`@bufbuild/protoc-gen-es`](https://github.com/bufbuild/protobuf-es) and
committed:

* `src/proto/bebop_pb.ts` — setup protocol (from `bebop.proto`).
* `src/proto/bebop_runtime_pb.ts` — runtime protocol (from `bebop_runtime.proto`).

Regenerate after editing a `.proto`:

```sh
npm run gen-proto
```

## Gamepad controls

The app-side gamepad flows use the
[Web Gamepad API](https://developer.mozilla.org/en-US/docs/Web/API/Gamepad_API)
and work with any pad the browser/WebView surfaces. The pad stays paired to
the device running the app and streams commands over the runtime WS — the
robot's own Bluetooth stack is not involved.

| Flow              | Drives                                                                 | Where it lives                                   |
|-------------------|------------------------------------------------------------------------|--------------------------------------------------|
| App-side dial-in  | Per-joint target position via `setMotorTarget` (legged)                | `src/input/`, `src/components/GamepadDriver.tsx` |
| App-side drive    | Body twist (vx, wz) via `setVelocityCommand` (wheeled)                 | `src/input/`, `src/components/GamepadDrive.tsx`  |

### Layout auto-detection

The driver supports both common HID layouts and picks one per pad in
`src/input/mapping.ts`:

| Layout      | Picked when                              | Examples                                              |
|-------------|------------------------------------------|-------------------------------------------------------|
| `standard`  | `Gamepad.mapping === "standard"`         | Xbox Wireless, DualSense over BT, 8BitDo in **X-input** mode (hold START + Y for 3 s) |
| `dinput`    | Anything else (8BitDo / generic HID)     | 8BitDo Mobile / Pro 2 / Zero 2 in their default Android mode, generic HID pads |

The active layout name appears as a small badge on the controller card. Each
layout knows the chord text printed on its physical pad, so the on-screen
hints read **"LB / RB"** under standard and **"L1 / R1"** under D-input.
Logical intents (`prevJoint`, `nextJoint`, `deadman`, `estop`, `resetEStop`,
`armToggle`) are the same across layouts; consumers read `snapshot.logical.*`
and don't deal with raw button indices.

### Bindings

**Motor bench (legged dial-in):**

* **LB / L1**, **RB / R1** — cycle the active joint
* **Left stick ↕** — nudge the active joint's target (rate scaled by trigger)
* **RT / R2** held — deadman; release to halt immediately
* **L3** — arm / disarm the active joint
* **B / Circle** — latch the runtime E-STOP
* **A / Cross** — clear a latched E-STOP

**Wheeled drive** (motor bench + teleop):

* **Sticks** — drive in *split* (default: left ↕ forward, right ↔ turn) or
  *arcade* (left stick does both) layout; toggled on the card, persisted per
  device. On the teleop screen with a camera, the right stick aims the camera
  instead and the layout locks to arcade.
* **RT / R2** held — deadman; release to halt immediately. Trigger pressure
  scales the request.
* **L3** — arm / disarm all wheels
* **B / Circle** — latch the runtime E-STOP
* **A / Cross** — clear a latched E-STOP

The firmware holds the last commanded twist until told otherwise, so the
bridge owns stopping: deadman release, E-STOP latch, pad disconnect, leaving
the screen, hiding the tab, or any interruption of the gamepad poll loop
(250 ms watchdog) all enqueue a zero twist exactly once per drive cycle.

### Control profiles

How sensitive the controls feel is a per-device setting switched in the app —
pickers live on the drive card and both gamepad cards. A profile sizes every
app-side input path at once and persists in localStorage under
`bebop.controlProfile`. Defined in `src/input/profile.ts`:

| Profile    | Stick deadzone | Expo | Drive limits        | Dial-in rate |
|------------|----------------|------|---------------------|--------------|
| Gentle     | 18%            | 0.5  | 0.5 m/s · 1.0 rad/s | 1.0 rad/s    |
| Standard   | 12%            | —    | 1.0 m/s · 2.0 rad/s | 2.0 rad/s    |
| Sport      | 8%             | —    | 1.5 m/s · 3.0 rad/s | 3.0 rad/s    |

The firmware still clamps twists to each wheel's `vel_max` and dial-in steps
to `slew.max_pos_step_per_tick`, so a faster profile can't exceed the robot's
hard ceilings.

### Keyboard chords per screen

| Screen      | Drive (vx / wz)  | Camera PTZ      |
|-------------|------------------|-----------------|
| Video       | —                | WASD + arrows   |
| Teleop      | WASD + arrows    | I / J / K / L   |

Every pad stops-on-exit: keys held at unmount or at a `disabled` flip enqueue
a stop/hold exactly once.

## Running on the robot

For bench work, the app can be served from the Jetson itself so any device on
the same network (or the robot's Hosted Network) can open the UI in a browser:

```sh
# on the robot, from the repo root
cd bebop-app && npm install          # once
sudo ./bebop-app/deploy/install-app-dev.sh   # installs + enables the unit
```

This installs `bebop-app.service`, which runs `deploy/dev-server.sh`
(`npm run dev`) as the `bebop` user on boot. Vite serves on `0.0.0.0:1420`
(`vite.config.ts`), so open `http://bebop.local:1420` or
`http://<robot-ip>:1420`. It's a development server, not a production build.

## Developing

```sh
# from bebop/bebop-app
nvm use        # use the node version from .nvmrc
npm install
npm run tauri dev      # desktop dev build
npm run tauri android init && npm run tauri android dev
npm run tauri ios init && npm run tauri ios dev
```

Prerequisites:

* Node 20+ (managed via `nvm`)
* Rust toolchain (`rustup` stable)
* `protoc` on `PATH` (for regenerating bindings). On macOS:
  `brew install protobuf`. On Debian/Ubuntu:
  `sudo apt-get install -y protobuf-compiler`.
* For mobile: Android Studio / Xcode toolchains (see Tauri docs)

## Repo layout

```
bebop-app/
├── src/                  # React + TypeScript UI
│   ├── App.tsx           # App shell + flow orchestration
│   ├── ble/              # SetupTransport + BebopTransport interface + types
│   ├── runtime/          # bebop-linux runtime WS client (motor bench, teleop)
│   ├── input/            # Web Gamepad API hook + per-layout button mapping
│   ├── proto/            # Generated protobuf bindings (npm run gen-proto)
│   │   ├── bebop_pb.ts            # Setup envelope
│   │   └── bebop_runtime_pb.ts    # bebop-linux runtime envelope
│   ├── components/       # Shared UI primitives + input bridges
│   │   ├── VideoFeed.tsx          # MJPEG feed tile (video/teleop)
│   │   ├── DriveJoystick.tsx      # Differential-drive pad (bench/teleop)
│   │   └── GamepadDriver/Drive    # Web-gamepad → dial-in / drive bridges
│   └── screens/          # Welcome, ConnectByIp, Wifi, Config, Dashboard,
│                         # MotorBench, Teleop, Video
├── src-tauri/            # Rust / Tauri shell
│   └── src/lib.rs        # Tauri builder
├── buf.gen.yaml          # protoc-gen-es codegen config (TS bindings)
└── README.md
```
