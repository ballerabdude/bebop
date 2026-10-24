# bebop-app

The operator app for a Bebop robot: first-time setup, driving, the voice and
text agent, place memory, diagnostics and settings. It is a web app (React +
TypeScript + Vite, Tailwind 4) that talks **directly** to the robot's
services from the browser, and it is built to work on phones, tablets and
desktops alike. There is no backend of its own and no login.

Many people can watch a robot at once; only one drives at a time (the
firmware's drive seat). The E-STOP is on screen on every robot page.

## Pages

The router is a hash router (`#/...`), so the same build works under the
Vite dev server, any static file server and a future native shell without
an SPA fallback.

| Route | Page |
|---|---|
| `#/` | **Robots**: remembered robots with live status (probes `:9090` and `:9091` every 10 s), open, rename, remove, add by address, start setup |
| `#/setup/connect` → `wifi` → `identity` | **Setup** wizard, outside the robot shell. Over the robot's hotspot the address is `192.168.42.1`. Finishes on the robot's Drive page |
| `#/r/:host/drive` | **Drive**: video-first teleop (touch joystick, WASD / arrows, gamepad) |
| `#/r/:host/autonomy` | **Autonomy**: the agent console (text and push-to-talk turns, tool and motion events, cancel) |
| `#/r/:host/autonomy/memory[?view=map]` | **Memory**: the Places tab (recognition, pending-place review and import, teaching, known places) and the Place graph tab (`?view=map`, read-only). Protocol: `docs/places.md` |
| `#/r/:host/diagnostics/:tab` | **Diagnostics**: `motors`, `power-imu`, `captures`, `services`, `cameras` |
| `#/r/:host/settings/:section` | **Settings**: `network` (Wi-Fi status, hotspot), `models` (GPU budget, model downloads), `voice` (agent engine, voice, persona) |

### The robot shell

Everything under `#/r/:host/` sits in one shell (`src/app/AppShell.tsx`):

- **Phones (<768 px)**: status bar on top, bottom tab bar, and an E-STOP
  dock directly above the tabs. Dock and tabs are part of the layout, not
  overlays, so they never cover page content. The dock shows the latch
  reason ("Right motor fault: DC bus over-voltage") and a small Reset.
  At 500 px tall or less the tab bar turns compact and the E-STOP moves
  into the status bar.
- **Tablets (768-1279 px)**: icon rail on the left, E-STOP at the right of
  the status bar, and a full-width strip under it with the reason and Reset
  while latched. Short viewports (≤500 px tall, i.e. landscape phones) drop
  the strip and show the reason and Reset inline in the status bar.
- **Desktop (≥1280 px)**: the same with a labeled rail.
- **Drive** gets the same navigation as every other section: on phones the
  in-flow E-STOP dock and tab bar, on wider screens a side rail that stays
  icon-only to give the video more width.
- **Dialogs never cover the E-STOP**: the shell publishes
  `--shell-top-inset` / `--shell-bottom-inset` on `<html>`, and `ui/Dialog`
  stops at those edges. Changing section scrolls the page to the top.
- **Space** triggers the E-STOP anywhere outside a text field. The button
  latches optimistically and follows telemetry; a press that never reached
  the robot turns into a pulsing RETRY. Reset is deliberately a quiet
  secondary control.

The status bar shows the robot's name and host, service dots (runtime,
video, voice, places), battery, firmware mode and who holds the drive seat.

## Services, ports and transports

Every port and URL comes from `src/robot/endpoints.ts`; screens never spell
one. Schemes follow the page (`ws`/`http` on plain HTTP, `wss`/`https`
behind a TLS proxy, see `src/runtime/urls.ts`).

| Port | Service | Transport | Used for |
|---|---|---|---|
| 9090 | `bebop-linux` firmware | WS `/ws`, binary protobuf (`bebop_runtime.proto`); HTTP `/healthz`, `/captures*` | Telemetry, twists, E-STOP, modes, unit start/stop, capture downloads. The safety authority |
| 9091 | `bebop-agent` | WS `/ws`, binary protobuf (`bebop.proto`); HTTP `/healthz` | Setup: Wi-Fi scan/join, robot name, hotspot settings |
| 9092 | `bebop-vision` videoserver | WHEP (`POST /whep?stream=…`, `DELETE` the session), H.264 over WebRTC; HTTP `/snapshot`, drive notes (`/intent`, `/event`, `/notes`) | Video (one encoder per stream shared by all viewers); Drive's drive notes |
| 9093 | `bebop-voice` | WS `/voice` (JSON events + PCM16); HTTP `/healthz`, `/config` | Agent turns (`user_text`, typed events, cancel), voice settings. Protocol: `docs/voice.md` |
| 9094 | `bebop-models` | HTTP `/healthz`, `/models`, `POST /models/{id}/load\|unload` (CORS `*`) | GPU budget and Edge-LLM backend lifecycle |
| 9095 | places (inside `bebop-vision`) | HTTP `/healthz`, `/places`, `/places/pending` (`/{id}/suggest`, `/import`), `/whereami`, `/places/graph`, `/places/{id}` (`/views`, `/facts`, `/keyframes/…`, `DELETE`) | Place memory: review and import, teaching, recognition, the place graph. One shared poller per robot session; the Place graph tab polls `/places/graph` itself |

`RobotProvider` (`src/robot/session.tsx`) owns one firmware socket per host,
a ref-counted telemetry store (rate = the fastest mounted consumer, link
liveness, backoff reconnect), the shared places poller and the E-STOP
latch. Pages use its hooks (`useTelemetry`, `useConnection`, `useEStop`,
`usePlaces`, `useRuntimeAction`, ...) and never touch ports or sockets.

Setup traffic uses `SetupTransport` (`src/setup/`), which opens lazily and
reconnects on the next request; get one with `useSetupTransport(host)`.

## Developing

Node 22 (`.nvmrc`). From `bebop-app/`:

```sh
npm install
npm run dev                  # Vite on :1420 (0.0.0.0)
# then open http://localhost:1420/#/  and add the robot (bebop.local or its IP)
```

Any page can be opened straight at a robot, e.g.
`http://localhost:1420/#/r/bebop.local/diagnostics/services`. The app
fetches from the robot's services cross-origin, which all allow it (CORS
`*`, no auth).

```sh
npx tsc --noEmit             # type check
npm test                     # node --test over tests/*.test.mjs
npm run build                # tsc + vite build into dist/
npm run gen-proto            # regenerate src/proto/* with buf (after .proto edits)
```

Tests are plain `node:test` files that import the TypeScript sources
directly (`--experimental-transform-types`), so pure modules import their
siblings with explicit `.ts` extensions. They cover the twist stream, drive
input arbiter, telemetry/reconnect logic, E-STOP latch and reason text, the
setup transport lifecycle, the agent turn model, the places poller, drive
notes and the drive HUD, the model-eviction preview and the place-graph
helpers (`npm run test:map` runs only those).

## On the robot

`deploy/install-app-dev.sh` installs `bebop-app.service`, which runs
`deploy/dev-server.sh` (`npm run dev`) as the `bebop` user on boot, so any
device on the robot's network (or its hotspot) can open
`http://bebop.local:1420` or `http://<robot-ip>:1420`. It is the Vite dev
server, not a production build: after `git pull` it picks up changes on its
own; after a dependency change run `npm install` and restart the unit.

The microphone (push-to-talk) needs a secure context: `localhost`, or
HTTPS through a TLS proxy such as `../Caddyfile.dev`.

## Structure

```
src/
├── app/            router, AppShell (status bar, nav, E-STOP), robot picker,
│   │               setup wizard, remembered robots, route helpers
│   └── routes/     one thin component per route
├── robot/          RobotProvider + hooks, endpoints registry, telemetry,
│                   places and E-STOP stores, E-STOP reason text
├── runtime/        firmware WS client (protobuf), twist stream, URL schemes
├── setup/          bebop-agent client (SetupTransport, useSetupTransport)
├── voice/          agent session, voice I/O worklets, voice service HTTP
├── input/          drive input arbiter, gamepad, keyboard, control profiles
├── components/
│   ├── ui/         the kit: Button, Card, Banner, StatusPill, Tabs, Spinner,
│   │               SegmentedControl, Switch, Field, Dialog/useConfirm, PageState
│   └── *.tsx       VideoFeed, DriveJoystick, GamepadDrive, AIModelCard, ...
├── screens/
│   ├── drive/      Drive page
│   ├── autonomy/   agent console and voice settings
│   ├── memory/     PlacesView, WhereAmICard, PendingReview, KnownPlaces,
│   │               MapView + PlaceGraphCard (the place graph), the graph
│   │               poller and helpers (mapPolling, mapView), places REST calls
│   ├── diagnostics/ one file per tab
│   ├── settings/   Network, model supervisor
│   └── WifiScreen, ConfigScreen (setup steps)
├── proto/          generated protobuf bindings (do not edit)
└── App.css         design tokens (dark only), type scale, control heights
```

Design tokens live in `App.css` (`@theme`): surfaces, intent colors with
`*-fg` tints, a dense type scale (`text-micro` … `text-body-sm`), control
heights 36/44/56 and a 44 px touch minimum (`pointer-coarse:h-touch`). The
app is dark-only.

### Control profiles

How sensitive the drive controls feel is a per-device setting
(`src/input/profile.ts`, stored in localStorage as `bebop.controlProfile`):

| Profile  | Stick deadzone | Expo | Drive limits        |
|----------|----------------|------|---------------------|
| Gentle   | 18%            | 0.5  | 0.5 m/s · 1.0 rad/s |
| Standard | 12%            | —    | 1.0 m/s · 2.0 rad/s |
| Sport    | 8%             | —    | 1.5 m/s · 3.0 rad/s |

The firmware still clamps twists to each wheel's `vel_max`.

## Tauri

`src-tauri/` holds an unused Tauri 2 shell, kept for a future native app
(mobile builds, microphone without HTTPS). Nothing in `src/` depends on it;
`npm run tauri dev` still works if you want a desktop window.
