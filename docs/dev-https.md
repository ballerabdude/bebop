# Dev HTTPS / microphone access

`getUserMedia` needs a **secure context**, but in dev the app is served over
plain HTTP (`vite`, `http://<workstation>:1420`) and the robot speaks cleartext
`ws://`. Two ways to make the mic work.

## Fastest: Chrome insecure-origin bypass (desktop / Android Chrome)

No cert, no proxy — just tell Chrome to treat the dev origin as secure:

1. Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
2. **Enabled**, then list origins (comma-separated), e.g.
   `http://192.168.0.68:1420,http://localhost:1420`.
3. Relaunch Chrome.

`getUserMedia` now works over `http://`, and because the page stays HTTP there
is no mixed content when it calls `ws://<robot>:9090` etc. **No iOS
equivalent** — this is fine for a Mac/Linux dev machine, not the phone.

## Self-signed TLS (any browser, incl. the phone)

The app derives its scheme from the page (`src/runtime/urls.ts`): over HTTPS it
uses `wss://` / `https://` automatically. So terminate TLS once in front of the
robot's (still-cleartext) services with [Caddy](https://caddyserver.com) and
point the app at the proxy's host.

```
# on a box that can reach the robot (e.g. this workstation)
ROBOT_IP=192.168.0.174 caddy run --config Caddyfile.dev
```

`Caddyfile.dev` binds the workstation's **9090 / 9092 / 9093** with TLS and
proxies to the robot, and serves the built app on **:8443**. One local CA issues
every endpoint (Caddy auto-reloads `caddy trust` not needed).

Trust Caddy's root CA on each client (one-time):

- **macOS:** import `~/.local/share/caddy/pki/authorities/local/root.crt` into
  Keychain Access and set it to **Always Trust**. Chrome/Safari then accept it.
- **iPhone:** AirDrop/AirDrop the `root.crt` (or serve it), install the profile
  (Settings → General → VPN & Device Management), then
  Settings → General → About → Certificate Trust Settings → enable full trust.

Then open `https://<workstation>:8443` and enter the **workstation** IP in the
app (that's where the TLS proxy lives); the runtime/voice/video endpoints resolve
to `wss://<workstation>:9090` etc.

## Zero-config alternative: Tailscale

`tailscale serve` issues a **publicly-trusted** `*.ts.net` cert — no CA install —
and proxies WebSockets. Put the robot (or the workstation) on the tailnet and
`tailscale serve --bg --https=<port> http://127.0.0.1:<port>` for 9090/9092/9093
plus the app. See the **Mobile** section of [`voice.md`](voice.md).
