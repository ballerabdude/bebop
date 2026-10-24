# Dev HTTPS and the microphone

Push-to-talk in the agent console uses `getUserMedia`, which needs a secure
context. In development the app is served over plain HTTP (Vite on `:1420`)
and the robot's services are cleartext (`ws://`, `http://`). Options,
simplest first.

## On the machine running Vite: localhost

`http://localhost:1420` counts as a secure context, and since the page stays
HTTP its `ws://<robot>:9090` and other cleartext calls are not mixed content.
Nothing to set up; it only covers that one machine.

## Other desktops and Android: Chrome's insecure-origin flag

1. Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
2. Enable it and list the dev origins, comma-separated, e.g.
   `http://192.168.0.68:1420,http://bebop.local:1420`.
3. Relaunch Chrome.

The Vite dev server answers only `localhost`, IP addresses and `bebop.local`
(`allowedHosts` in `bebop-app/vite.config.ts`), so name the origin one of
those ways. There is no iOS equivalent.

## Any browser, including iOS: the Caddy TLS proxy

When the page is HTTPS, the app switches every robot URL to `wss://` /
`https://` on the same port (`bebop-app/src/runtime/urls.ts`).
`Caddyfile.dev` terminates TLS on a workstation and forwards to the robot's
cleartext services. From the repo root:

```sh
(cd bebop-app && npm run build)                            # Caddy serves bebop-app/dist
ROBOT_IP=192.168.0.174 caddy run --config Caddyfile.dev    # ROBOT_IP defaults to that address
```

| Proxy port | Serves |
|---|---|
| 8443 | the built app |
| 9090 | `bebop-linux` runtime WS and captures |
| 9092 | `bebop-vision` WHEP video |
| 9093 | `bebop-voice` |

It has no blocks for 9091 (setup), 9094 (models) or 9095 (places), so behind
it the setup wizard, Settings › Network, the GPU budget card and Memory get
no answer. Add a block shaped like the 9090 one for each port you need. Only
the WHEP signalling goes through Caddy; the WebRTC media (SRTP over UDP)
flows directly between the browser and the robot.

Trust Caddy's local CA once per client. On a Linux proxy host its root
certificate is `~/.local/share/caddy/pki/authorities/local/root.crt`:

- **macOS:** import it into Keychain Access and set it to Always Trust.
- **iPhone:** install it as a profile (Settings › General › VPN & Device
  Management), then enable full trust under Settings › General › About ›
  Certificate Trust Settings.

Then open `https://<proxy-host>:8443` and add the proxy host as the robot in
the app; every service URL resolves to it.

## Tailscale (untested)

`tailscale serve` gets a publicly trusted `*.ts.net` certificate (no CA to
install) and proxies WebSockets, so it could replace Caddy: serve the app and
each robot port on the same port number, e.g. on the robot
`tailscale serve --bg --https=9090 http://127.0.0.1:9090`. A `*.ts.net` Host
header reaching the Vite dev server would need adding to `allowedHosts`.
Nothing in this repo sets it up, and it has not been verified with the app.
