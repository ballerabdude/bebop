# Deploy

Artifacts for shipping `bebop-agent` onto a Jetson.

| File                                | Purpose                                        |
|-------------------------------------|------------------------------------------------|
| `systemd/bebop-agent.service`       | systemd unit that starts the agent on boot.    |
| `scripts/install.sh`                | Copies binary, config, and enables the unit.   |
| `scripts/uninstall.sh`              | Removes the binary and unit (keeps state dir). |
| `examples/agent.toml`               | Reference config (installed to `/etc/bebop/`). |
| `debian/`                           | Reserved for `.deb` packaging (TODO).          |

## Typical flow

On a build box (from the repo root):

```sh
just build-jetson
# equivalent to:
# cd jetson-agent && cross build --release --target aarch64-unknown-linux-gnu -p bebop-agent
```

Copy the binary and this `deploy/` tree to the Jetson (e.g. via `scp`,
Ansible, or bake into your golden image), then on the Jetson:

```sh
sudo ./deploy/scripts/install.sh ./target/aarch64-unknown-linux-gnu/release/bebop-agent
```

The `just deploy user@robot.local` recipe at the repo root does this
end-to-end: scp the binary, rsync the deploy tree, run `install.sh` over
SSH.

## Prereqs on the Jetson

The installer does not install these (your base image should):

- `bluez` (BlueZ 5.x)
- `network-manager`
- Docker + `nvidia-container-toolkit`
- `docker` set to start on boot

## Debian packaging (TODO)

A `cargo-deb` recipe or a proper `debian/` directory will be added under
`debian/` so fleet rollouts can happen via `apt`.
