# robot-app/

Container image for the Bebop robot application — the actual workload that
runs on the Jetson under `bebop-agent`'s supervision.

| File                  | Purpose                                                              |
|-----------------------|----------------------------------------------------------------------|
| `Dockerfile`          | NVIDIA L4T base, installs the Python entrypoint.                     |
| `docker-compose.yml`  | Local convenience for running the image standalone.                  |
| `main.py`             | Stub entrypoint (replace with your real robot application).          |

## Building

```sh
# From the repo root, via just:
just build-app    # buildx for linux/arm64

# Or directly:
docker buildx build \
    --platform linux/arm64 \
    -t your-registry/bebop-app:dev \
    -f jetson-agent/robot-app/Dockerfile \
    jetson-agent/robot-app

# On the Jetson itself (native build):
docker build \
    -t your-registry/bebop-app:dev \
    -f jetson-agent/robot-app/Dockerfile \
    jetson-agent/robot-app
```

## Why containers

- Deploys are just new image tags pushed to a registry — OTA is a
  `docker pull` followed by a `docker run`, both driven by `bebop-agent`.
- Dev and robot environments stay in sync (same CUDA / cuDNN / TensorRT
  versions baked into the image).
- Rollbacks are trivial: point the channel manifest at the previous tag.

## Publishing an update

1. Push the new image to your registry (`just push-app`).
2. Update the channel manifest (`updates.bebop.example.com/channels/stable.json`
   or whatever you configured in `ota.manifest_url`):

   ```json
   {
     "image":  "your-registry/bebop-app:1.2.3",
     "digest": "sha256:...",
     "notes":  "bugfix release"
   }
   ```

3. Robots pick it up on their next poll interval (default 5 minutes).

See [`../../docs/ota-flow.md`](../../docs/ota-flow.md) for the full update
lifecycle.
