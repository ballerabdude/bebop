# OTA Flow

Bebop's OTA system updates **the robot application container**. Host OS
updates (JetPack, kernel, bebop-agent itself) are out of scope here — use
your normal Debian / image-based flow for those.

## Pieces

- **Registry**: stores tagged container images. Can be any OCI registry
  (`nvcr.io`, GHCR, ECR, a self-hosted Harbor, etc.).
- **Manifest server**: serves a small JSON document per channel:

  ```json
  {
    "image":  "your-registry/bebop-app:1.2.3",
    "digest": "sha256:abc123...",
    "notes":  "improve arm kinematics"
  }
  ```

  Kept tiny so robots can poll it cheaply. Serve from S3 + CloudFront or
  similar.

- **`bebop-agent` OTA subsystem**: polls the manifest every
  `ota.poll_interval_secs` (default 300s). When `image` differs from what's
  running, it triggers an update.

## Lifecycle

```
┌───────────┐    poll     ┌────────────┐
│ agent     │ ──────────▶ │ manifest   │
│ (robot)   │ ◀────────── │ server     │
└─────┬─────┘   JSON      └────────────┘
      │ target != current
      ▼
┌───────────┐   docker pull   ┌────────────┐
│ agent     │ ──────────────▶ │ registry   │
└─────┬─────┘                 └────────────┘
      │ success
      ▼
┌───────────┐
│ container │ stop old → start new (same name)
│ manager   │
└───────────┘
```

## Guarantees

- **Atomic-ish swap**: the new container starts only after the image pull
  succeeds; on failure the old container keeps running.
- **Idempotent**: if `target == current`, the poll is a no-op.
- **Manual trigger**: the mobile app can force a check via `TriggerOtaRequest`.
- **Observable**: current state is always available via `GetOtaStatusRequest`
  (`IDLE` / `CHECKING` / `DOWNLOADING` / `APPLYING` / `SUCCESS` / `FAILED`).

## Rollback

To roll back a bad release, update the manifest to point at the previous
tag. Robots will pick it up on their next poll and downgrade. Keep old
tags in the registry — never overwrite.

## Channels

A robot's channel is controlled by its `ota.manifest_url`. Typical setup:

- `stable` — rolls forward after bake-time on `beta`.
- `beta` — receives new images first; run on a few internal robots.
- `dev` — latest `main` builds.

You can move a specific robot to another channel by updating
`/etc/bebop/agent.toml` on it (via Ansible, SSH, or — eventually — a
"set channel" BLE RPC).

## Security (roadmap)

- Pin manifests by `digest` so a compromised registry can't swap image
  contents under a tag.
- Sign manifests (cosign / minisign) and verify in `bebop-agent` before
  applying.
- Require TLS + client-cert auth to the manifest server.
