import { useEffect, useState } from "react";

import type { BebopTransport, DeviceInfo, RobotConfig } from "../ble";
import { Banner, Button, Field, Spinner } from "../components/ui";

export function ConfigScreen({
  transport,
  onDone,
}: {
  transport: BebopTransport;
  onDone: () => void;
}) {
  const [info, setInfo] = useState<DeviceInfo | null>(null);
  const [config, setConfig] = useState<RobotConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const [i, c] = await Promise.all([
          transport.getDeviceInfo(),
          transport.getRobotConfig(),
        ]);
        setInfo(i);
        setConfig(c);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [transport]);

  async function save() {
    if (!config) return;
    setError(null);
    setSaving(true);
    try {
      await transport.setRobotConfig(config);
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  if (!config || !info) {
    return (
      <div className="flex flex-col flex-1 justify-center items-center">
        <Spinner large />
      </div>
    );
  }

  const inputClass =
    "bg-bg-elev border border-border rounded-[var(--radius-card)] px-3.5 py-3 text-text outline-none focus:border-accent transition-colors duration-120";

  return (
    <div className="flex flex-col flex-1 gap-4">
      <h2 className="text-2xl font-bold mt-2">Name your robot</h2>
      <p className="text-text-dim leading-relaxed">
        Connected to <strong>{info.model}</strong> · SN {info.serialNumber} ·
        agent v{info.agentVersion}
      </p>

      {error ? <Banner tone="error">{error}</Banner> : null}

      <Field label="Robot name">
        <input
          className={inputClass}
          value={config.robotName}
          onChange={(e) =>
            setConfig({ ...config, robotName: e.currentTarget.value })
          }
          placeholder="e.g. Hallway Bebop"
          maxLength={32}
        />
      </Field>

      <Field
        label="Owner ID"
        hint="Used to associate this robot with your account."
      >
        <input
          className={inputClass}
          value={config.ownerId}
          onChange={(e) =>
            setConfig({ ...config, ownerId: e.currentTarget.value })
          }
          placeholder="user@example.com"
        />
      </Field>

      <Field label="Timezone">
        <input
          className={inputClass}
          value={config.timezone}
          onChange={(e) =>
            setConfig({ ...config, timezone: e.currentTarget.value })
          }
          placeholder="America/Los_Angeles"
        />
      </Field>

      <div className="mt-auto pt-4 flex flex-col gap-3">
        <Button
          onClick={save}
          loading={saving}
          disabled={config.robotName.trim().length === 0}
        >
          Save and continue
        </Button>
      </div>
    </div>
  );
}
