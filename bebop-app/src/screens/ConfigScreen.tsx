import { useEffect, useState } from "react";

import type { BebopTransport, DeviceInfo, RobotConfig } from "../setup";
import { Banner, Button, Field, INPUT_CLASS, PageState } from "../components/ui";

/// Setup › Identity: name the robot. Only `robot_name` is persisted by the
/// agent; the other RobotConfig fields are sent back unchanged.
export function ConfigScreen({
  transport,
  onDone,
  submitLabel = "Save and continue",
}: {
  transport: BebopTransport;
  onDone: (config: RobotConfig) => void;
  submitLabel?: string;
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
      onDone(config);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  if (!config || !info) {
    return error ? (
      <PageState kind="error" title="Couldn't read the robot's identity">
        {error}
      </PageState>
    ) : (
      <PageState kind="loading" />
    );
  }

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
          className={INPUT_CLASS}
          value={config.robotName}
          onChange={(e) =>
            setConfig({ ...config, robotName: e.currentTarget.value })
          }
          placeholder="e.g. Hallway Bebop"
          maxLength={32}
        />
      </Field>


      <div className="mt-auto pt-4 flex flex-col gap-3">
        <Button
          onClick={save}
          loading={saving}
          disabled={config.robotName.trim().length === 0}
        >
          {submitLabel}
        </Button>
      </div>
    </div>
  );
}
