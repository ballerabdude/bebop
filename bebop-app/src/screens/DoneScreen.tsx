import { useEffect, useState } from "react";

import type { AppStatus, BebopTransport, OtaStatus, WifiStatus } from "../ble";
import { Banner, Button, Card } from "../components/ui";

export function DoneScreen({
  transport,
  wifi,
  onStartOver,
}: {
  transport: BebopTransport;
  wifi: WifiStatus;
  onStartOver: () => void;
}) {
  const [app, setApp] = useState<AppStatus | null>(null);
  const [ota, setOta] = useState<OtaStatus | null>(null);
  const [otaRunning, setOtaRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      const [a, o] = await Promise.all([
        transport.getAppStatus(),
        transport.getOtaStatus(),
      ]);
      setApp(a);
      setOta(o);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function checkForUpdate() {
    setError(null);
    setOtaRunning(true);
    try {
      await transport.triggerOta();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setOtaRunning(false);
    }
  }

  return (
    <div className="flex flex-col flex-1 gap-4">
      <div className="text-center mt-4">
        <div
          className="w-16 h-16 mx-auto mb-3 rounded-full bg-success/15 text-success flex items-center justify-center text-[34px] font-bold"
          aria-hidden
        >
          ✓
        </div>
        <h2 className="text-2xl font-bold">You're all set</h2>
        <p className="text-text-dim leading-relaxed mt-1">
          Your robot is online and ready to go.
        </p>
      </div>

      {error ? <Banner tone="error">{error}</Banner> : null}

      <Card>
        <Kv label="Wi-Fi" value={wifi.ssid} />
        <Kv label="IP address" value={wifi.ipAddress || "—"} />
        {app ? (
          <>
            <Kv label="Robot app" value={app.state} />
            <Kv label="Image" value={app.image} mono />
          </>
        ) : null}
        {ota ? <Kv label="Updates" value={ota.state} /> : null}
      </Card>

      <div className="mt-auto pt-4 flex flex-row gap-3">
        <Button
          variant="secondary"
          onClick={checkForUpdate}
          loading={otaRunning}
          className="flex-1"
        >
          Check for updates
        </Button>
        <Button variant="ghost" onClick={onStartOver} className="flex-1">
          Set up another
        </Button>
      </div>
    </div>
  );
}

function Kv({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex justify-between gap-3 py-2.5 border-b border-border text-sm last:border-b-0">
      <span className="text-text-dim">{label}</span>
      <strong className={mono ? "font-mono text-xs break-all text-right" : ""}>
        {value}
      </strong>
    </div>
  );
}
