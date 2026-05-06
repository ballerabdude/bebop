import { useEffect, useRef, useState } from "react";

import { DEFAULT_AGENT_PORT, WsAgentTransport } from "../ble";
import { Banner, Button, Spinner } from "../components/ui";
import { ControllersScreen } from "./ControllersScreen";

interface DirectControllersScreenProps {
  /** Robot IP address (typically reused from the user's connect-by-IP form). */
  robotIp: string;
  /** Optional override for the agent control-surface port. Defaults to 9091. */
  agentPort?: number;
  /** Called when the user taps the back button. */
  onBack: () => void;
}

/// Adapter that makes the IP-only flow able to render `ControllersScreen`.
///
/// The controllers screen expects a `BebopTransport`; on the BLE path
/// that's the live BLE/Tauri transport. There is no BLE link in the
/// IP-only flow, so we open a WebSocket to bebop-agent's network
/// control surface (`bebop-agent/src/ws.rs`, default port 9091) and use
/// that as the transport instead. From the screen's perspective the
/// API is identical — same protobuf messages, same response shapes.
///
/// Lifecycle: the WS is opened on mount and closed on unmount. We
/// don't share the socket across screen visits the way the runtime WS
/// does (motor bench), because controller pairing is an infrequent
/// operation and a fresh socket avoids any stale-connection edge cases.
export function DirectControllersScreen({
  robotIp,
  agentPort = DEFAULT_AGENT_PORT,
  onBack,
}: DirectControllersScreenProps) {
  // Keep a single transport instance across re-renders. We can't store
  // it in `useState` directly because constructing it twice (React
  // StrictMode dev double-mount) opens a second WS we never close; the
  // ref + lazy-init dance is the standard fix.
  const transportRef = useRef<WsAgentTransport | null>(null);
  if (transportRef.current === null) {
    transportRef.current = new WsAgentTransport(robotIp, agentPort);
  }

  const [connecting, setConnecting] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const t = transportRef.current!;

    void (async () => {
      setConnecting(true);
      setError(null);
      try {
        await t.connectWs();
      } catch (e) {
        if (!cancelled) {
          setError(
            e instanceof Error
              ? e.message
              : `failed to reach agent at ${robotIp}:${agentPort}`,
          );
        }
        return;
      } finally {
        if (!cancelled) setConnecting(false);
      }
    })();

    return () => {
      cancelled = true;
      // Always tear down on unmount — pairing is infrequent enough
      // that holding the socket open between visits isn't worth the
      // extra state-management complexity.
      try {
        t.disconnectWs();
      } catch {
        /* ignore */
      }
      transportRef.current = null;
    };
  }, [robotIp, agentPort]);

  if (connecting) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 text-text-dim">
        <Spinner large />
        <div className="text-sm">
          Connecting to agent at <code>{robotIp}:{agentPort}</code>…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col gap-3">
        <Banner tone="error">
          Couldn&rsquo;t reach the agent control surface at{" "}
          <code>{robotIp}:{agentPort}</code>.<br />
          {error}
        </Banner>
        <p className="text-text-dim text-sm leading-relaxed">
          Check that <code>bebop-agent</code> is running on the robot and that
          the agent&rsquo;s <code>net.ws_bind_addr</code> is reachable on this
          network. The default bind is <code>0.0.0.0:9091</code>.
        </p>
        <Button variant="secondary" onClick={onBack}>
          Back
        </Button>
      </div>
    );
  }

  return (
    <ControllersScreen
      transport={transportRef.current!}
      onDone={onBack}
      backLabel="Back to motor bench"
    />
  );
}
