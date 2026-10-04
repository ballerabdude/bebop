import { useEffect, useRef, useState } from "react";

import {
  getOrCreateRuntimeTransport,
  type RuntimeTransport,
  type VoiceView,
} from "../runtime";
import { Banner, Button, Card } from "../components/ui";

const EMPTY_VOICE: VoiceView = {
  present: false,
  running: false,
  state: "",
  detail: "",
  service: "",
  model: "",
};

/// Voice (speech-to-speech) control + status. Start/stop the on-robot
/// `bebop-voice.service` (Qwen3-Omni via TensorRT Edge-LLM) over the runtime
/// WebSocket; the audio itself is served by that unit on :9093.
///
/// The push-to-talk audio bridge lands with the voice service; this screen
/// owns service lifecycle and status so the operator can provision the model
/// (Models page), start the service, and see when it is ready.
export function VoiceScreen({
  robotIp,
  runtimePort = 9090,
  onBack,
}: {
  robotIp: string;
  runtimePort?: number;
  onBack: () => void;
}) {
  const transportRef = useRef<RuntimeTransport | null>(null);
  const [voice, setVoice] = useState<VoiceView>(EMPTY_VOICE);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!robotIp) return;
    const transport = getOrCreateRuntimeTransport(robotIp, runtimePort);
    transportRef.current = transport;
    let cancelled = false;
    const offTelemetry = transport.onTelemetry((s) => {
      if (!cancelled) setVoice(s.voice);
    });
    void transport
      .connect(robotIp, runtimePort)
      .then(() => transport.subscribeTelemetry(5))
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      offTelemetry();
    };
  }, [robotIp, runtimePort]);

  const transport = transportRef.current;

  async function toggle() {
    if (!transport) return;
    setError(null);
    setBusy(true);
    try {
      await transport.setVoiceEnabled(!voice.running);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col flex-1 gap-4">
      <h2 className="text-2xl font-bold mt-2">Voice</h2>
      <p className="text-text-dim leading-relaxed">
        Talk to the robot. Speech-to-speech runs on-device with Qwen3-Omni;
        download the voice model from the Models page first.
      </p>

      {error ? <Banner tone="error">{error}</Banner> : null}

      {!voice.present ? (
        <Banner tone="info">
          Voice support isn&apos;t installed on this robot yet. Deploy the
          current firmware and the <code>bebop-voice.service</code> unit (see{" "}
          <code>scripts/install-jetson.sh</code>), then reconnect.
        </Banner>
      ) : (
        <Card>
          <div className="py-2 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="text-xs text-text-dim uppercase tracking-wider">
                Speech-to-speech
              </div>
              <div className="flex items-center gap-2">
                <div
                  className={`w-2.5 h-2.5 rounded-full ${
                    voice.running ? "bg-success" : "bg-text-dim/40"
                  }`}
                />
                <span className="text-[12px] text-text-dim">
                  {voice.running ? "Running" : voice.state || "Stopped"}
                </span>
              </div>
            </div>
            {voice.model ? (
              <div className="text-[13px] text-text-dim font-mono">
                {voice.model}
              </div>
            ) : null}
            {voice.detail ? (
              <div className="text-[12px] text-text-dim leading-snug">
                {voice.detail}
              </div>
            ) : null}
            <Button
              variant={voice.running ? "ghost" : "secondary"}
              loading={busy}
              onClick={toggle}
            >
              {voice.running ? "Stop voice service" : "Start voice service"}
            </Button>
            {!voice.running ? (
              <p className="text-[12px] text-text-dim leading-relaxed">
                The first start downloads nothing but builds the model&apos;s
                TensorRT engines — this can take several minutes.
              </p>
            ) : null}
          </div>
        </Card>
      )}

      <Card>
        <div className="py-2 flex flex-col gap-2">
          <div className="text-xs text-text-dim uppercase tracking-wider">
            Talk
          </div>
          <p className="text-[13px] text-text-dim leading-relaxed">
            Push-to-talk audio is served by <code>bebop-voice.service</code> on{" "}
            <code>:9093</code>. Start the service, then hold to talk.
          </p>
          <Button disabled variant="secondary">
            Hold to talk
          </Button>
        </div>
      </Card>

      <div className="mt-auto pt-4">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
      </div>
    </div>
  );
}
