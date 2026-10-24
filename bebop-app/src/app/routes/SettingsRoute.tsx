import { Navigate, useParams } from "react-router";

import { Tabs } from "../../components/ui";
import { AIModelCard } from "../../components/AIModelCard";
import { useRobot } from "../../robot/session";
import { VoiceSettingsPanel } from "../../screens/autonomy/VoiceSettings";
import { ModelSupervisorPanel } from "../../screens/settings/ModelSupervisorPanel";
import { NetworkSettings } from "../../screens/settings/NetworkSettings";
import { robotPath, SETTINGS_TABS } from "../paths";
import type { SettingsTab } from "../paths";

const LABEL: Record<SettingsTab, string> = {
  network: "Network",
  models: "Models",
  voice: "Voice & agent",
};

/// `#/r/:host/settings/:section`: network/hotspot, models and voice. Not a
/// config-file editor; robot configuration stays out of the app.
export default function SettingsRoute() {
  const { section } = useParams();
  const { host } = useRobot();
  if (!SETTINGS_TABS.includes(section as SettingsTab)) {
    return <Navigate to={robotPath(host, "settings/network")} replace />;
  }
  const s = section as SettingsTab;
  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-4 px-4 pt-4 sm:px-6">
      <h1 className="text-lg font-semibold">Settings</h1>
      <Tabs
        label="Settings sections"
        items={SETTINGS_TABS.map((id) => ({
          id,
          label: LABEL[id],
          to: robotPath(host, `settings/${id}`),
        }))}
      />
      {s === "network" ? <NetworkSettings /> : null}
      {s === "models" ? (
        <div className="flex flex-col gap-4">
          <ModelSupervisorPanel />
          <AIModelCard showWhenUnavailable />
        </div>
      ) : null}
      {s === "voice" ? <VoiceSettingsPanel host={host} /> : null}
    </div>
  );
}
