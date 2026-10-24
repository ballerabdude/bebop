import { Navigate, useParams } from "react-router";

import { Tabs } from "../../components/ui";
import { CamerasTab } from "../../screens/diagnostics/CamerasTab";
import { CapturesTab } from "../../screens/diagnostics/CapturesTab";
import { MotorsTab } from "../../screens/diagnostics/MotorsTab";
import { PowerImuTab } from "../../screens/diagnostics/PowerImuTab";
import { ServicesTab } from "../../screens/diagnostics/ServicesTab";
import { useRobot } from "../../robot/session";
import { DIAGNOSTICS_TABS, robotPath } from "../paths";
import type { DiagnosticsTab } from "../paths";

const LABEL: Record<DiagnosticsTab, string> = {
  motors: "Motors",
  "power-imu": "Power & IMU",
  captures: "Captures",
  services: "Services",
  cameras: "Cameras",
};

/// `#/r/:host/diagnostics/:tab`
export default function DiagnosticsRoute() {
  const { tab } = useParams();
  const { host } = useRobot();
  if (!DIAGNOSTICS_TABS.includes(tab as DiagnosticsTab)) {
    return <Navigate to={robotPath(host, "diagnostics/motors")} replace />;
  }
  const t = tab as DiagnosticsTab;
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-3 px-4 pt-4 sm:px-6">
      <h1 className="text-lg font-semibold">Diagnostics</h1>
      <Tabs
        label="Diagnostics sections"
        items={DIAGNOSTICS_TABS.map((id) => ({
          id,
          label: LABEL[id],
          to: robotPath(host, `diagnostics/${id}`),
        }))}
      />
      {t === "motors" ? <MotorsTab /> : null}
      {t === "power-imu" ? <PowerImuTab /> : null}
      {t === "captures" ? <CapturesTab /> : null}
      {t === "services" ? <ServicesTab /> : null}
      {t === "cameras" ? <CamerasTab /> : null}
    </div>
  );
}
