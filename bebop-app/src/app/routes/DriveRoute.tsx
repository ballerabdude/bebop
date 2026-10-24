import { TeleopScreen } from "../../screens/TeleopScreen";

/// `#/r/:host/drive`. The video-first Drive redesign comes later; for now
/// this is the teleop screen inside the shell.
export default function DriveRoute() {
  return <TeleopScreen />;
}
