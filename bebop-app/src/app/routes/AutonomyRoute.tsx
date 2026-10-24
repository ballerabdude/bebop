import { AutonomyScreen } from "../../screens/autonomy/AutonomyScreen";

/// `#/r/:host/autonomy`: the agent console (text + voice turns, transcript
/// with tool cards, live camera and robot status).
export default function AutonomyRoute() {
  return <AutonomyScreen />;
}
