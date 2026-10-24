import { VoiceScreen } from "../../screens/VoiceScreen";

/// `#/r/:host/autonomy`. The voice + text agent console comes with the
/// Autonomy redesign; for now this is push-to-talk inside the shell.
export default function AutonomyRoute() {
  return <VoiceScreen />;
}
