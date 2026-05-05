import { Button } from "../components/ui";

interface WelcomeProps {
  onStart: () => void;
  /** Skip BLE setup and connect by IP directly. Shown as a secondary
   *  action so users who already have the robot on Wi-Fi can bypass the
   *  pairing wizard. */
  onConnectByIp: () => void;
}

export function WelcomeScreen({ onStart, onConnectByIp }: WelcomeProps) {
  return (
    <div className="flex flex-col flex-1 justify-center items-center text-center gap-6">
      <div className="mb-4">
        <div className="text-[56px] mb-3" aria-hidden>
          🤖
        </div>
        <h1 className="text-2xl font-bold mb-2">Set up your Bebop</h1>
        <p className="text-text-dim leading-relaxed max-w-sm">
          Power on your robot and stay within Bluetooth range. This wizard will
          help you connect it to Wi-Fi and finish first-time setup.
        </p>
      </div>
      <div className="flex flex-col gap-3 w-full max-w-xs">
        <Button onClick={onStart}>Get started</Button>
        <Button variant="ghost" onClick={onConnectByIp}>
          Already on Wi-Fi? Connect by IP
        </Button>
      </div>
    </div>
  );
}
