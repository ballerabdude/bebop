import { Button } from "../components/ui";

interface WelcomeProps {
  /** Begin first-time setup (join the robot's hotspot, then configure Wi-Fi). */
  onStart: () => void;
  /** Connect to a robot that is already on the network. */
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
          Power on your robot. If it isn&rsquo;t on Wi-Fi yet it will broadcast
          a <strong>Bebop-XXXX</strong> hotspot — join it from your phone&rsquo;s
          Wi-Fi settings, then continue here.
        </p>
      </div>
      <div className="flex flex-col gap-3 w-full max-w-xs">
        <Button onClick={onStart}>Set up a robot</Button>
        <Button variant="ghost" onClick={onConnectByIp}>
          Already on Wi-Fi? Open controls
        </Button>
      </div>
    </div>
  );
}
