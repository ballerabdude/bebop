import { Button } from "../components/ui";

export function WelcomeScreen({ onStart }: { onStart: () => void }) {
  return (
    <div className="flex flex-col flex-1 justify-center items-center text-center gap-6">
      <div className="mb-6">
        <div className="text-[56px] mb-3" aria-hidden>
          🤖
        </div>
        <h1 className="text-2xl font-bold mb-2">Set up your Bebop</h1>
        <p className="text-text-dim leading-relaxed max-w-sm">
          Power on your robot and stay within Bluetooth range. This wizard will
          help you connect it to Wi-Fi and finish first-time setup.
        </p>
      </div>
      <Button onClick={onStart}>Get started</Button>
    </div>
  );
}
