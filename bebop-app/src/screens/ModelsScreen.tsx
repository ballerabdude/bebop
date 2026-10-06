import { AIModelCard } from "../components/AIModelCard";
import { ModelSupervisorCard } from "../components/ModelSupervisorCard";
import { Button } from "../components/ui";

/// Dedicated model provisioning + purpose-configuration screen. Reached from
/// the Dashboard and the operator screens so it's discoverable anywhere.
export function ModelsScreen({
  robotIp,
  runtimePort = 9090,
  onBack,
}: {
  robotIp: string;
  runtimePort?: number;
  onBack: () => void;
}) {
  return (
    <div className="flex flex-col flex-1 gap-4">
      <h2 className="text-2xl font-bold mt-2">Models</h2>
      <p className="text-text-dim leading-relaxed">
        Download model weights and choose which model serves each purpose.
        Gated models need a Hugging Face token.
      </p>

      <AIModelCard robotIp={robotIp} runtimePort={runtimePort} showWhenUnavailable />

      <ModelSupervisorCard robotIp={robotIp} />

      <div className="mt-auto pt-4">
        <Button variant="ghost" onClick={onBack}>
          Back
        </Button>
      </div>
    </div>
  );
}
