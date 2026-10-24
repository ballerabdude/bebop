import { AIModelCard } from "../../components/AIModelCard";
import { ModelSupervisorCard } from "../../components/ModelSupervisorCard";
import { NetworkSettings } from "../../screens/settings/NetworkSettings";
import { VoiceSettings } from "../../screens/settings/VoiceSettings";

/// `#/r/:host/settings`: network/hotspot, models and voice. Not a config
/// file editor; robot configuration stays out of the app for now.
export default function SettingsRoute() {
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-4 px-4 pt-4 sm:px-6">
      <h1 className="text-lg font-semibold">Settings</h1>
      <nav aria-label="Settings sections" className="-mx-4 flex gap-1.5 overflow-x-auto px-4 sm:mx-0 sm:px-0">
        {[
          ["network", "Network"],
          ["models", "Models"],
          ["voice", "Voice"],
        ].map(([id, label]) => (
          <a
            key={id}
            href={`#settings-${id}`}
            onClick={(e) => {
              // Hash routing owns location.hash; scroll instead of navigating.
              e.preventDefault();
              document.getElementById(`settings-${id}`)?.scrollIntoView({ behavior: "smooth" });
            }}
            className="inline-flex h-control-sm shrink-0 items-center rounded-full px-3.5 text-sm font-semibold text-text-dim hover:bg-bg-elev-2 hover:text-text pointer-coarse:h-touch"
          >
            {label}
          </a>
        ))}
      </nav>
      <section id="settings-network" className="scroll-mt-20">
        <NetworkSettings />
      </section>
      <section id="settings-models" className="flex scroll-mt-20 flex-col gap-3">
        <AIModelCard showWhenUnavailable />
        <ModelSupervisorCard />
      </section>
      <section id="settings-voice" className="scroll-mt-20">
        <VoiceSettings />
      </section>
    </div>
  );
}
