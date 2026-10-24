import { useSearchParams } from "react-router";

import { Tabs } from "../../components/ui";
import { usePlaces } from "../../robot/session";
import { MapView } from "../../screens/memory/MapView";
import { PlacesView } from "../../screens/memory/PlacesView";

/// `#/r/:host/autonomy/memory[?view=map]`: place memory (review, teach,
/// known places) and the place graph. Both views read the session's shared
/// places poller.
export default function MemoryRoute() {
  const [params, setParams] = useSearchParams();
  const view = params.get("view") === "map" ? "map" : "places";
  // Keeps the pending queue polled on both views (the tab badge).
  const { pending } = usePlaces(true, { pending: true });
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-4 pt-4 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <h1 className="text-lg font-semibold">Memory</h1>
        <Tabs
          label="Memory views"
          value={view}
          onChange={(id) => setParams(id === "map" ? { view: "map" } : {}, { replace: true })}
          items={[
            {
              id: "places",
              label: (
                <span className="inline-flex items-center gap-1.5">
                  Places
                  {pending.length > 0 ? (
                    <span className="rounded-full bg-warning/20 px-1.5 text-caption font-bold text-warning-fg">
                      {pending.length}
                    </span>
                  ) : null}
                </span>
              ),
            },
            { id: "map", label: "Place graph" },
          ]}
          className="mx-0 px-0"
        />
      </div>
      {view === "map" ? <MapView /> : <PlacesView />}
    </div>
  );
}
