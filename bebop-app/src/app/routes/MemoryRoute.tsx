import { useSearchParams } from "react-router";

import { Tabs } from "../../components/ui";
import { MapScreen } from "../../screens/MapScreen";
import { PlacesScreen } from "../../screens/PlacesScreen";

/// `#/r/:host/autonomy/memory[?view=map]`: place memory, with the per-place
/// map and the place graph as a second tab. Both views read the session's
/// shared places poller.
export default function MemoryRoute() {
  const [params, setParams] = useSearchParams();
  const view = params.get("view") === "map" ? "map" : "places";
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-3 px-4 pt-4 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-lg font-semibold">Memory</h1>
        <Tabs
          label="Memory views"
          value={view}
          onChange={(id) => setParams(id === "map" ? { view: "map" } : {}, { replace: true })}
          items={[
            { id: "places", label: "Places" },
            { id: "map", label: "Map & graph" },
          ]}
          className="mx-0 px-0"
        />
      </div>
      {view === "map" ? <MapScreen /> : <PlacesScreen />}
    </div>
  );
}
