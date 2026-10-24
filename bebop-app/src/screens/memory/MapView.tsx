// Memory › Place graph (`?view=map`): places and the routes recorded between
// them, from the places service (:9095). Observe-only: it never invents
// routes; an empty graph is an explicit blank state.

import { useEffect, useState } from "react";

import { Banner, Button } from "../../components/ui";
import { PORTS } from "../../robot/endpoints";
import { usePlaces, useRobot } from "../../robot/session";
import { GRAPH_POLL_MS, usePlaceGraph } from "./mapPolling";
import { formatAge } from "./mapView";
import { PlaceGraphCard } from "./PlaceGraphCard";

const secs = (ms: number) => `${(ms / 1000).toFixed(ms % 1000 ? 1 : 0)} s`;

export function MapView() {
  const { endpoints } = useRobot();
  const shared = usePlaces();
  const graph = usePlaceGraph(endpoints.places);
  const [now, setNow] = useState(() => performance.now());

  useEffect(() => {
    const id = window.setInterval(() => setNow(performance.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col gap-4">
      {shared.error ? (
        <Banner
          tone="error"
          title={`Places service unreachable (:${PORTS.places})`}
          action={
            <Button variant="secondary" size="sm" onClick={() => void shared.refresh()}>
              Retry
            </Button>
          }
        >
          {shared.error}
        </Banner>
      ) : null}

      <PlaceGraphCard state={graph} />

      <p className="text-caption text-text-dim">
        Read-only · graph {secs(GRAPH_POLL_MS)}
        {graph.receivedAt ? ` · ${formatAge((now - graph.receivedAt) / 1000)} old` : ""}
      </p>
    </div>
  );
}
