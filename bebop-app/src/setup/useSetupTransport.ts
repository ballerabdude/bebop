import { useEffect, useMemo } from "react";

import { SetupTransport } from "./setupTransport.ts";

/// One SetupTransport per host for the calling component. The socket opens
/// on the first request and is closed on unmount / host change. Safe under
/// React StrictMode: `disconnect()` isn't terminal, so the re-run effect's
/// requests open a fresh socket instead of inheriting the aborted one.
export function useSetupTransport(host: string): SetupTransport;
export function useSetupTransport(host: string | null): SetupTransport | null;
export function useSetupTransport(host: string | null): SetupTransport | null {
  const transport = useMemo(() => (host ? new SetupTransport(host) : null), [host]);
  useEffect(() => () => transport?.disconnect(), [transport]);
  return transport;
}
