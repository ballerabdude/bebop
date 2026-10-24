// Hash router: `#/...` URLs work under the Vite dev server, a plain static
// server (no SPA fallback needed) and `tauri://`.

import { lazy } from "react";
import { createHashRouter, Navigate } from "react-router";

import { RobotPicker } from "./RobotPicker";

const RobotLayout = lazy(() => import("./AppShell"));
const SetupFlow = lazy(() => import("./SetupFlow"));
const DriveRoute = lazy(() => import("./routes/DriveRoute"));
const AutonomyRoute = lazy(() => import("./routes/AutonomyRoute"));
const MemoryRoute = lazy(() => import("./routes/MemoryRoute"));
const DiagnosticsRoute = lazy(() => import("./routes/DiagnosticsRoute"));
const SettingsRoute = lazy(() => import("./routes/SettingsRoute"));

export const router = createHashRouter([
  { path: "/", element: <RobotPicker /> },
  { path: "/setup/*", element: <SetupFlow /> },
  {
    path: "/r/:host",
    element: <RobotLayout />,
    children: [
      { index: true, element: <Navigate to="drive" replace /> },
      { path: "drive", element: <DriveRoute /> },
      { path: "autonomy", element: <AutonomyRoute /> },
      { path: "autonomy/memory", element: <MemoryRoute /> },
      { path: "diagnostics", element: <Navigate to="motors" replace /> },
      { path: "diagnostics/:tab", element: <DiagnosticsRoute /> },
      { path: "settings", element: <SettingsRoute /> },
      { path: "*", element: <Navigate to="drive" replace /> },
    ],
  },
  { path: "*", element: <Navigate to="/" replace /> },
]);
