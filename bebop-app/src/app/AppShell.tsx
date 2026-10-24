// Robot shell: status bar, navigation and the global E-STOP around every
// operator page (`#/r/:host/*`).
//
//   phones   (<768)      bottom tab bar (safe-area aware) with the E-STOP
//                        dock directly above it, both in the layout flow so
//                        they never cover page content
//   tablets  (768-1279)  compact icon rail on the left, E-STOP top-right
//   desktop  (>=1280)    labeled side rail, E-STOP top-right
//
// The Drive page hides most chrome: the rail collapses to icons, the phone
// tab bar is replaced by a menu button, and only the status bar and E-STOP
// remain. On phones the E-STOP then floats bottom-right at z-40 (above any
// page overlay up to z-39). Its size depends on the latch state (about
// 240 x 112 px latched), so the shell publishes the measured corner as
// `--estop-float-w` / `--estop-float-h` on <html>; Drive pads against those.
//
// The shell also publishes `--shell-top-inset` / `--shell-bottom-inset` (the
// status bar, and the phone E-STOP dock + tab bar or floating E-STOP). Modal
// layers (ui/Dialog, the nav sheet) stop at those edges, so the E-STOP is
// never dimmed or covered while a confirm is open.

import { Suspense, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ReactNode, Ref, RefObject } from "react";
import { Link, NavLink, Outlet, useLocation, useParams } from "react-router";

import { cn, PageState, StatusDot, StatusPill } from "../components/ui";
import type { Tone } from "../components/ui";
import {
  RobotProvider,
  useConnection,
  usePlacesReachable,
  useRobot,
  useTelemetry,
} from "../robot/session";
import type { RuntimeSnapshot } from "../runtime/types";
import { EStopBar, EStopDock, EStopHotkey, EStopStrip } from "./EStopButton";
import {
  AutonomyIcon,
  DiagnosticsIcon,
  DriveIcon,
  MemoryIcon,
  MenuIcon,
  RobotsIcon,
  SettingsIcon,
} from "./icons";
import { robotPath } from "./paths";
import type { RobotSection } from "./paths";
import { rememberRobot, useRememberedRobots } from "./robots";

interface NavItem {
  label: string;
  section: RobotSection;
  /// Prefix that marks the item active (defaults to the exact section).
  match: (path: string) => boolean;
  icon: () => ReactNode;
}

const NAV: NavItem[] = [
  { label: "Drive", section: "drive", match: (p) => p.endsWith("/drive"), icon: DriveIcon },
  {
    label: "Autonomy",
    section: "autonomy",
    match: (p) => p.endsWith("/autonomy"),
    icon: AutonomyIcon,
  },
  {
    label: "Memory",
    section: "autonomy/memory",
    match: (p) => p.includes("/autonomy/memory"),
    icon: MemoryIcon,
  },
  {
    label: "Diagnostics",
    section: "diagnostics/motors",
    match: (p) => p.includes("/diagnostics"),
    icon: DiagnosticsIcon,
  },
  {
    label: "Settings",
    section: "settings",
    match: (p) => p.includes("/settings"),
    icon: SettingsIcon,
  },
];

// ------------------------------------------------------------- status bar

interface StatusSlice {
  mode: RuntimeSnapshot["mode"] | null;
  visionPresent: boolean;
  visionRunning: boolean;
  voicePresent: boolean;
  voiceRunning: boolean;
  socKnown: boolean;
  soc: number;
  volts: number;
  powerPresent: boolean;
  powerReceived: boolean;
  drivePresent: boolean;
  driver: "you" | "other" | "none";
  driverName: string;
}

function selectStatus(s: RuntimeSnapshot | null): StatusSlice {
  const p = s?.power;
  const d = s?.drive;
  return {
    mode: s?.mode ?? null,
    visionPresent: !!s?.vision.present,
    visionRunning: !!s?.vision.running,
    voicePresent: !!s?.voice.present,
    voiceRunning: !!s?.voice.running,
    powerPresent: !!p?.present,
    powerReceived: !!p?.statusReceived,
    socKnown: !!p?.present && !!p.statusReceived && p.stateOfChargePct >= 0,
    soc: Math.round(p?.stateOfChargePct ?? -1),
    volts: Math.round((p?.batteryVoltageV ?? 0) * 10) / 10,
    drivePresent: !!d?.present,
    driver: d?.youAreActiveOperator ? "you" : d?.hasActiveOperator ? "other" : "none",
    driverName: d?.activeOperatorName ?? "",
  };
}

const MODE_LABEL: Record<string, string> = {
  UNSPECIFIED: "Unknown",
  IDLE: "Idle",
  DIAL_IN: "Armed mode",
  RUN_POLICY: "Policy",
};

function StatusBar({
  onMenu,
  showMenu,
  headerRef,
}: {
  onMenu: () => void;
  showMenu: boolean;
  headerRef: Ref<HTMLElement>;
}) {
  const { host } = useRobot();
  const link = useConnection();
  const s = useTelemetry(selectStatus, 5);
  const robots = useRememberedRobots();
  const name = robots.find((r) => r.host === host)?.name;
  const placesUp = usePlacesReachable();

  const runtimeTone: Tone =
    link.state === "connected" ? "ok" : link.state === "degraded" ? "warn" : link.state === "connecting" ? "warn" : "err";
  const svc = (present: boolean, running: boolean): Tone =>
    link.state !== "connected" && link.state !== "degraded"
      ? "dim"
      : !present
        ? "dim"
        : running
          ? "ok"
          : "dim";
  const batteryTone: Tone = !s.socKnown ? "dim" : s.soc < 25 ? "err" : s.soc < 60 ? "warn" : "ok";

  return (
    <header
      ref={headerRef}
      className="sticky top-0 z-30 border-b border-border bg-bg/90 pt-safe backdrop-blur-md"
    >
      <div className="flex min-h-14 items-center gap-2 px-3 py-1.5 sm:px-4 md:min-h-16 [@media(max-height:500px)]:min-h-12 [@media(max-height:500px)]:py-1">
        {showMenu ? (
          <button
            type="button"
            onClick={onMenu}
            aria-label="Open navigation"
            className="inline-flex h-touch w-touch shrink-0 items-center justify-center rounded-card text-text-dim hover:bg-bg-elev-2 hover:text-text md:hidden"
          >
            <MenuIcon />
          </button>
        ) : null}
        <Link
          to="/"
          className="flex min-h-touch min-w-0 shrink items-center gap-1 rounded-card pr-2 text-left hover:bg-bg-elev-2"
          title="Switch robot"
        >
          <span className="hidden text-text-dim md:inline-flex">
            <RobotsIcon />
          </span>
          <span className="min-w-0">
            <span className="block truncate text-sm font-semibold leading-tight">
              {name || host}
            </span>
            {name ? (
              <span className="block truncate font-mono text-micro leading-tight text-text-dim">
                {host}
              </span>
            ) : null}
          </span>
        </Link>

        <div
          className="flex shrink-0 items-center gap-2 rounded-full border border-border px-2.5 py-1"
          aria-label="Service links"
        >
          <StatusDot
            tone={runtimeTone}
            pulse={link.state === "connecting"}
            label={`Runtime ${link.state}`}
          />
          <StatusDot tone={svc(s.visionPresent, s.visionRunning)} label={`Video ${s.visionRunning ? "running" : "stopped"}`} />
          <StatusDot tone={svc(s.voicePresent, s.voiceRunning)} label={`Voice ${s.voiceRunning ? "running" : "stopped"}`} />
          <StatusDot
            tone={placesUp === null ? "dim" : placesUp ? "ok" : "err"}
            label={`Places ${placesUp === null ? "unknown" : placesUp ? "reachable" : "unreachable"}`}
          />
        </div>

        <div className="flex min-w-0 flex-1 items-center gap-1.5 overflow-hidden">
          {link.state === "degraded" ? (
            <StatusPill tone="warn" dot pulse>
              link degraded
            </StatusPill>
          ) : link.state !== "connected" ? (
            <StatusPill tone="warn" dot pulse>
              {link.state === "connecting" ? "connecting" : "offline"}
            </StatusPill>
          ) : null}
          {s.powerPresent ? (
            <StatusPill tone={batteryTone} title="Battery">
              {s.socKnown ? `${s.soc}%` : s.powerReceived ? `${s.volts.toFixed(1)} V` : "batt —"}
            </StatusPill>
          ) : null}
          {s.mode ? (
            <StatusPill tone={s.mode === "IDLE" ? "dim" : "info"} className="hidden sm:inline-flex" title="Firmware mode">
              {MODE_LABEL[s.mode] ?? s.mode}
            </StatusPill>
          ) : null}
          {s.drivePresent && s.driver !== "none" ? (
            <StatusPill tone={s.driver === "you" ? "ok" : "warn"} title="Who holds the drive seat">
              {s.driver === "you" ? "you drive" : s.driverName ? `${s.driverName} driving` : "other driving"}
            </StatusPill>
          ) : null}
        </div>
        <EStopBar className="ml-auto shrink-0" />
      </div>
      <EStopStrip />
    </header>
  );
}

// ------------------------------------------------------------- navigation

function SideRail({ host, compact }: { host: string; compact: boolean }) {
  const { pathname } = useLocation();
  return (
    <nav
      aria-label="Robot sections"
      className={cn(
        "hidden shrink-0 flex-col gap-1 border-r border-border bg-bg-elev/40 px-2 pt-[max(0.75rem,env(safe-area-inset-top))] pb-3 md:flex",
        compact ? "w-[72px]" : "w-[72px] xl:w-56",
      )}
    >
      {NAV.map((item) => {
        const active = item.match(pathname);
        return (
          <NavLink
            key={item.label}
            to={robotPath(host, item.section)}
            aria-current={active ? "page" : undefined}
            title={item.label}
            className={cn(
              "flex h-12 items-center gap-3 rounded-card px-3 text-sm font-semibold transition-colors",
              compact ? "justify-center" : "justify-center xl:justify-start",
              active ? "bg-accent/15 text-accent" : "text-text-dim hover:bg-bg-elev-2 hover:text-text",
            )}
          >
            <item.icon />
            <span className={cn(compact ? "sr-only" : "sr-only xl:not-sr-only")}>{item.label}</span>
          </NavLink>
        );
      })}
    </nav>
  );
}

function BottomTabs({ host, navRef }: { host: string; navRef: Ref<HTMLElement> }) {
  const { pathname } = useLocation();
  return (
    <nav
      ref={navRef}
      aria-label="Robot sections"
      className="shrink-0 border-t border-border bg-bg pb-safe md:hidden"
    >
      <div className="grid grid-cols-5">
        {NAV.map((item) => {
          const active = item.match(pathname);
          return (
            <NavLink
              key={item.label}
              to={robotPath(host, item.section)}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex h-14 flex-col items-center justify-center gap-0.5 text-caption font-semibold",
                active ? "text-accent" : "text-text-dim",
              )}
            >
              <item.icon />
              {item.label}
            </NavLink>
          );
        })}
      </div>
    </nav>
  );
}

function NavSheet({ host, open, onClose }: { host: string; open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-x-0 top-0 bottom-[var(--shell-bottom-inset,0px)] z-[55] bg-black/55 md:hidden"
      onClick={onClose}
    >
      <div
        className="absolute inset-x-0 bottom-0 rounded-t-card border-t border-border bg-bg-elev pb-safe"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="grid grid-cols-3 gap-1 p-3">
          {NAV.map((item) => (
            <NavLink
              key={item.label}
              to={robotPath(host, item.section)}
              onClick={onClose}
              className="flex h-20 flex-col items-center justify-center gap-1 rounded-card text-label font-semibold text-text-dim hover:bg-bg-elev-2 hover:text-text"
            >
              <item.icon />
              {item.label}
            </NavLink>
          ))}
          <Link
            to="/"
            onClick={onClose}
            className="flex h-20 flex-col items-center justify-center gap-1 rounded-card text-label font-semibold text-text-dim hover:bg-bg-elev-2 hover:text-text"
          >
            <RobotsIcon />
            Robots
          </Link>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------- shell

const INSET_VARS = [
  "--shell-top-inset",
  "--shell-bottom-inset",
  "--estop-float-w",
  "--estop-float-h",
] as const;

/// Publish the shell's fixed edges as CSS variables on <html> (see the
/// header comment). Re-measured on resize and whenever one of the parts
/// changes size (the E-STOP grows while latched).
function useShellInsets(
  header: RefObject<HTMLElement | null>,
  dock: RefObject<HTMLDivElement | null>,
  tabs: RefObject<HTMLElement | null>,
  layoutKey: string,
): void {
  useLayoutEffect(() => {
    const root = document.documentElement;
    const visible = (el: Element | null | undefined) => {
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.height > 0 ? r : null;
    };
    const measure = () => {
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const h = visible(header.current);
      const d = visible(dock.current);
      const t = visible(tabs.current);
      const floating = !!dock.current?.hasAttribute("data-estop-float");
      let bottomTop = vh;
      if (d) bottomTop = Math.min(bottomTop, floating ? d.top - 8 : d.top);
      if (t) bottomTop = Math.min(bottomTop, t.top);
      const px = (n: number) => `${Math.max(0, Math.round(n))}px`;
      root.style.setProperty("--shell-top-inset", px(h ? h.bottom : 0));
      root.style.setProperty("--shell-bottom-inset", px(vh - bottomTop));
      root.style.setProperty("--estop-float-w", px(floating && d ? vw - d.left : 0));
      root.style.setProperty("--estop-float-h", px(floating && d ? vh - d.top : 0));
    };
    measure();
    const ro = new ResizeObserver(measure);
    for (const el of [header.current, dock.current, tabs.current]) if (el) ro.observe(el);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
      for (const v of INSET_VARS) root.style.removeProperty(v);
    };
  }, [header, dock, tabs, layoutKey]);
}

function Shell() {
  const { host } = useRobot();
  const { pathname } = useLocation();
  const immersive = pathname.endsWith("/drive");
  const [menuOpen, setMenuOpen] = useState(false);
  const headerRef = useRef<HTMLElement | null>(null);
  const dockRef = useRef<HTMLDivElement | null>(null);
  const tabsRef = useRef<HTMLElement | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);
  useShellInsets(headerRef, dockRef, tabsRef, immersive ? "drive" : "page");

  useEffect(() => {
    rememberRobot(host);
  }, [host]);

  // Every section shares one scroll container: start a new page at the top
  // (keyed on the path, so Memory's ?view toggle keeps its place).
  useEffect(() => {
    mainRef.current?.scrollTo(0, 0);
  }, [pathname]);

  return (
    <div className="flex h-dvh w-full overflow-hidden">
      <SideRail host={host} compact={immersive} />
      <div className="flex min-w-0 flex-1 flex-col">
        <StatusBar headerRef={headerRef} showMenu={immersive} onMenu={() => setMenuOpen(true)} />
        <main
          ref={mainRef}
          id="main"
          className={cn(
            "flex min-h-0 flex-1 flex-col overflow-x-hidden overflow-y-auto",
            immersive ? "" : "pb-6",
          )}
        >
          <Suspense fallback={<PageState kind="loading" />}>
            <Outlet />
          </Suspense>
        </main>
        {/* Phones: E-STOP dock + tabs sit below the scroll area, in flow.
            On the full-bleed Drive page the dock floats bottom-right. */}
        <EStopDock rootRef={dockRef} floating={immersive} />
        {immersive ? null : <BottomTabs host={host} navRef={tabsRef} />}
      </div>
      <NavSheet host={host} open={menuOpen} onClose={() => setMenuOpen(false)} />
      <EStopHotkey />
    </div>
  );
}

/// Route element for `/r/:host/*`.
export default function RobotLayout() {
  // React Router hands params back already decoded.
  const { host = "" } = useParams();
  return (
    <RobotProvider key={host} host={host}>
      <Shell />
    </RobotProvider>
  );
}
