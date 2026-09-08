import { Kbd, Modal } from "./ui";

const GROUPS: { title: string; keys: [string, string][] }[] = [
  {
    title: "Navigate",
    keys: [
      ["← / →", "previous / next tick"],
      ["d", "jump to next tick with disagree cells"],
      ["c", "jump to next high-caution, unreviewed tick"],
      ["n", "jump to next unreviewed tick"],
      ["f", "jump to first tick"],
    ],
  },
  {
    title: "Paint",
    keys: [
      ["1 / 2 / 3", "blocked / navigable / caution"],
      ["[ / ]", "brush size down / up"],
      ["u", "undo last stroke"],
      ["s", "save hand grid"],
      ["r", "revert to saved state"],
    ],
  },
  {
    title: "View",
    keys: [
      ["t", "cycle displayed grid (auto / hand / fused)"],
      ["o", "toggle label overlay on the color image"],
      ["?", "this help"],
    ],
  },
];

export default function ShortcutsModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  return (
    <Modal open={open} onClose={onClose} title="Keyboard shortcuts">
      <div className="space-y-5">
        {GROUPS.map((g) => (
          <div key={g.title}>
            <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
              {g.title}
            </h3>
            <div className="space-y-1.5">
              {g.keys.map(([k, d]) => (
                <div key={k} className="flex items-center justify-between gap-4 text-xs">
                  <span className="text-zinc-400">{d}</span>
                  <span className="flex gap-1">
                    {k.split(" ").map((part, i) => (
                      <Kbd key={i}>{part}</Kbd>
                    ))}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
        <p className="border-t border-white/[0.06] pt-3 text-[11px] text-zinc-500">
          Corrections save additively as <code className="text-zinc-300">hand</code>{" "}
          in the label npz — teacher keys are never modified, and training
          prefers <code className="text-zinc-500">hand</code> where present.
        </p>
      </div>
    </Modal>
  );
}
