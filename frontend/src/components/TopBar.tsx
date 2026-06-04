import { AoiSummary } from "../lib/useAois";

export type ViewMode = "subsidence" | "drift";

export function TopBar({
  aois, activeCode, onSelect, observationDate,
  mode, onModeChange,
  showBlocks, onToggleBlocks,
}: {
  aois: AoiSummary[] | null;
  activeCode: string | null;
  onSelect: (code: string) => void;
  observationDate?: string;
  mode: ViewMode;
  onModeChange: (m: ViewMode) => void;
  showBlocks: boolean;
  onToggleBlocks: (v: boolean) => void;
}) {
  return (
    <div className="absolute top-0 left-0 right-0 px-6 py-3 bg-gradient-to-b from-ink-950/95 via-ink-950/70 to-transparent flex items-start justify-between pointer-events-none">
      <div className="pointer-events-auto">
        <div className="text-[10px] uppercase tracking-[0.3em] text-slate-500">
          infra-proptech / structural deformation monitor
        </div>
        <div className="mt-1 flex gap-2">
          {aois?.map(a => {
            const active = a.aoi_code === activeCode;
            return (
              <button
                key={a.aoi_code}
                onClick={() => onSelect(a.aoi_code)}
                className={[
                  "px-3 py-1.5 text-xs uppercase tracking-widest border transition",
                  active
                    ? "border-signal-cyan text-signal-cyan bg-signal-cyan/10"
                    : "border-wire-700 text-slate-400 hover:border-wire-500 hover:text-slate-200",
                ].join(" ")}
                title={a.phenomenon}
              >
                {a.name}
              </button>
            );
          })}
        </div>
      </div>

      <div className="pointer-events-auto flex items-center gap-6">
        <BlocksToggle on={showBlocks} onChange={onToggleBlocks} />
        <ModeToggle mode={mode} onChange={onModeChange} />
        <div className="text-right">
          <div className="text-[10px] uppercase tracking-[0.3em] text-slate-500">Observation</div>
          <div className="text-xl tabular-nums">{observationDate ?? "—"}</div>
        </div>
      </div>
    </div>
  );
}


function ModeToggle({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.3em] text-slate-500 text-right">Mode</div>
      <div className="mt-1 flex border border-wire-700">
        <ModeButton active={mode === "subsidence"} onClick={() => onChange("subsidence")}>Subsidence</ModeButton>
        <ModeButton active={mode === "drift"}      onClick={() => onChange("drift")}>Drift</ModeButton>
      </div>
    </div>
  );
}


/** ARCHITECTURE_THREE C3 — toggle the block-aggregation overlay. */
function BlocksToggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.3em] text-slate-500 text-right">Overlay</div>
      <div className="mt-1 flex border border-wire-700">
        <ModeButton active={on} onClick={() => onChange(!on)}>Blocks</ModeButton>
      </div>
    </div>
  );
}


function ModeButton({
  active, onClick, children,
}: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={[
        "px-3 py-1.5 text-xs uppercase tracking-widest transition",
        active
          ? "bg-signal-cyan/10 text-signal-cyan"
          : "text-slate-400 hover:text-slate-200",
      ].join(" ")}
    >
      {children}
    </button>
  );
}
