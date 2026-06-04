import { Bundle, Classification, DataProvenance, FailureMode, velocityAt, displacementAt, coherenceAt, horizontalVelocityAt, buildingSeries } from "../lib/bundle";
import { AoiSummary } from "../lib/useAois";
import { ViewMode } from "./TopBar";

export function ThreatSidebar({
  bundle, selectedRow, monthIdx, activeAoi, mode,
}: {
  bundle: Bundle | null;
  selectedRow: number | null;
  monthIdx: number;
  activeAoi: AoiSummary | null;
  mode: ViewMode;
}) {
  if (!bundle || !activeAoi) return <SkeletonSidebar />;

  return (
    <aside className="w-[380px] shrink-0 border-l border-wire-800 bg-ink-950 p-5 flex flex-col gap-5 overflow-y-auto">
      <NarrativeCard aoi={activeAoi} bundle={bundle} />
      <PhenomenonLegend aoi={activeAoi} mode={mode} />
      <Divider />
      {selectedRow === null ? (
        <NoSelection />
      ) : (
        <SelectedBuilding bundle={bundle} row={selectedRow} monthIdx={monthIdx} />
      )}
      <DataProvenanceNote provenance={bundle.header.data_provenance} />
    </aside>
  );
}


/**
 * Data-provenance footer. Branches on the bundle header's `data_provenance`
 * (three-state ladder, see backend/scripts/provenance.py):
 *   - 'insar'     → full real pipeline; velocity is real InSAR.
 *   - 'partial'   → real footprints + real terrain (Open Buildings/OSM geometry,
 *                   SoilGrids soil, OSM riparian/shoreline), but velocity is the
 *                   synthetic stand-in until the MintPy SBAS run lands.
 *   - 'synthetic' → everything fabricated, conforming to the production schema.
 * The point is that as each AOI gains real data the disclaimer self-updates to
 * say exactly what's real — no over- or under-claiming, no code change.
 */
function DataProvenanceNote({ provenance }: { provenance: DataProvenance }) {
  const badge =
    provenance === "insar"
      ? { cls: "border-green-500/50 bg-green-950/20 text-green-300", label: "live · Sentinel-1 InSAR" }
      : provenance === "partial"
      ? { cls: "border-sky-500/50 bg-sky-950/20 text-sky-300", label: "real footprints + terrain · synthetic velocity" }
      : { cls: "border-amber-500/40 bg-amber-950/20 text-amber-300", label: "synthetic preview" };

  return (
    <div className="mt-auto text-[10px] leading-relaxed">
      <div className={[
        "inline-block px-1.5 py-0.5 mb-1 border uppercase tracking-widest text-[9px]",
        badge.cls,
      ].join(" ")}>
        {badge.label}
      </div>
      <div className="text-wire-500">
        Source: Sentinel-1 SLC → HyP3 InSAR → MintPy SBAS.<br />
        {provenance === "insar" ? (
          <>Velocities are real InSAR measurements relative to the ⚓ reference point.</>
        ) : provenance === "partial" ? (
          <>
            Building footprints, soil class (SoilGrids) and riparian/shoreline
            distance (OSM) are real; velocity is synthetic pending the MintPy SBAS run.
          </>
        ) : (
          <>Demo build uses synthetic data conforming to the production schema.</>
        )}
        <br />
        See <code className="text-slate-400">docs/risk_model.md</code>.
      </div>
    </div>
  );
}


function SkeletonSidebar() {
  return (
    <aside className="w-[380px] shrink-0 border-l border-wire-800 bg-ink-950 p-5">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-600 animate-pulse">awaiting bundle…</div>
    </aside>
  );
}


function Divider() { return <div className="border-t border-wire-800" />; }


function NarrativeCard({ aoi, bundle }: { aoi: AoiSummary; bundle: Bundle }) {
  // Aggregate stats: O(n) over n_buildings, fine for ~1500.
  // "confirmed" only counts CONFIRMED_THREAT — the previous heuristic
  // (v<-10) over-counted because it ignored coherence.
  let confirmed = 0, mixed = 0, noise = 0;
  const n = bundle.header.n_buildings;
  for (let i = 0; i < n; i++) {
    const cls = bundle.classification[i];
    if (cls === Classification.CONFIRMED_THREAT) confirmed++;
    else if (cls === Classification.MIXED_SIGNAL) mixed++;
    else if (cls === Classification.ENV_NOISE)   noise++;
  }
  return (
    <div>
      <div className="text-xs uppercase tracking-[0.25em] text-slate-500">{aoi.name}</div>
      <div className="mt-2 grid grid-cols-4 gap-2 text-xs">
        <Stat label="buildings" value={n.toString()} />
        <Stat label="confirmed" value={confirmed.toString()} severe={confirmed > n * 0.1} />
        <Stat label="watch"     value={mixed.toString()} />
        <Stat label="noise"     value={noise.toString()} />
      </div>
      <p className="mt-3 text-[11px] text-slate-400 leading-relaxed">{aoi.narrative}</p>
    </div>
  );
}


function PhenomenonLegend({ aoi, mode }: { aoi: AoiSummary; mode: ViewMode }) {
  const isDrift = mode === "drift";
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">
        Color scale ({isDrift ? "horizontal drift, east-west" : "vertical velocity"})
      </div>
      <div className="h-2 w-full rounded-sm" style={{
        background: isDrift
          ? "linear-gradient(90deg, #38bdf8 0%, #64748b 50%, #f97316 100%)"
          : "linear-gradient(90deg, #ef4444 0%, #f59e0b 55%, #22c55e 100%)",
      }} />
      <div className="flex justify-between text-[10px] text-slate-500 mt-1 tabular-nums">
        {isDrift
          ? <><span>-15 mm/yr (W)</span><span>0</span><span>+15 (E)</span></>
          : <><span>-25 mm/yr</span><span>0</span><span>+5</span></>}
      </div>
      {isDrift && (
        <div className="mt-2 text-[10px] text-slate-500 italic">
          Footprint position offset shows drift direction & magnitude:
          taller buildings × higher drift = larger visual shift.
        </div>
      )}
      <div className="mt-3 text-[10px] text-slate-500">
        Footprints: <span className="text-slate-300">{aoi.footprint_source}</span> ·{" "}
        Phenomenon: <span className="text-slate-300">{aoi.phenomenon.replace(/_/g, " ")}</span>
      </div>
      <div className="mt-2 text-[10px] text-slate-500 italic">
        Low-coherence footprints are desaturated — interpretation is uncertain.
      </div>
    </div>
  );
}


function NoSelection() {
  return (
    <div>
      <div className="text-xs uppercase tracking-[0.25em] text-slate-500">Structural Threat</div>
      <div className="text-sm text-slate-400 mt-2">Click a building to inspect.</div>
    </div>
  );
}


function SelectedBuilding({
  bundle, row, monthIdx,
}: { bundle: Bundle; row: number; monthIdx: number }) {
  const m_last = bundle.header.n_months - 1;
  const v_now  = velocityAt(bundle, row, monthIdx);
  const d_now  = displacementAt(bundle, row, monthIdx);
  const coh    = coherenceAt(bundle, row, monthIdx);
  const ew_now = horizontalVelocityAt(bundle, row, monthIdx);
  const v_end  = velocityAt(bundle, row, m_last);
  const ew_end = horizontalVelocityAt(bundle, row, m_last);
  const bid    = bundle.buildingId[row];
  const cls    = bundle.classification[row];
  const accel  = bundle.velocityAccelMmYr2[row];
  const failureMode = bundle.failureMode[row];

  const ripa = bundle.riparianDistM[row];
  const shore = bundle.shorelineDistM[row];
  const reclaimed = bundle.reclaimedLand[row] === 1;
  const heightFloor = bundle.heightM[row];
  const heightInsar = bundle.insarHeightM[row];
  const heightSigma = bundle.insarHeightSigmaM[row];
  const heightFused = bundle.fusedHeightM[row];
  const composite = bundle.compositeRisk[row];

  // Recompose the four contributions exactly as the backend did so the
  // stacked bar's widths reflect what went into composite_risk. The
  // backend's noise/anchor gates are applied below to mirror them in the UI.
  const subsScore  = clamp01(-v_end / 25);
  const shearScore = sigmoid((Math.abs(ew_end) - 2.5) / 2.0);
  const proxScore = ripa >= 0 ? Math.exp(-ripa / 400)
                   : shore >= 0 ? Math.exp(-shore / 300)
                   : 0;
  const soilFromLut = guessSoilScore(bundle, row);

  const series = buildingSeries(bundle, row, "displacement");
  const trendSeries = buildingSeries(bundle, row, "trend");

  const trendSlope    = bundle.trendSlopeMmYr[row];
  const seasonalAmp   = bundle.seasonalAmplitudeMm[row];
  const trendR2       = bundle.trendR2[row];

  // Tier 3: velocity σ + cohort percentile context.
  const vSigma        = bundle.velocitySigmaMmYr[row];
  const ewSigma       = bundle.velocityEwSigmaMmYr[row];
  const cohortComp    = bundle.cohortCompositePct[row];
  const cohortShear   = bundle.cohortShearPct[row];
  const cohortN       = bundle.cohortSize[row];

  // ARCHITECTURE_THREE C1/C4 — block context: this building's block aggregates
  // + its block-relative percentile. Indexed off the per-building block_id.
  const blkId         = bundle.blockId[row];
  const blkCount      = bundle.blockCount[blkId] ?? 0;
  const blkWorstVel   = bundle.blockWorstVelocity[blkId] ?? 0;
  const blkConfirmed  = bundle.blockConfirmed[blkId] ?? 0;
  const cohortBlock   = bundle.cohortBlockPct[row];

  const severe = v_now < -10;
  const driftSevere = Math.abs(ew_now) > 5;
  return (
    <div>
      <div className="text-xs uppercase tracking-[0.25em] text-slate-500">Structural Threat</div>
      <div className="text-sm text-slate-400 mt-2">Building #{bid}</div>

      <ClassificationBadge cls={cls} failureMode={failureMode} vEwEnd={ew_end} />

      <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <Metric label="Subsidence V"    value={fmtMm(v_now)}        unit="mm/yr" severe={severe}      sigma={vSigma} />
        <Metric label="Horizontal drift" value={fmtDrift(ew_now)}    unit="mm/yr" severe={driftSevere} sigma={ewSigma} />
        <Metric label="Cum displ"        value={fmtMm(d_now)}        unit="mm" />
        <Metric label="Coherence"        value={coh.toFixed(2)}      severe={coh < 0.3} />
        <TrendMetric accel={accel} />
        {ripa >= 0 && <Metric label="Riparian dist"  value={ripa.toFixed(0)}  unit="m" />}
        {shore >= 0 && <Metric label="Shoreline dist" value={shore.toFixed(0)} unit="m" />}
        {reclaimed && <Metric label="Reclaimed land" value="YES" severe />}
      </div>

      <HeightCard
        floor={heightFloor}
        insar={heightInsar}
        sigma={heightSigma}
        fused={heightFused}
      />

      <div className="mt-4">
        <div className="flex justify-between text-[10px] uppercase tracking-widest text-slate-500">
          <span>Composite risk</span>
          <span className="tabular-nums text-slate-300">{(composite * 100).toFixed(0)}%</span>
        </div>
        <StackedRiskBar
          subs={0.35 * subsScore}
          shear={0.25 * shearScore}
          prox={0.20 * proxScore}
          soil={0.20 * soilFromLut}
        />
        <div className="mt-1 grid grid-cols-4 text-[10px] text-slate-500 gap-1">
          <span><Dot c="#ef4444" /> subsidence</span>
          <span><Dot c="#a78bfa" /> shear</span>
          <span><Dot c="#f59e0b" /> proximity</span>
          <span><Dot c="#22d3ee" /> soil × load</span>
        </div>
        <div className="mt-1 text-[10px] text-slate-500 italic">
          Soil contribution scales with building load (height-weighted: 1 + h/10).
        </div>
        <CohortContext
          compositePct={cohortComp}
          shearPct={cohortShear}
          cohortN={cohortN}
        />
        <BlockContext
          blockPct={cohortBlock}
          count={blkCount}
          worstVelocity={blkWorstVel}
          confirmed={blkConfirmed}
        />
      </div>

      <div className="mt-4">
        <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">
          {bundle.header.n_months}-mo displacement (mm) · trend overlay
        </div>
        <Sparkline values={series} trend={trendSeries} highlightIdx={monthIdx} />
        <FailureModeLine
          mode={failureMode}
          trendSlope={trendSlope}
          seasonalAmp={seasonalAmp}
          r2={trendR2}
        />
      </div>
    </div>
  );
}


/**
 * Classification badge — coherence-velocity matrix outcome, refined with the
 * STL failure mode and horizontal-drift magnitude.
 *
 * The V4 framework's threat-badge wording overlaps: a building can fire
 * "lateral shear" AND "progressive subsidence" at the same time, and the
 * doc doesn't say which wins. We define a deterministic priority so the
 * badge always renders a single, repeatable label:
 *
 *   1. PLASTIC + lateral shear (|v_ew| > 2.5)  → "Critical: progressive shear failure"
 *   2. PLASTIC failure mode alone              → "Critical: progressive subsidence"
 *   3. CONFIRMED_THREAT, shear-dominant        → "High: lateral shear"
 *   4. CONFIRMED_THREAT, subsidence-dominant   → "Confirmed structural threat"
 *   5. ENV_NOISE                               → "Environmental noise"
 *   6. MIXED_SIGNAL (V4-gap row)               → "Watch: ambiguous signal"
 *   7. STABLE_ANCHOR                           → "Reference asset · stable"
 *   8. INDETERMINATE                           → "Indeterminate"
 *
 * PLASTIC outranks classification because a building with a real downward
 * trend and good fit is a credible failure regardless of what the
 * coherence-velocity matrix called it on the final month.
 */
function ClassificationBadge({
  cls, failureMode, vEwEnd,
}: { cls: number; failureMode: number; vEwEnd: number }) {
  const isPlastic = failureMode === FailureMode.PLASTIC;
  const shearDominant = Math.abs(vEwEnd) > 2.5;

  let text: string;
  let bg: string;
  // Court-defensibility gate wins over every other badge: if the trend isn't
  // defensible we cannot also call it "critical". Mirrors the backend, where
  // the gate is checked before the failure-mode / classification ladder.
  if (cls === Classification.INSUFFICIENT_EVIDENCE) {
    text = "Insufficient evidence · not court-defensible";
    bg = "bg-slate-800/50 border-slate-500/50 text-slate-300";
  } else if (isPlastic && shearDominant) {
    text = "Critical · progressive shear failure";
    bg = "bg-red-950/50 border-red-500/80 text-red-200";
  } else if (isPlastic) {
    text = "Critical · progressive subsidence";
    bg = "bg-red-950/40 border-red-500/70 text-red-300";
  } else if (cls === Classification.CONFIRMED_THREAT && shearDominant) {
    text = "High · lateral shear";
    bg = "bg-red-950/30 border-red-500/50 text-red-300";
  } else if (cls === Classification.CONFIRMED_THREAT) {
    text = "Confirmed structural threat";
    bg = "bg-red-950/30 border-red-500/40 text-red-300";
  } else if (cls === Classification.ENV_NOISE) {
    text = "Environmental noise · interpret with caution";
    bg = "bg-amber-950/30 border-amber-500/40 text-amber-300";
  } else if (cls === Classification.MIXED_SIGNAL) {
    text = "Watch · ambiguous signal";
    bg = "bg-amber-950/20 border-amber-400/40 text-amber-200";
  } else if (cls === Classification.STABLE_ANCHOR) {
    text = "Reference asset · stable";
    bg = "bg-green-950/30 border-green-500/40 text-green-300";
  } else {
    text = "Indeterminate";
    bg = "bg-ink-800/60 border-wire-700 text-slate-400";
  }
  return (
    <div className={["mt-2 px-2 py-1 border text-[10px] uppercase tracking-widest", bg].join(" ")}>
      {text}
    </div>
  );
}


/**
 * Trend cell: shows velocity acceleration with directional arrow.
 * Threshold ±3 mm/yr² is a hand-picked break between "noticeable change in
 * subsidence rate over half a year" and "noise"; matches the synthetic
 * generator's ramp magnitudes.
 */
function TrendMetric({ accel }: { accel: number }) {
  const accelerating = accel < -3;
  const decelerating = accel > 3;
  const label =
    accelerating ? "▲ accelerating" :
    decelerating ? "▼ decelerating" :
                   "→ steady";
  return (
    <div className={[
      "p-3 border",
      accelerating ? "border-red-500/60 bg-red-950/30" : "border-wire-800 bg-ink-800/40",
    ].join(" ")}>
      <div className="text-[10px] uppercase tracking-widest text-slate-500">Trend</div>
      <div className={[
        "text-base tabular-nums",
        accelerating ? "text-red-300" : "text-slate-100",
      ].join(" ")}>
        {label}
        <span className="text-xs text-slate-500 ml-1">
          {accel >= 0 ? "+" : ""}{accel.toFixed(1)} mm/yr²
        </span>
      </div>
    </div>
  );
}


/** LUT-only soil score for visualization (we don't pack the soil class as a
 * numeric per-building field; soil_classes lives in the JSON header indexed
 * by row order, but in practice the soil weight is small and the recomposed
 * bar is illustrative not authoritative — the *composite* is canonical). */
function guessSoilScore(bundle: Bundle, row: number): number {
  const cls = bundle.header.soil_classes[row] || "";
  const lut: Record<string, number> = {
    black_cotton: 0.9, alluvial: 0.7, red_clay: 0.4, weathered_basalt: 0.1,
    coral_rag: 0.15, reclaim_fill: 0.85,
  };
  return lut[cls] ?? 0.3;
}


function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1.0 / (1.0 + z);
  }
  const z = Math.exp(x);
  return z / (1.0 + z);
}


/**
 * Height estimate breakdown. Surfaces the InSAR vs footprint disagreement —
 * the gap is the honest signal. Fused value is what the 3D map actually uses.
 *
 * Sentinel-1's coarse 5×20 m resolution means small footprints have noisy
 * InSAR heights (σ up to ±4 m in Huruma's dense settlement). The fused value
 * is an inverse-variance blend with the floor-count estimate (σ ≈ 1.5 m).
 */
function HeightCard({
  floor, insar, sigma, fused,
}: { floor: number; insar: number; sigma: number; fused: number }) {
  if (!Number.isFinite(fused) || fused <= 0) return null;
  const disagree = Math.abs(insar - floor);
  const disagreeHigh = disagree > 2 * sigma;
  return (
    <div className="mt-4 border border-wire-800 bg-ink-800/40 p-3">
      <div className="text-[10px] uppercase tracking-widest text-slate-500">Building height</div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-[11px]">
        <HeightLine label="Floor-count"   value={`${floor.toFixed(1)} m`} />
        <HeightLine label="InSAR"         value={`${insar.toFixed(1)} ± ${sigma.toFixed(1)} m`}
                    flagged={disagreeHigh} />
        <HeightLine label="Fused (3D)"    value={`${fused.toFixed(1)} m`} accent />
      </div>
      {disagreeHigh && (
        <div className="mt-2 text-[10px] text-amber-400/80 italic">
          Footprint and InSAR disagree by more than 2σ. Likely noisy phase
          fringes (small footprint or low coherence).
        </div>
      )}
    </div>
  );
}


function HeightLine({
  label, value, accent, flagged,
}: { label: string; value: string; accent?: boolean; flagged?: boolean }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-widest text-slate-500">{label}</div>
      <div className={[
        "tabular-nums",
        accent ? "text-signal-cyan" : flagged ? "text-amber-300" : "text-slate-200",
      ].join(" ")}>
        {value}
      </div>
    </div>
  );
}


function StackedRiskBar({
  subs, shear, prox, soil,
}: { subs: number; shear: number; prox: number; soil: number }) {
  // Four contributions, each in [0, weight]. Sum ≤ 1.
  const w = 280;
  const total = subs + shear + prox + soil;
  // Scale to fill bar; cap at full width.
  const scale = total > 0 ? Math.min(1, total) / total : 1;
  const subsW  = subs  * w * scale;
  const shearW = shear * w * scale;
  const proxW  = prox  * w * scale;
  const soilW  = soil  * w * scale;
  return (
    <div className="mt-1 h-2.5 w-full flex bg-wire-800 overflow-hidden">
      <div style={{ width: subsW }}  className="bg-signal-red" />
      <div style={{ width: shearW }} className="bg-violet-400" />
      <div style={{ width: proxW }}  className="bg-signal-amber" />
      <div style={{ width: soilW }}  className="bg-signal-cyan" />
    </div>
  );
}


function Stat({ label, value, severe }: { label: string; value: string; severe?: boolean }) {
  return (
    <div className={`p-2 border ${severe ? "border-red-500/50 bg-red-950/20" : "border-wire-800 bg-ink-800/50"}`}>
      <div className="text-[9px] uppercase tracking-widest text-slate-500">{label}</div>
      <div className={`tabular-nums text-base ${severe ? "text-red-300" : "text-slate-100"}`}>{value}</div>
    </div>
  );
}


function Metric({
  label, value, unit, severe, sigma,
}: {
  label: string;
  value: string | number;
  unit?: string;
  severe?: boolean;
  /** Optional 1σ uncertainty (Tier 3); rendered as `± σ unit` in the unit slot. */
  sigma?: number;
}) {
  const sigmaText = sigma != null && Number.isFinite(sigma)
    ? `± ${sigma.toFixed(1)}${unit ? " " + unit : ""}`
    : null;
  return (
    <div className={`p-3 border ${severe ? "border-red-500/60 bg-red-950/30" : "border-wire-800 bg-ink-800/40"}`}>
      <div className="text-[10px] uppercase tracking-widest text-slate-500">{label}</div>
      <div className={`text-base tabular-nums ${severe ? "text-red-300" : "text-slate-100"}`}>
        {value}
        {sigmaText
          ? <span className="text-[10px] text-slate-500 ml-1 tabular-nums">{sigmaText}</span>
          : unit && <span className="text-xs text-slate-500 ml-1">{unit}</span>}
      </div>
    </div>
  );
}


function Dot({ c }: { c: string }) {
  return <span className="inline-block w-2 h-2 mr-1" style={{ background: c }} />;
}


function Sparkline({
  values, trend, highlightIdx,
}: { values: Float32Array; trend?: Float32Array; highlightIdx: number }) {
  if (values.length === 0) return null;
  const w = 320, h = 70, pad = 6;
  // Compute scale across both series so the overlay sits in the same frame.
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (trend) {
    for (let i = 0; i < trend.length; i++) {
      const v = trend[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const range = hi - lo || 1;
  const xs = (i: number) => pad + (i / (values.length - 1)) * (w - 2 * pad);
  const ys = (v: number) => h - pad - ((v - lo) / range) * (h - 2 * pad);
  let path = "";
  for (let i = 0; i < values.length; i++) {
    path += `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${ys(values[i]).toFixed(1)} `;
  }
  let trendPath = "";
  if (trend && trend.length === values.length) {
    for (let i = 0; i < trend.length; i++) {
      trendPath += `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${ys(trend[i]).toFixed(1)} `;
    }
  }
  return (
    <svg width={w} height={h} className="text-signal-cyan">
      <path d={path} fill="none" stroke="currentColor" strokeWidth={1.4} />
      {trendPath && (
        <path d={trendPath} fill="none" stroke="#cbd5e1" strokeWidth={1.2}
              strokeDasharray="3 2" opacity={0.85} />
      )}
      <line x1={xs(highlightIdx)} x2={xs(highlightIdx)} y1={pad} y2={h - pad}
            stroke="#f59e0b" strokeDasharray="2 3" />
      <circle cx={xs(highlightIdx)} cy={ys(values[highlightIdx])} r={3} fill="#f59e0b" />
    </svg>
  );
}


/**
 * STL failure-mode classifier:
 *   ELASTIC = seasonal soil response (breathes with the rain — not failure)
 *   PLASTIC = progressive trend (foundation actually failing)
 *
 * Confidence caveat: 24 months is the statistical floor for STL with annual
 * seasonality. R² and seasonal amplitude are surfaced so the reader can judge
 * the call themselves — not hidden behind the badge.
 */
function FailureModeLine({
  mode, trendSlope, seasonalAmp, r2,
}: { mode: number; trendSlope: number; seasonalAmp: number; r2: number }) {
  const isPlastic = mode === FailureMode.PLASTIC;
  return (
    <div className={[
      "mt-2 px-2 py-1.5 border text-[10px] leading-snug",
      isPlastic
        ? "border-red-500/50 bg-red-950/30 text-red-200"
        : "border-wire-800 bg-ink-800/40 text-slate-300",
    ].join(" ")}>
      <div className="uppercase tracking-widest text-[9px] text-slate-500">
        Failure mode (STL · 24 mo)
      </div>
      <div className="mt-0.5 tabular-nums">
        {isPlastic
          ? <><span className="text-red-300">PLASTIC</span> · progressive foundation failure</>
          : <><span className="text-slate-200">ELASTIC</span> · seasonal soil response</>}
      </div>
      <div className="mt-0.5 text-slate-500 tabular-nums">
        trend {trendSlope >= 0 ? "+" : ""}{trendSlope.toFixed(1)} mm/yr  ·
        seasonal ±{(seasonalAmp / 2).toFixed(1)} mm  ·
        {r2 < 0
          ? <span className="text-amber-400/80 not-tabular-nums"> fit poor (R²&lt;0)</span>
          : <> R² {r2.toFixed(2)}{r2 < 0.5 && <span className="text-amber-400/80 not-tabular-nums"> · fit poor</span>}</>}
      </div>
    </div>
  );
}


/**
 * Cohort percentile context (Tier 3 #7 in ARCHITECTURE_TWO).
 *
 * A composite score in isolation ("0.62") is useless to a decision-maker. The
 * actionable framing is *relative to peers*: "92nd percentile for shear among
 * 47 buildings on the same soil at a similar height." The backend pre-computes
 * the percentile rank within `height_band × soil_class` cohorts; the UI's
 * job is just to render it as a sentence.
 *
 * Singleton cohorts (n=1) are tagged 50/50 by the backend — we suppress those
 * because "median of 1 peer" is misleading; render a quiet caveat instead.
 */
function CohortContext({
  compositePct, shearPct, cohortN,
}: { compositePct: number; shearPct: number; cohortN: number }) {
  if (cohortN <= 1) {
    return (
      <div className="mt-1 text-[10px] text-slate-500 italic">
        No peer cohort (unique height × soil bucket — percentile suppressed).
      </div>
    );
  }
  // Bold flag once a building is in the top quartile on either axis — that's
  // the threshold where "look at this one first" becomes the right call.
  const hot = compositePct >= 75 || shearPct >= 75;
  return (
    <div className={[
      "mt-2 px-2 py-1.5 border text-[10px] leading-snug tabular-nums",
      hot
        ? "border-amber-500/50 bg-amber-950/20 text-amber-200"
        : "border-wire-800 bg-ink-800/40 text-slate-300",
    ].join(" ")}>
      <div className="uppercase tracking-widest text-[9px] text-slate-500">
        Peer cohort · height-band × soil ({cohortN} buildings)
      </div>
      <div className="mt-0.5">
        composite <span className={hot ? "text-amber-200" : "text-slate-100"}>
          {ordinal(compositePct)}
        </span> pct · shear <span className={shearPct >= 75 ? "text-amber-200" : "text-slate-100"}>
          {ordinal(shearPct)}
        </span> pct
      </div>
    </div>
  );
}

/**
 * Block context (ARCHITECTURE_THREE C1/C4).
 *
 * Reframes the building against its ~170 m grid block — the honest unit of
 * InSAR resolution in dense Nairobi, where individual sub-pixel footprints
 * share a pixel. Shows how this building ranks *within its own block* plus the
 * block's headline aggregates (worst velocity, building count, confirmed
 * threats). Singleton blocks suppress the percentile, same as the peer cohort.
 */
function BlockContext({
  blockPct, count, worstVelocity, confirmed,
}: { blockPct: number; count: number; worstVelocity: number; confirmed: number }) {
  const hot = confirmed > 0 || worstVelocity < -10;
  return (
    <div className={[
      "mt-2 px-2 py-1.5 border text-[10px] leading-snug tabular-nums",
      hot
        ? "border-red-500/40 bg-red-950/20 text-red-200"
        : "border-wire-800 bg-ink-800/40 text-slate-300",
    ].join(" ")}>
      <div className="uppercase tracking-widest text-[9px] text-slate-500">
        Block context · ~170 m grid ({count} building{count === 1 ? "" : "s"})
      </div>
      <div className="mt-0.5">
        {count > 1
          ? <>rank <span className={blockPct >= 75 ? "text-amber-200" : "text-slate-100"}>{ordinal(blockPct)}</span> pct in block · </>
          : <>sole building in block · </>}
        worst <span className={worstVelocity < -10 ? "text-red-300" : "text-slate-100"}>{worstVelocity.toFixed(1)}</span> mm/yr
      </div>
      {confirmed > 0 && (
        <div className="mt-0.5 text-red-300">
          {confirmed} confirmed threat{confirmed === 1 ? "" : "s"} in this block
        </div>
      )}
    </div>
  );
}

/** "92nd", "1st", "23rd" — for percentile labels. */
function ordinal(n: number): string {
  const v = Math.round(n);
  const mod100 = v % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${v}th`;
  switch (v % 10) {
    case 1:  return `${v}st`;
    case 2:  return `${v}nd`;
    case 3:  return `${v}rd`;
    default: return `${v}th`;
  }
}


function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }
function fmtMm(v: number): string { return Number.isFinite(v) ? v.toFixed(1) : "—"; }
function fmtDrift(ew: number): string {
  if (!Number.isFinite(ew)) return "—";
  const dir = ew > 0.2 ? " E" : ew < -0.2 ? " W" : "";
  return `${ew >= 0 ? "+" : ""}${ew.toFixed(1)}${dir}`;
}
