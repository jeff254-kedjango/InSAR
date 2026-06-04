/**
 * RiskMap — single-screen dashboard.
 *
 * Performance model:
 *   - One bundle fetch per AOI per session (cached). Slider/play/click never
 *     refetch.
 *   - Geometry is uploaded to the GPU once per AOI via deck.gl's binary data
 *     path (no per-feature objects).
 *   - Slider tick = one ref update + one `updateTriggers` flip. The color
 *     accessor reads from a typed array (O(1) per polygon, no allocations).
 *   - Building click → O(1) index lookup via `byBuildingId` Map.
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { Map as MlMap } from "maplibre-gl";
import { Protocol } from "pmtiles";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { SolidPolygonLayer, PolygonLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import "maplibre-gl/dist/maplibre-gl.css";

import { Bundle, velocityAt, displacementAt, coherenceAt, horizontalVelocityAt, buildingSeries, blockPolygon } from "../lib/bundle";
import { ApiError, AoiSummary, useAoiBundle, useAoiRegistry } from "../lib/useAois";
import { ThreatSidebar } from "./ThreatSidebar";
import { TopBar, ViewMode } from "./TopBar";
import { TimeSlider } from "./TimeSlider";

const VELOCITY_FLOOR_MM_YR = -25; // subsidence ramp: map to most-red
const VELOCITY_CEIL_MM_YR  =  +5; // subsidence ramp: map to green
const EW_VELOCITY_BOUND    = 15;  // drift ramp: ±15 mm/yr saturates the ramp
const ELEVATION_GAIN = 1.0;       // true scale — buildings extrude at their real
                                  // measured height (Google Open Buildings), so the
                                  // 3D matches satellite/3D basemaps 1:1.

/**
 * Drift visualization: in Drift mode, each building's rendered footprint is
 * translated east-west by an amount proportional to (fused_height × ew_velocity).
 * This is *not* a literal physical projection — SolidPolygonLayer extrudes
 * straight up and can't tilt — but it conveys "structural displacement direction
 * and magnitude" at a glance. Buildings under heavy westward drift jump west
 * relative to their footprint; the gap is the visual cue.
 *
 * The scalar below is tuned so a 10 m tall building with ±10 mm/yr drift
 * shifts ~3 meters on the map at zoom 15.5. Adjust to taste.
 */
const DRIFT_VISUAL_GAIN_M_PER_MM_PER_M = 0.03;

// Selected-building highlight. Full-opacity signal-red (#ef4444, the design-token
// red from tailwind.config.js), returned BEFORE the velocity ramp + coherence
// fade so a clicked building is always solid red regardless of its measured
// velocity/coherence. Paired with a white outline layer (see selectionOutline)
// so it can never be mistaken for an unselected critical (ramp-red) building.
const SELECTED_RGBA: [number, number, number, number] = [239, 68, 68, 255];


export function RiskMap() {
  const { aois, error: regErr } = useAoiRegistry();
  const [activeCode, setActiveCode] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("subsidence");

  // Seed activeCode once we have the registry.
  useEffect(() => {
    if (!activeCode && aois && aois.length) setActiveCode(aois[0].aoi_code);
  }, [activeCode, aois]);

  const { bundle, error: bundleErr } = useAoiBundle(activeCode);
  const activeAoi = aois?.find(a => a.aoi_code === activeCode) ?? null;

  if (regErr) return <ErrorPanel err={regErr} where="/aois" />;
  if (bundleErr) return <ErrorPanel err={bundleErr} where={`bundle for ${activeCode}`} />;

  return (
    <div className="h-screen w-screen flex bg-ink-950 text-slate-200 font-mono select-none">
      <MapPane
        aois={aois}
        activeCode={activeCode}
        setActiveCode={setActiveCode}
        bundle={bundle}
        activeAoi={activeAoi}
        mode={mode}
        onModeChange={setMode}
      />
    </div>
  );
}


/**
 * Icon-only map zoom control. Stacked +/− buttons that drive MapLibre's native
 * `zoomIn`/`zoomOut` (smooth ±1 step, camera-preserving; the deck.gl overlay
 * re-syncs automatically). `React.memo` + stable `useCallback` handlers mean
 * this never reconciles during slider scrubbing or playback ticks — the parent
 * re-renders on every animation frame, and this subtree is skipped entirely.
 */
const ZoomControl = memo(function ZoomControl({
  onZoomIn, onZoomOut,
}: {
  onZoomIn: () => void;
  onZoomOut: () => void;
}) {
  // Contrast strategy: a SOLID (non-translucent) ink chip so the control never
  // washes out over a light basemap, a bright wire-500 outline + brighter
  // slate-100 glyphs for legibility on dark canvas, and a black ring/shadow
  // halo to separate the chip edge from a light map. Hover keeps the shared
  // signal-cyan accent used by the other map buttons (TopBar/TimeSlider).
  const btn =
    "w-8 h-8 grid place-items-center border border-wire-500 bg-ink-900 " +
    "text-slate-100 hover:bg-ink-800 hover:border-signal-cyan hover:text-signal-cyan " +
    "transition";
  return (
    <div className="absolute bottom-24 right-4 pointer-events-auto flex flex-col gap-px ring-1 ring-black/40 shadow-lg shadow-black/40">
      <button type="button" aria-label="Zoom in" onClick={onZoomIn} className={btn}>
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
          <path d="M7 2v10M2 7h10" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" />
        </svg>
      </button>
      <button type="button" aria-label="Zoom out" onClick={onZoomOut} className={btn}>
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
          <path d="M2 7h10" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
});


function MapPane({
  aois, activeCode, setActiveCode, bundle, activeAoi, mode, onModeChange,
}: {
  aois: AoiSummary[] | null;
  activeCode: string | null;
  setActiveCode: (c: string) => void;
  bundle: Bundle | null;
  activeAoi: AoiSummary | null;
  mode: ViewMode;
  onModeChange: (m: ViewMode) => void;
}) {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MlMap | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  /** Current month index. Lives in a ref so slider scrubbing is allocation-free. */
  const monthIdxRef = useRef(0);
  const [monthIdx, setMonthIdxState] = useState(0); // mirror for React-driven UI
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [showBlocks, setShowBlocks] = useState(false);  // C3 — block overlay toggle

  /* ---- Map init (once) -------------------------------------------------- */
  useEffect(() => {
    if (!mapContainer.current) return;

    const protocol = new Protocol();
    maplibregl.addProtocol("pmtiles", protocol.tile);

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: emptyDarkStyle(),
      center: [36.874, -1.251],
      zoom: 15.5,
      pitch: 55,
      bearing: -20,
      attributionControl: false,
    });
    mapRef.current = map;

    const overlay = new MapboxOverlay({ interleaved: false, layers: [] });
    overlayRef.current = overlay;
    map.addControl(overlay as unknown as maplibregl.IControl);

    // Probe for an installed PMTiles basemap. If present, upgrade the style
    // from the flat dark canvas to the full Protomaps schema.
    //
    // Why not just `if (res.ok)`: the Vite dev server (and most SPA hosts)
    // returns index.html with status 200 for missing static files so that
    // client-side routing can take over. A bare `res.ok` check therefore
    // happily tries to install a PMTiles source on top of an HTML body,
    // and MapLibre throws "Wrong magic number for PMTiles archive".
    //
    // Fix: read the first 7 bytes and verify the actual PMTiles magic
    // string. Tiny range-request — costs nothing if the file is real, and
    // correctly bails out when the response is the SPA fallback.
    (async () => {
      try {
        const res = await fetch("/tiles/nairobi.pmtiles", {
          headers: { Range: "bytes=0-6" },
        });
        if (!res.ok) return;
        const head = new Uint8Array(await res.arrayBuffer());
        // ASCII "PMTiles" = 0x50 0x4D 0x54 0x69 0x6C 0x65 0x73
        const MAGIC = [0x50, 0x4d, 0x54, 0x69, 0x6c, 0x65, 0x73];
        if (head.length < 7 || !MAGIC.every((b, i) => head[i] === b)) return;
        map.setStyle(protomapsDarkStyle());
      } catch {
        /* offline / no tiles installed — keep the empty dark canvas */
      }
    })();

    return () => {
      map.remove();
      maplibregl.removeProtocol("pmtiles");
    };
  }, []);

  /* ---- Recenter when AOI changes --------------------------------------- */
  useEffect(() => {
    if (!mapRef.current || !activeAoi) return;
    mapRef.current.flyTo({
      center: [activeAoi.center_lon, activeAoi.center_lat],
      zoom: 15.5, pitch: 55, bearing: -20, duration: 1200,
    });
    setSelectedRow(null);
    monthIdxRef.current = 0;
    setMonthIdxState(0);
  }, [activeAoi?.aoi_code]);

  /* ---- Build deck.gl binary data once per bundle -----------------------
   *
   * SolidPolygonLayer in binary mode wants:
   *   data: {
   *     length: <polygon count>,
   *     startIndices: Int32Array of length (n_polygons + 1), in VERTEX units,
   *     attributes: {
   *       getPolygon:   { value: Float32Array<lon,lat,...>, size: 2 },
   *       getElevation: { value: Float32Array of length n_vertices_total },  // per-VERTEX
   *       getFillColor: { value: Uint8Array of length n_vertices_total*4, size: 4 },
   *     }
   *   }
   *
   * Critical and easy to get wrong: per-vertex attributes must be sized to
   * TOTAL VERTEX COUNT, not polygon count. The previous code passed callback
   * accessors against binary data, which is undocumented and was rendering
   * nothing — that's why "Play" advanced the date but the screen stayed dark.
   * ------------------------------------------------------------------------ */
  const staticBinary = useMemo(() => {
    if (!bundle) return null;
    const n_polys    = bundle.header.n_buildings;
    const n_vertices = bundle.ringCoords.length / 2;   // (lon,lat) pairs

    // startIndices is in vertex units. ringOffsets is in lon-lat-float units.
    const startIndices = new Int32Array(bundle.ringOffsets.length);
    for (let i = 0; i < bundle.ringOffsets.length; i++) {
      startIndices[i] = bundle.ringOffsets[i] >> 1;   // /2
    }

    // Per-vertex elevation: each vertex of polygon i extruded to fused_height.
    // Fused = inverse-variance blend of floor-count estimate and InSAR phase
    // inversion — see backend/ARCHITECTURE_ONE.md, study 1.
    const elevation = new Float32Array(n_vertices);
    for (let i = 0; i < n_polys; i++) {
      const start = startIndices[i];
      const end   = startIndices[i + 1];
      const fused = bundle.fusedHeightM[i];
      const h = ((Number.isFinite(fused) && fused > 0 ? fused : (bundle.heightM[i] || 6))
                 * ELEVATION_GAIN);
      for (let v = start; v < end; v++) elevation[v] = h;
    }

    // Drift-skewed ringCoords: same length/shape as bundle.ringCoords, but
    // each vertex of polygon i is shifted east-west by an amount proportional
    // to (fused_height × ew_velocity_at_current_month × GAIN). Written by the
    // mode/month effect below; the actual values are stale here, which is OK
    // because the effect repopulates it before the layer reads it.
    const driftCoords = new Float32Array(bundle.ringCoords.length);
    driftCoords.set(bundle.ringCoords);

    // Per-vertex color buffer; values written by the month-tick effect below.
    const fillColor = new Uint8Array(n_vertices * 4);

    return { n_polys, n_vertices, startIndices, elevation, fillColor, driftCoords };
  }, [bundle]);

  /* ---- C1/C3 — block overlay polygons, precomputed once per bundle --------
   * One PolygonLayer feature per *non-empty* block. Each carries its grid-cell
   * ring plus the aggregate metrics, so deck.gl colors and the tooltip read
   * straight off the object. Empty blocks are dropped (no feature emitted).
   * ------------------------------------------------------------------------ */
  const blockData = useMemo(() => {
    if (!bundle) return [];
    const n = bundle.header.block_grid.n_blocks;
    const out: {
      polygon: [number, number][];
      worstVelocity: number;
      meanRisk: number;
      maxRisk: number;
      count: number;
      confirmed: number;
    }[] = [];
    for (let b = 0; b < n; b++) {
      const count = bundle.blockCount[b];
      if (count <= 0) continue;
      out.push({
        polygon: blockPolygon(bundle, b),
        worstVelocity: bundle.blockWorstVelocity[b],
        meanRisk: bundle.blockMeanRisk[b],
        maxRisk: bundle.blockMaxRisk[b],
        count,
        confirmed: bundle.blockConfirmed[b],
      });
    }
    return out;
  }, [bundle]);

  /* ---- Repaint the per-vertex color buffer when month, mode, or selection
   * changes. O(n_buildings) writes per tick, zero allocations. The same
   * Uint8Array is mutated in place and re-uploaded via updateTriggers.
   *
   * Mode dispatch is at the per-building level: each building's color is
   * driven by `velocity_mm_yr` (subsidence ramp) OR `velocity_horizontal_ew`
   * (drift ramp). Coherence desaturation applies to both ramps.
   * ------------------------------------------------------------------------ */
  const [colorVersion, setColorVersion] = useState(0);
  useEffect(() => {
    if (!bundle || !staticBinary) return;
    const { n_polys, startIndices, fillColor } = staticBinary;
    const m = monthIdx;
    for (let i = 0; i < n_polys; i++) {
      const coh = coherenceAt(bundle, i, m);
      const sel = i === selectedRow;
      let r: number, g: number, b: number, a: number;
      if (mode === "subsidence") {
        const v = velocityAt(bundle, i, m);
        [r, g, b, a] = velocityToRGBA(v, coh, sel);
      } else {
        const ew = horizontalVelocityAt(bundle, i, m);
        [r, g, b, a] = driftToRGBA(ew, coh, sel);
      }
      const start = startIndices[i] * 4;
      const end   = startIndices[i + 1] * 4;
      for (let off = start; off < end; off += 4) {
        fillColor[off]     = r;
        fillColor[off + 1] = g;
        fillColor[off + 2] = b;
        fillColor[off + 3] = a;
      }
    }
    setColorVersion(v => v + 1);
  }, [bundle, staticBinary, monthIdx, selectedRow, mode]);

  /* ---- Rebuild drift-skewed ringCoords when month or mode changes ---------
   * Subsidence mode: zero offset, drift coords identity-copy of upright.
   * Drift mode: offset each polygon's vertices by
   *     offset_m = fused_height × ew_velocity × VISUAL_GAIN
   * converted to lon-degrees at the polygon's latitude.
   *
   * O(total_vertices), one allocation-free pass. Could be skipped in
   * subsidence mode (driftCoords unused) but keeping it cheap and unconditional
   * means the layer never reads stale data after a mode toggle.
   * ------------------------------------------------------------------------ */
  const [driftVersion, setDriftVersion] = useState(0);
  useEffect(() => {
    if (!bundle || !staticBinary) return;
    const { n_polys, startIndices, driftCoords } = staticBinary;
    const src = bundle.ringCoords;
    if (mode === "subsidence") {
      driftCoords.set(src);
      setDriftVersion(v => v + 1);
      return;
    }
    const m = monthIdx;
    for (let i = 0; i < n_polys; i++) {
      const start = startIndices[i];
      const end   = startIndices[i + 1];
      const fused = bundle.fusedHeightM[i];
      const h_m   = Number.isFinite(fused) && fused > 0 ? fused : (bundle.heightM[i] || 6);
      const ew    = horizontalVelocityAt(bundle, i, m);
      // Offset in meters; convert to longitude degrees at this building's latitude.
      const offset_m = h_m * ew * DRIFT_VISUAL_GAIN_M_PER_MM_PER_M;
      // Use the first vertex's latitude as the reference (footprint is small).
      const lat0 = src[start * 2 + 1];
      const cosLat = Math.cos(lat0 * Math.PI / 180);
      const dLon = offset_m / (111_320.0 * (cosLat || 1));
      for (let v = start; v < end; v++) {
        driftCoords[v * 2]     = src[v * 2] + dLon;
        driftCoords[v * 2 + 1] = src[v * 2 + 1];
      }
    }
    setDriftVersion(v => v + 1);
  }, [bundle, staticBinary, monthIdx, mode]);

  /* ---- Selected-building outline ring ----------------------------------
   * One-feature polygon: the selected building's ring, read straight off the
   * (possibly drift-skewed) vertex buffer. O(vertices of one building) ≈ O(1),
   * rebuilt only when the selection or the drift skew changes. Empty array when
   * nothing is selected → the outline layer renders nothing.
   * ------------------------------------------------------------------------ */
  const selectionOutline = useMemo(() => {
    if (!bundle || !staticBinary || selectedRow == null) return [];
    const { startIndices, driftCoords, elevation } = staticBinary;
    const s = startIndices[selectedRow];
    const e = startIndices[selectedRow + 1];
    const ring: [number, number][] = [];
    for (let v = s; v < e; v++) ring.push([driftCoords[v * 2], driftCoords[v * 2 + 1]]);
    return [{ polygon: ring, height: elevation[s] }];
    // driftVersion: ring tracks the east-west skew applied to driftCoords.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bundle, staticBinary, selectedRow, driftVersion]);

  /* ---- deck.gl layer --------------------------------------------------- */
  const layers = useMemo(() => {
    if (!bundle || !staticBinary) return [];
    const { n_polys, startIndices, elevation, fillColor, driftCoords } = staticBinary;

    const ref = bundle.header.aoi.reference;
    const refPt = ref && Number.isFinite(ref.lon) && Number.isFinite(ref.lat)
      ? [{ position: [ref.lon, ref.lat] as [number, number], note: ref.note }]
      : [];

    return [
      // C3 — block aggregation overlay (toggleable). Drawn under the 3D
      // buildings as a flat choropleth colored by the block's worst velocity.
      ...(showBlocks ? [
        new PolygonLayer({
          id: "block-overlay",
          data: blockData,
          getPolygon: (d: { polygon: [number, number][] }) => d.polygon,
          getFillColor: (d: { worstVelocity: number }) => {
            const [r, g, b] = velocityToRGBA(d.worstVelocity, 1, false);
            return [r, g, b, 70] as [number, number, number, number];
          },
          getLineColor: [148, 163, 184, 140],
          getLineWidth: 1,
          lineWidthUnits: "pixels",
          stroked: true,
          filled: true,
          extruded: false,
          pickable: true,
          parameters: { depthTest: false },
          updateTriggers: { getFillColor: [blockData] },
        } as never),
      ] : []),

      new SolidPolygonLayer({
        id: "buildings-3d",
        data: {
          length: n_polys,
          startIndices,
          attributes: {
            // driftCoords mirrors bundle.ringCoords in subsidence mode and
            // carries per-building east-west offsets in drift mode. Pointer
            // is stable across mode toggles; the version trigger forces a
            // GPU re-upload when contents change.
            getPolygon:   { value: driftCoords, size: 2 },
            getElevation: { value: elevation,   size: 1 },
            // normalized: false tells deck.gl/luma to read the uint8 0..255
            // values as-is in the shader (i.e. NOT divide by 255).
            getFillColor: { value: fillColor, size: 4, normalized: false },
          },
        },
        _normalize: false,           // we hand it well-formed rings; skip O(n) check
        extruded: true,
        wireframe: false,
        pickable: true,
        material: { ambient: 0.5, diffuse: 0.6, shininess: 12 },

        // updateTriggers refreshes the GPU attribute when the buffer content
        // changes (even though the array reference is stable).
        updateTriggers: {
          getFillColor: colorVersion,
          getPolygon:   driftVersion,
        },

        onClick: (info: { index?: number }) => {
          if (typeof info.index === "number" && info.index >= 0) {
            setSelectedRow(info.index);
          }
        },
        autoHighlight: false,
        parameters: { depthTest: true, depthMask: true },
        getLineColor: [0, 0, 0, 0],
      } as never),

      // Selected-building highlight: a white outline ring drawn on the selected
      // building (its fill is already solid red via SELECTED_RGBA). Single
      // feature, never pickable (so it can't steal clicks from buildings-3d),
      // depthTest off so the halo stays visible behind taller neighbours.
      new PolygonLayer({
        id: "selection-outline",
        data: selectionOutline,
        getPolygon: (d: { polygon: [number, number][] }) => d.polygon,
        extruded: true,
        getElevation: (d: { height: number }) => d.height,
        stroked: true,
        filled: false,
        getLineColor: [255, 255, 255, 255],   // white halo
        getLineWidth: 2,
        lineWidthUnits: "pixels",
        pickable: false,
        parameters: { depthTest: false },
        updateTriggers: { getPolygon: [selectionOutline] },
      } as never),

      // B4 — InSAR reference point (⚓). Stable anchor every velocity is
      // measured against. A ring + a glyph; tooltip carries the note.
      new ScatterplotLayer({
        id: "reference-ring",
        data: refPt,
        getPosition: (d: { position: [number, number] }) => d.position,
        getRadius: 9,
        radiusUnits: "pixels",
        stroked: true,
        filled: false,
        getLineColor: [56, 189, 248, 255],   // sky-400
        getLineWidth: 2,
        lineWidthUnits: "pixels",
        pickable: true,
        parameters: { depthTest: false },
      } as never),
      new TextLayer({
        id: "reference-label",
        data: refPt,
        getPosition: (d: { position: [number, number] }) => d.position,
        getText: () => "⚓",
        getSize: 16,
        sizeUnits: "pixels",
        getColor: [56, 189, 248, 255],
        getTextAnchor: "middle",
        getAlignmentBaseline: "center",
        parameters: { depthTest: false },
        pickable: true,
      } as never),
    ];
  }, [bundle, staticBinary, colorVersion, driftVersion, showBlocks, blockData, selectionOutline]);

  useEffect(() => {
    overlayRef.current?.setProps({
      layers,
      // Tooltip for the block overlay + reference pin. Building picks are
      // handled by onClick → sidebar, so we only surface the non-building
      // layers here.
      getTooltip: (info: { layer?: { id?: string } | null; object?: unknown }) => {
        const id = info.layer?.id;
        const o = info.object as Record<string, number> | { note?: string } | undefined;
        if (!o) return null;
        if (id === "block-overlay") {
          const d = o as Record<string, number>;
          return {
            text:
              `block · ${d.count} buildings\n` +
              `worst velocity: ${d.worstVelocity.toFixed(1)} mm/yr\n` +
              `mean risk: ${(d.meanRisk * 100).toFixed(0)} · max ${(d.maxRisk * 100).toFixed(0)}\n` +
              `confirmed threats: ${d.confirmed}`,
          };
        }
        if (id === "reference-ring" || id === "reference-label") {
          const note = (o as { note?: string }).note;
          return note ? { text: `⚓ reference\n${note}` } : null;
        }
        return null;
      },
    });
  }, [layers]);

  // Force a resize whenever the window dimensions change OR after layers first
  // appear. MapLibre measures its container once at mount; if the container
  // grew (e.g. flex children settled, devtools opened/closed), the GL viewport
  // stays at the old size and 3D buildings render outside the frustum.
  useEffect(() => {
    if (!mapRef.current) return;
    const m = mapRef.current;
    const r = () => m.resize();
    window.addEventListener("resize", r);
    // Kick a resize a beat after layers exist so any late layout settles.
    const t = window.setTimeout(r, 100);
    return () => {
      window.removeEventListener("resize", r);
      window.clearTimeout(t);
    };
  }, [layers.length > 0]);

  /* ---- Playback -------------------------------------------------------- */
  useEffect(() => {
    if (!playing || !bundle) return;
    const id = window.setInterval(() => {
      const next = (monthIdxRef.current + 1) % bundle.header.n_months;
      monthIdxRef.current = next;
      setMonthIdxState(next);
    }, 320);
    return () => window.clearInterval(id);
  }, [playing, bundle]);

  const handleScrub = (m: number) => {
    monthIdxRef.current = m;
    setMonthIdxState(m);
  };

  /* ---- Zoom controls ---------------------------------------------------
   * Stable identities (mapRef is a ref → empty deps) so the memoized
   * ZoomControl never reconciles on the parent's per-frame playback renders.
   * MapLibre's zoomIn/zoomOut animate ±1 and clamp to the style's min/max. */
  const handleZoomIn  = useCallback(() => mapRef.current?.zoomIn(),  []);
  const handleZoomOut = useCallback(() => mapRef.current?.zoomOut(), []);

  return (
    <>
      {/* h-full is essential: the parent is a horizontal flex container, so
          flex-1 makes us fill the WIDTH. But there is no child in flow to
          give us height (TopBar/TimeSlider/mapContainer are all position
          absolute), so without h-full the wrapper collapses to height 0 and
          the absolute-positioned mapContainer renders at 0×0 — the screen
          stays dark. Diagnostic confirmed: mapContainer {h: 0}, parent
          {h: 1293}. h-full pins us to the parent's height (1293). */}
      <div className="flex-1 h-full relative">
        {/* Inline style: maplibre-gl.css ships `.maplibregl-map { position: relative }`
            which loads AFTER Tailwind and beats `absolute` on specificity tie. The
            container then collapses to height 0 because nothing in flow gives it
            height. Inline style wins over any stylesheet rule, full stop. */}
        <div
          ref={mapContainer}
          style={{ position: "absolute", inset: 0 }}
        />

        <TopBar
          aois={aois}
          activeCode={activeCode}
          onSelect={setActiveCode}
          observationDate={bundle?.header.dates[monthIdx]}
          mode={mode}
          onModeChange={onModeChange}
          showBlocks={showBlocks}
          onToggleBlocks={setShowBlocks}
        />

        <TimeSlider
          n={bundle?.header.n_months ?? 0}
          idx={monthIdx}
          onScrub={handleScrub}
          playing={playing}
          onTogglePlay={() => setPlaying(p => !p)}
          dates={bundle?.header.dates}
        />

        <ZoomControl onZoomIn={handleZoomIn} onZoomOut={handleZoomOut} />

        {!bundle && (
          <div className="absolute inset-0 grid place-items-center pointer-events-none">
            <div className="text-xs uppercase tracking-[0.3em] text-slate-500 animate-pulse">
              Loading bundle…
            </div>
          </div>
        )}
      </div>

      <ThreatSidebar
        bundle={bundle}
        selectedRow={selectedRow}
        monthIdx={monthIdx}
        activeAoi={activeAoi}
        mode={mode}
      />
    </>
  );
}


/* ===========================================================================
 *  Color ramp
 *  velocity_mm_yr in [-25, +5] → red → amber → green → cyan
 *  Low coherence: desaturated.
 *  Selected: bright cyan outline (alpha boost).
 * =========================================================================== */
function velocityToRGBA(v: number, coh: number, selected: boolean): [number, number, number, number] {
  // Selected building: solid red highlight, short-circuiting the ramp so it
  // reads as "this is the one you clicked", not as a velocity value.
  if (selected) return SELECTED_RGBA;

  // Clamp + normalize
  let t = (v - VELOCITY_FLOOR_MM_YR) / (VELOCITY_CEIL_MM_YR - VELOCITY_FLOOR_MM_YR);
  if (t < 0) t = 0; else if (t > 1) t = 1;

  // 3-stop ramp:  red (t=0) → amber (t≈0.55) → green (t=1)
  let r: number, g: number, b: number;
  if (t < 0.55) {
    const u = t / 0.55;
    r = 239 + (245 - 239) * u;
    g = 68  + (158 - 68)  * u;
    b = 68  + (11  - 68)  * u;
  } else {
    const u = (t - 0.55) / 0.45;
    r = 245 + (34  - 245) * u;
    g = 158 + (197 - 158) * u;
    b = 11  + (94  - 11)  * u;
  }

  // Desaturate when InSAR coherence is low — "we don't trust this number".
  if (coh < 0.30) {
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    const mix = (coh / 0.30); // 0 at low end, 1 at threshold
    r = gray + (r - gray) * mix;
    g = gray + (g - gray) * mix;
    b = gray + (b - gray) * mix;
  }

  let a = 220;
  if (selected) a = 255;

  return [Math.round(r), Math.round(g), Math.round(b), a];
}


/* ===========================================================================
 *  Drift color ramp
 *  ew_velocity_mm_yr in [-EW_VELOCITY_BOUND, +EW_VELOCITY_BOUND]:
 *      strong west = blue, zero = neutral grey, strong east = orange.
 *  Low coherence: desaturated (same as subsidence ramp).
 *  Selected: alpha boost.
 * =========================================================================== */
function driftToRGBA(ew: number, coh: number, selected: boolean): [number, number, number, number] {
  // Selected building: solid red highlight, short-circuiting the drift ramp
  // (same rationale as velocityToRGBA).
  if (selected) return SELECTED_RGBA;

  // Normalize ew to t in [0, 1] where 0 = -bound (west) and 1 = +bound (east).
  let t = (ew + EW_VELOCITY_BOUND) / (2 * EW_VELOCITY_BOUND);
  if (t < 0) t = 0; else if (t > 1) t = 1;

  // 3-stop diverging ramp: blue (west) → grey (zero) → orange (east).
  // Anchor colors:  #38bdf8 (sky-400) → #64748b (slate-500) → #f97316 (orange-500)
  let r: number, g: number, b: number;
  if (t < 0.5) {
    const u = t / 0.5;
    r =  56 + (100 -  56) * u;
    g = 189 + (116 - 189) * u;
    b = 248 + (139 - 248) * u;
  } else {
    const u = (t - 0.5) / 0.5;
    r = 100 + (249 - 100) * u;
    g = 116 + (115 - 116) * u;
    b = 139 + ( 22 - 139) * u;
  }

  if (coh < 0.30) {
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    const mix = coh / 0.30;
    r = gray + (r - gray) * mix;
    g = gray + (g - gray) * mix;
    b = gray + (b - gray) * mix;
  }

  const a = selected ? 255 : 220;
  return [Math.round(r), Math.round(g), Math.round(b), a];
}


/* ===========================================================================
 *  MapLibre style — falls back to a flat dark canvas if no PMTiles file is
 *  available. As soon as `frontend/public/tiles/nairobi.pmtiles` exists, the
 *  pmtiles:// source will resolve and overlay road/building base layers.
 * =========================================================================== */
function emptyDarkStyle(): maplibregl.StyleSpecification {
  return {
    version: 8,
    sources: {},
    layers: [{ id: "bg", type: "background", paint: { "background-color": "#070a0f" } }],
  } as unknown as maplibregl.StyleSpecification;
}

/**
 * Dark basemap built against the Protomaps schema. Source layers used:
 *   water, landuse, roads, buildings, boundaries, places
 * Activated only after the probe in the init effect sees the .pmtiles file.
 */
function protomapsDarkStyle(): maplibregl.StyleSpecification {
  return {
    version: 8,
    sources: {
      basemap: {
        type: "vector",
        url: "pmtiles:///tiles/nairobi.pmtiles",
        attribution: "© OpenStreetMap · Protomaps",
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#070a0f" } },
      {
        id: "water",
        type: "fill",
        source: "basemap",
        "source-layer": "water",
        paint: { "fill-color": "#0a1726" },
      },
      {
        id: "landuse",
        type: "fill",
        source: "basemap",
        "source-layer": "landuse",
        paint: { "fill-color": "#0e141c", "fill-opacity": 0.6 },
      },
      {
        id: "roads-casing",
        type: "line",
        source: "basemap",
        "source-layer": "roads",
        minzoom: 12,
        paint: {
          "line-color": "#11161e",
          "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.5, 16, 4],
        },
      },
      {
        id: "roads",
        type: "line",
        source: "basemap",
        "source-layer": "roads",
        paint: {
          "line-color": "#3a4756",
          "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.3, 16, 1.6],
        },
      },
      {
        id: "buildings-base",
        type: "fill",
        source: "basemap",
        "source-layer": "buildings",
        minzoom: 13,
        paint: { "fill-color": "#1a2230", "fill-opacity": 0.25 },
      },
      {
        id: "boundaries",
        type: "line",
        source: "basemap",
        "source-layer": "boundaries",
        paint: { "line-color": "#2a3445", "line-dasharray": [2, 2], "line-width": 0.6 },
      },
    ],
  } as unknown as maplibregl.StyleSpecification;
}


/**
 * Per-error-kind copy. Each branch tells the user exactly what to check next:
 *   - network: API process is the suspect (uvicorn down, wrong port).
 *   - http:    API process answered but rejected the call (4xx/5xx body).
 *   - parse:   wire format mismatch; almost always a stale browser cache
 *              holding a pre-fix bundle. Hard-refresh (Ctrl-Shift-R) clears it.
 */
function ErrorPanel({ err, where }: { err: ApiError; where: string }) {
  let title: string;
  let hint: string;
  switch (err.kind) {
    case "network":
      title = `API unreachable — could not connect to ${where}`;
      hint = "Is uvicorn running on :8000?  Check with `curl -s localhost:8000/health` or `pgrep -af uvicorn`.";
      break;
    case "http":
      title = `API error ${err.status} from ${where}`;
      hint = err.status >= 500
        ? "The backend is up but threw an error. Check the uvicorn terminal for the traceback."
        : "The request was rejected. Verify the AOI code or query parameters.";
      break;
    case "parse":
      title = `Failed to decode ${where}`;
      hint = "Stale browser cache is the usual cause for the Int32Array alignment error. Hard-refresh (Ctrl-Shift-R) to drop the cached bundle, or clear site data.";
      break;
  }
  return (
    <div className="h-screen w-screen grid place-items-center bg-ink-950 text-slate-200 font-mono">
      <div className="max-w-md border border-red-500/40 bg-red-950/30 p-6">
        <div className="text-xs uppercase tracking-widest text-red-400">Error</div>
        <div className="mt-2 text-sm">{title}</div>
        <div className="mt-1 text-xs text-slate-500 break-words">{err.message}</div>
        <div className="mt-3 text-xs text-slate-400">{hint}</div>
      </div>
    </div>
  );
}

/* re-export so consumers can read inline (e.g. risk panel) */
export { velocityAt, displacementAt, coherenceAt, horizontalVelocityAt, buildingSeries };
