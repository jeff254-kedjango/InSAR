"""
Stage 4: join MintPy raster outputs to building footprints, emit GeoParquet.

The schema is fixed by `scripts/init_db.sql` — we have to produce exactly the
columns the DuckDB views expect, in Hive-partitioned layout:

    data/parquet/buildings/aoi=<code>/part-0.parquet
    data/parquet/subsidence/aoi=<code>/part-0.parquet
    data/parquet/env_index/aoi=<code>/part-0.parquet     (synthesized; no real env data yet)
    data/parquet/aoi_registry.parquet

Per footprint, we aggregate raster pixels into one (velocity, σ, displacement-
time-series) row. The aggregation is coherence-weighted least-squares: pixels
with γ < 0.3 are dropped (incoherent), the rest are blended with weight γ². This
is the same model the risk engine assumes downstream — σ ∝ (1 - γ).

Performance: footprints are O(10³) per AOI, pixels are O(10⁶). The inner loop is
**numpy-vectorized** rather than per-pixel Python — we rasterize every footprint
once into a uint32 label grid, then groupby-aggregate against the displacement /
coherence stacks. That's the only way to do this in seconds instead of minutes.

Run from backend/:
    python -m scripts.join_insar --aoi huruma --track ASCENDING/57
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from scripts.aois import AOI, REGISTRY, by_code
from scripts.provenance import set_provenance
from scripts.postprocess import (
    BUILDINGS_SCHEMA,
    SUBSIDENCE_SCHEMA,
    ENV_SCHEMA,
    COH_SERIES_SCHEMA,
    CLASS_INDETERMINATE,
    DEM_ERR_FLAG_M,
    FAILURE_ELASTIC,
    _trailing_velocity,
    _stl_decompose,
    _acceleration_mm_yr2,
    _velocity_sigma_from_coherence,
    _classify,
    defensibility_thresholds,
    _cohort_percentiles,
    _rank_within_groups,
    assign_blocks,
    extract_closure_rms,
    extract_dem_err,
    extract_coh_per_epoch,
    pack_coh_series_per_building,
    synthesize_env_context,
    synthesize_env_index_rows,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
PARQUET_ROOT = BACKEND_DIR / "data" / "parquet"
MINTPY_DIR = BACKEND_DIR / "data" / "mintpy"
FOOTPRINT_DIR = BACKEND_DIR / "data" / "footprints"
DB_PATH = BACKEND_DIR / "data" / "demo.duckdb"

# Coherence floor. Pixels below this are dropped before aggregation — they
# carry effectively zero information and bias the LS fit if included.
COH_MIN = 0.30

# σ_v = K_SIGMA × (1 - γ_mean). K_SIGMA calibrated from MintPy validation papers
# (Lazecký et al. 2020) for Sentinel-1 IW SBAS at 80 m looks: σ saturates at
# ~5 mm/yr for γ → 0, drops to ~0.5 mm/yr at γ = 0.9.
K_SIGMA = 5.0


@dataclass(frozen=True)
class RasterStack:
    """A georeferenced (time, y, x) stack: displacement series + coherence."""
    # mm, shape (T, H, W). Sign convention: + = subsidence (negative LOS away
    # from satellite). MintPy emits metres along LOS; we convert.
    displacement_mm: np.ndarray
    # 0-1, shape (H, W). Temporal coherence — fit quality of the SBAS inversion.
    coherence: np.ndarray
    # Annualized LS slope in mm/yr; shape (H, W). MintPy's velocity.h5.
    velocity_mm_yr: np.ndarray
    # ISO date strings, length T.
    dates: list[str]
    # Spatial axes; both monotonically increasing. Units may be metres (UTM,
    # when MintPy reads HyP3 GAMMA which is already geocoded into UTM) or
    # degrees (WGS84, when MintPy geocodes from radar). The CRS lives in `epsg`.
    xs: np.ndarray  # shape (W,)
    ys: np.ndarray  # shape (H,)
    epsg: int       # raster CRS — e.g. 32737 for our HyP3 outputs, 4326 if degrees


def _resolve_mintpy_paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Locate the four MintPy outputs we need.

    MintPy writes geocoded outputs in two places depending on whether the input
    was already in geographic/projected coords:
      - radar-coords input → run_dir/geo/geo_velocity.h5, etc.
      - already-geocoded input (HyP3 GAMMA) → run_dir/velocity.h5 (flat)
    Probe both, raise with a useful error if neither shape is present.

    Coherence note: we deliberately use `avgSpatialCoh.h5` (mean interferometric
    coherence γ over the *pre-inversion* network), NOT `temporalCoherence.h5`
    (post-inversion fit quality). With coherence-based pair filtering enabled,
    temporalCoherence saturates at 1.0 by construction — it would make σ_v = 0
    everywhere, which is dishonest. avgSpatialCoh preserves the physical noise
    signal we need for the σ_v ∝ (1 - γ) model downstream.
    """
    geo = run_dir / "geo"
    if (geo / "geo_velocity.h5").exists():
        return (
            geo / "geo_timeseries.h5",
            geo / "geo_velocity.h5",
            geo / "geo_avgSpatialCoh.h5" if (geo / "geo_avgSpatialCoh.h5").exists() else geo / "geo_temporalCoherence.h5",
            geo / "geo_geometryRadar.h5",
        )
    return (
        run_dir / "timeseries.h5",
        run_dir / "velocity.h5",
        run_dir / "avgSpatialCoh.h5",
        run_dir / "inputs" / "geometryGeo.h5",
    )


def load_mintpy_stack(run_dir: Path) -> RasterStack:
    """Read the four HDF5 outputs MintPy emits and stitch into a RasterStack."""
    import h5py

    ts_path, vel_path, coh_path, geom_path = _resolve_mintpy_paths(run_dir)
    for p in (ts_path, vel_path, coh_path, geom_path):
        if not p.exists():
            raise FileNotFoundError(f"missing MintPy output: {p}")

    with h5py.File(ts_path, "r") as f:
        # timeseries: shape (n_date, H, W), metres along LOS
        disp_m = np.asarray(f["timeseries"], dtype=np.float32)
        date_bytes = np.asarray(f["date"])
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in date_bytes]
        # MintPy date strings are YYYYMMDD — convert to ISO YYYY-MM-DD.
        dates = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d for d in dates]
        attrs = dict(f.attrs)

    # Sign convention: MintPy's `timeseries`/`velocity` are LOS-along, with
    # POSITIVE = motion toward the satellite (i.e. uplift for near-vertical LOS).
    # We project LOS onto vertical via the incidence angle: vertical = LOS / cos(inc).
    # The schema convention shared with phenomena.py is **negative = subsidence**
    # (matching `composite_risk:subs_score = -vel/25`), so we DO NOT flip sign
    # at this step — LOS-positive-uplift carries straight through to
    # mm/yr-positive-uplift, leaving subsidence negative as expected.
    # (Tier 3 ASC/DESC decomposition would split LOS into v_up + v_ew; for now
    # we report vertical-equivalent.)
    with h5py.File(geom_path, "r") as f:
        inc_deg = np.asarray(f["incidenceAngle"], dtype=np.float32)
    cos_inc = np.cos(np.radians(inc_deg))
    cos_inc = np.where(cos_inc < 0.1, 0.1, cos_inc)  # clamp to keep numerics sane
    # broadcast (H, W) over (T, H, W). disp_m is metres → mm.
    disp_mm = (disp_m / cos_inc) * 1000.0

    with h5py.File(vel_path, "r") as f:
        vel = np.asarray(f["velocity"], dtype=np.float32)  # m/yr LOS
        vel_attrs = dict(f.attrs)
    vel_mm_yr = (vel / cos_inc) * 1000.0

    # ---- Dry-run guard (ARCHITECTURE_THREE A1) ----------------------------
    # scripts/_dryrun_stage4.py fabricates a tiny MintPy-shaped dir to exercise
    # this join while the real HyP3/MintPy run is pending. Those products are
    # ~25×25 px and carry troposphericDelay.method=height_correlation (never
    # gacos). Shipping them to a public broadcast would be a fabricated-data
    # incident, so we refuse to do it silently. A real geocoded SBAS stack over
    # a 2 km AOI at ~30 m is hundreds of px per side and was run with GACOS.
    H_v, W_v = vel.shape
    tropo = str(vel_attrs.get("mintpy.troposphericDelay.method", "")).lower()
    if H_v <= 30 and W_v <= 30 and tropo != "gacos":
        msg = (
            f"\n  ‼ DRY-RUN / PLACEHOLDER VELOCITY DETECTED in {vel_path}\n"
            f"    grid is {H_v}×{W_v} px (real SBAS over this AOI is hundreds of px)\n"
            f"    troposphericDelay.method = '{tropo or 'unset'}' (expected 'gacos')\n"
            f"    These look like _dryrun_stage4.py outputs, NOT a real MintPy run.\n"
            f"    Run smallbaselineApp.py on OpenSARLab (see docs/opensarlab_runbook.md)\n"
            f"    and download the real geocoded products before joining for the demo.\n"
            f"    Set GACOS_JOIN_ALLOW_PLACEHOLDER=1 to override (dev/testing only).\n"
        )
        if os.environ.get("GACOS_JOIN_ALLOW_PLACEHOLDER") == "1":
            print(msg, file=sys.stderr)
            print("  ↳ override set — proceeding with placeholder data.", file=sys.stderr)
        else:
            raise SystemExit(msg)

    with h5py.File(coh_path, "r") as f:
        # avgSpatialCoh.h5 stores the layer as "coherence"; temporalCoherence.h5
        # stores it as "temporalCoherence". Pick whichever dataset is in the file.
        ds = "coherence" if "coherence" in f else "temporalCoherence"
        coh = np.asarray(f[ds], dtype=np.float32)

    # Reconstruct x/y axes from geocoded attrs. MintPy stores X_FIRST, Y_FIRST,
    # X_STEP, Y_STEP, WIDTH, LENGTH in the file attributes. Units depend on the
    # CRS — for our HyP3 GAMMA stacks they're metres in UTM 37S (EPSG 32737);
    # if MintPy ever geocodes from radar into 4326 they'd be degrees. The
    # axis-construction math is identical either way; only the building-centroid
    # transform downstream needs to match.
    H, W = coh.shape
    x_first = float(attrs["X_FIRST"])
    y_first = float(attrs["Y_FIRST"])
    x_step  = float(attrs["X_STEP"])    # positive (east / increasing x)
    y_step  = float(attrs["Y_STEP"])    # negative (south / decreasing y)
    xs = x_first + np.arange(W, dtype=np.float64) * x_step
    ys = y_first + np.arange(H, dtype=np.float64) * y_step
    if y_step < 0:
        # Flip so ys[0] is the south edge; mirror disp/coh/vel rows to match.
        ys = ys[::-1]
        disp_mm = disp_mm[:, ::-1, :]
        coh = coh[::-1, :]
        vel_mm_yr = vel_mm_yr[::-1, :]

    epsg = int(attrs.get("EPSG", 4326))

    return RasterStack(
        displacement_mm=disp_mm,
        coherence=coh,
        velocity_mm_yr=vel_mm_yr,
        dates=dates,
        xs=xs.astype(np.float64),
        ys=ys.astype(np.float64),
        epsg=epsg,
    )


def rasterize_footprints(
    footprints_pq: Path, stack: RasterStack,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Map every in-AOI footprint to one InSAR pixel (its centroid's pixel).

    Returns (pixel_index, building_ids, geom_wkb_array):
        pixel_index: list of (row, col) into the (H, W) raster; one entry per
                     building, aligned with building_ids.
        building_ids: (N,) int64 — the footprint's ROW INDEX in footprints_pq.
                     This is the canonical internal key: it is always present
                     and source-agnostic. The native source id (`osm_id` for
                     OSM, `open_buildings_id` for Open Buildings) is sparse —
                     `osm_id` is 100% null on Open-Buildings AOIs (Huruma) — so
                     it cannot serve as the join key. Carrying the row index
                     also lets emit_parquet gather static columns by direct
                     O(1) indexing instead of a dict lookup.
        keep_geoms:   (N,) object array of WKB bytes

    Why centroid lookup and not full rasterization
    ----------------------------------------------
    HyP3 InSAR_GAMMA at 20x4 looks gives ~80 m pixels (≈ 6,400 m² each). The
    median Huruma footprint is 64 m² — three orders of magnitude smaller.
    Trying to rasterize a hut into the pixel grid via point-in-polygon at
    pixel centres drops virtually all of them (only those that happen to
    contain a centre survive). Mombasa is similar.

    The physical truth is that **InSAR cannot resolve a sub-pixel building**:
    we observe the deformation of a 6,400 m² ground cell, and every footprint
    inside that cell shares the same measurement. Centroid-to-pixel lookup
    expresses that truth: each building inherits its containing pixel's
    velocity, coherence, and time series. Neighbours within the same pixel
    will report identical values — that's not a bug, it's the resolution
    limit. The UI should surface this honestly.

    For large buildings (Mombasa concrete blocks >> 1 pixel), centroid lookup
    is still correct; finer aggregation buys nothing because the InSAR phase
    is itself a many-look average.
    """
    import pyarrow.parquet as pq

    tbl = pq.read_table(footprints_pq)
    geom_wkbs = tbl.column("geom_wkb").to_pylist()
    # The footprints table already carries centroid_lon/lat in WGS84 — use those
    # rather than recomputing from WKB (faster, and matches what we write
    # downstream).
    c_lons = np.asarray(tbl.column("centroid_lon").to_pylist(), dtype=np.float64)
    c_lats = np.asarray(tbl.column("centroid_lat").to_pylist(), dtype=np.float64)

    # If the raster is in a projected CRS (metres), project the centroids from
    # WGS84 into it once, vectorized. For our HyP3 stacks this is the path:
    # EPSG 32737 (UTM 37S). If it ever runs on a 4326-geocoded MintPy output,
    # the transformer is identity and centroids stay as lon/lat — code below
    # doesn't care which axis system it's in, only that footprint coords match
    # raster axes.
    if stack.epsg == 4326:
        c_xs, c_ys = c_lons, c_lats
    else:
        from pyproj import Transformer
        tx = Transformer.from_crs(4326, stack.epsg, always_xy=True)
        c_xs, c_ys = tx.transform(c_lons, c_lats)
        c_xs = np.asarray(c_xs, dtype=np.float64)
        c_ys = np.asarray(c_ys, dtype=np.float64)

    H, W = stack.coherence.shape
    xs, ys = stack.xs, stack.ys
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])  # positive — load_mintpy_stack flips so ys increase
    x0, y0 = float(xs[0]), float(ys[0])

    pixel_index: list[tuple[int, int]] = []
    keep_geoms: list[bytes] = []
    keep_ids: list[int] = []

    x_lo, x_hi = float(xs[0]), float(xs[-1])
    y_lo, y_hi = float(ys[0]), float(ys[-1])
    for row_idx, (raw_wkb, cx, cy) in enumerate(zip(geom_wkbs, c_xs, c_ys)):
        # Cheap bbox cull before pixel-index math.
        if not (x_lo <= cx <= x_hi and y_lo <= cy <= y_hi):
            continue
        j = int(round((cx - x0) / dx))
        i = int(round((cy - y0) / dy))
        # Clamp to grid (round can push past edge by one).
        if not (0 <= i < H and 0 <= j < W):
            continue
        pixel_index.append((i, j))
        keep_geoms.append(raw_wkb)
        keep_ids.append(row_idx)

    return pixel_index, np.array(keep_ids, dtype=np.int64), np.array(keep_geoms, dtype=object)


def aggregate(stack: RasterStack, pixel_index: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """Look up per-building (velocity, coherence, time-series) from its pixel.

    Because every building maps to exactly one InSAR pixel (see
    `rasterize_footprints` for why), aggregation degenerates to a fancy
    indexing operation: gather (T, H, W) along (H, W) at the building's
    (i, j) coordinates. Multiple buildings sharing a pixel get identical
    values — that's the InSAR resolution limit, not a bug.

    Buildings whose pixel has coh < COH_MIN or NaN coherence are flagged
    dead (NaN velocity); they stay in the output for row alignment so the
    risk engine can mask them downstream.
    """
    N = len(pixel_index)
    if N == 0:
        empty32 = np.zeros(0, dtype=np.float32)
        return {
            "velocity_mm_yr": empty32,
            "velocity_sigma_mm_yr": empty32,
            "coherence": empty32,
            "n_pixels": np.zeros(0, dtype=np.int32),
            "displacement_mm": np.zeros((0, stack.displacement_mm.shape[0]), dtype=np.float32),
        }
    rows = np.array([i for i, _ in pixel_index], dtype=np.int64)
    cols = np.array([j for _, j in pixel_index], dtype=np.int64)

    coh = stack.coherence[rows, cols].astype(np.float32)
    vel = stack.velocity_mm_yr[rows, cols].astype(np.float32)
    # disp shape (T, H, W) → take (T, N): for each date, gather at the same pixel.
    disp = stack.displacement_mm[:, rows, cols].astype(np.float32)  # (T, N)
    displacement_mm = disp.T.copy()  # (N, T)

    dead = ~np.isfinite(coh) | (coh < COH_MIN)
    sigma = (K_SIGMA * (1.0 - coh)).astype(np.float32)
    vel = vel.copy()
    coh = coh.copy()
    vel[dead] = np.nan
    coh[dead] = np.nan
    sigma[dead] = np.nan
    displacement_mm[dead, :] = np.nan

    # n_pixels = 1 per building when alive (it's a single-pixel lookup). 0 when
    # dead. The downstream risk engine uses this as a "has-InSAR-data" flag.
    n_px = np.where(dead, 0, 1).astype(np.int32)

    return {
        "velocity_mm_yr": vel,
        "velocity_sigma_mm_yr": sigma,
        "coherence": coh,
        "n_pixels": n_px,
        "displacement_mm": displacement_mm,   # shape (N, T)
    }


def _synthesize_aoi_periods(dates_iso: list[str]) -> list[date]:
    """Map the per-month observation dates to a quarterly cadence for env_index.

    Returns first-of-quarter dates spanning the observed range, plus the
    observed final month so the UI's "latest" slot has a point.
    Determinism: a pure function of the date list.
    """
    if not dates_iso:
        return []
    parsed = sorted({date.fromisoformat(d) for d in dates_iso})
    first, last = parsed[0], parsed[-1]
    # Pick the first day of every quarter (Jan, Apr, Jul, Oct) that lies within
    # [first, last]. We snap forward to the next quarter-start ≥ first.
    quarters: list[date] = []
    y, m = first.year, ((first.month - 1) // 3) * 3 + 1
    cur = date(y, m, 1)
    if cur < first:
        # Advance one quarter so we don't predate the data.
        m2 = m + 3
        if m2 > 12:
            cur = date(y + 1, m2 - 12, 1)
        else:
            cur = date(y, m2, 1)
    while cur <= last:
        quarters.append(cur)
        m2 = cur.month + 3
        if m2 > 12:
            cur = date(cur.year + 1, m2 - 12, 1)
        else:
            cur = date(cur.year, m2, 1)
    # Pin the final observed date so the UI has a fresh "now" anchor.
    last_anchor = date(last.year, last.month, 1)
    if last_anchor not in quarters:
        quarters.append(last_anchor)
    return quarters


def emit_parquet(
    aoi: AOI,
    footprints_pq: Path,
    keep_ids: np.ndarray,
    keep_geoms: np.ndarray,
    agg: dict[str, np.ndarray],
    dates: list[str],
    *,
    run_dir: Path,
    stack_shape: tuple[int, int],
    pixel_index: list[tuple[int, int]],
) -> None:
    """Write Hive-partitioned parquet matching scripts/init_db.sql expectations.

    Buildings get the full BUILDINGS_SCHEMA: InSAR-derived columns are real
    measurements; environmental-context columns (soil_class, riparian_dist_m,
    shoreline_dist_m, reclaimed_land, built_year overrides) are synthesized
    via `postprocess.synthesize_env_context` and tagged as plausible-fake in
    the module docstring.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    N = len(keep_ids)
    if N == 0:
        print("  no buildings retained after coherence filter", file=sys.stderr)
        return
    T = len(dates)
    if T < 24:
        raise RuntimeError(
            f"too few observation dates ({T}) for STL decomposition; need T >= 24. "
            f"Re-run with a longer date range or relax mintpy.network.minCoherence."
        )

    # ---- Pull static footprint attributes ---------------------------------
    # OSM tags are sparse: any of `n_floors`, `height_m`, `built_year` can be
    # NULL on a footprint without tagging. Replace with sensible defaults
    # (1 floor → ~3 m, 0 built_year → "unknown"); downstream synth fills the
    # gaps where appropriate.
    # keep_ids ARE footprint row indices (see rasterize_footprints), so we
    # index fp_tbl columns directly — no id→row dict. The old code keyed a dict
    # on `osm_id`, which is null on Open-Buildings AOIs and collapsed every row
    # onto a single None key.
    fp_tbl = pq.read_table(footprints_pq).to_pydict()
    ids_list = keep_ids.tolist()

    def _gather_float(col: str, default: float) -> np.ndarray:
        vals = [fp_tbl[col][i] for i in ids_list]
        return np.array([default if v is None else float(v) for v in vals], dtype=np.float64)

    def _gather_int(col: str, default: int) -> np.ndarray:
        vals = [fp_tbl[col][i] for i in ids_list]
        return np.array([default if v is None else int(v) for v in vals], dtype=np.int32)

    c_lon       = _gather_float("centroid_lon", 0.0)
    c_lat       = _gather_float("centroid_lat", 0.0)
    n_floors_fp = _gather_int("n_floors", 1)         # 1 floor when missing
    height_fp   = _gather_float("height_m", 3.0)     # ~3 m single-storey when missing
    built_yr_fp = _gather_int("built_year", 0)       # 0 → "use synthesized year"

    # Native source ids, gathered by row index, kept sparse (None where the
    # source doesn't carry one). `footprint_source` drives which column the UI
    # treats as authoritative; both are written verbatim so provenance survives.
    osm_id_fp = [fp_tbl["osm_id"][i] for i in ids_list]
    ob_id_fp  = [fp_tbl["open_buildings_id"][i] for i in ids_list]

    # ---- Derive Tier-1/2/3 fields from the InSAR stack --------------------
    disp = agg["displacement_mm"]               # (N, T), mm, + = uplift
    coh  = agg["coherence"]                     # (N,)
    vel  = agg["velocity_mm_yr"]                # (N,), + = uplift / - = subsidence
    dead = ~np.isfinite(coh) | ~np.isfinite(vel)

    # Per-month trailing-12mo velocity. NaN disp rows would poison STL; substitute
    # zeros for them and mask the row to NaN post-hoc.
    disp_safe = np.where(np.isnan(disp), 0.0, disp).astype(np.float64)
    v_per_month = _trailing_velocity(disp_safe, window=12).astype(np.float32)
    v_per_month[dead, :] = np.nan

    period = min(12, T // 2)
    trend_disp, trend_slope, seasonal_amp, trend_r2, failure_mode = _stl_decompose(
        disp_safe, period=period
    )
    # Mask STL outputs for dead buildings — STL on a zero series returns
    # zero-trend ELASTIC, which is misleading. Keep arrays aligned (don't drop rows).
    trend_disp_masked = trend_disp.copy()
    trend_disp_masked[dead, :] = np.nan
    trend_slope_m = trend_slope.copy(); trend_slope_m[dead] = np.nan
    seasonal_amp_m = seasonal_amp.copy(); seasonal_amp_m[dead] = np.nan
    trend_r2_m = trend_r2.copy(); trend_r2_m[dead] = np.nan
    failure_mode_m = failure_mode.copy(); failure_mode_m[dead] = FAILURE_ELASTIC

    accel = _acceleration_mm_yr2(v_per_month, lookback=6)
    accel[dead] = np.nan
    # velocity_sigma: same model as phenomena.py — σ ≈ k(1-γ). Use this (real
    # γ-derived) σ rather than the K_SIGMA constant from aggregate().
    v_sigma = _velocity_sigma_from_coherence(np.nan_to_num(coh, nan=0.0))
    v_sigma[dead] = np.nan
    # v_ew_sigma: same model until ASC+DESC decomposition. Identical to v_sigma
    # for now — flag in the schema comment.
    v_ew_sigma = v_sigma.copy()

    # End-of-series state for classification + composite_risk.
    v_end = np.where(dead, 0.0, vel.astype(np.float64))
    v_ew_end = np.zeros(N, dtype=np.float64)        # no horizontal yet
    coh_end = np.where(dead, 0.0, coh.astype(np.float64))
    # Court-defensibility gate thresholds: σ_max from THIS AOI's own σ p75, r2_min
    # an absolute floor (see postprocess.defensibility_thresholds). trend_r2_m and
    # v_sigma are NaN for dead rows → they fail the gate per-building; the line
    # below then re-labels dead rows INDETERMINATE.
    r2_min, sigma_max = defensibility_thresholds(v_sigma)
    classification = np.array(
        [_classify(float(v_end[i]), float(v_ew_end[i]), float(coh_end[i]),
                   float(trend_r2_m[i]), float(v_sigma[i]), r2_min, sigma_max)
         for i in range(N)],
        dtype=np.uint8,
    )
    # Dead rows: their _classify result is meaningless because we fed it zeros;
    # downgrade explicitly so the UI badges them as "no data" not "stable".
    classification[dead] = CLASS_INDETERMINATE

    # ---- Synthesize env context + env_index ------------------------------
    seed = hash(aoi.code) & 0xFFFFFFFF
    env = synthesize_env_context(aoi, c_lon, c_lat, seed=seed)
    # `_insar_height` not used — we don't (yet) compute InSAR heights from
    # phase fringes. fused_height falls back to floor-count for now.
    insar_h = np.full(N, np.nan, dtype=np.float64)
    insar_h_sigma = np.full(N, np.nan, dtype=np.float64)
    fused_h = height_fp.copy()                  # floor-count fallback

    periods = _synthesize_aoi_periods(dates)
    env_tbl, composite_latest = synthesize_env_index_rows(
        aoi_code=aoi.code,
        building_ids=keep_ids,
        soil_class=env["soil_class"],
        riparian_dist_m=env["riparian_dist_m"],
        shoreline_dist_m=env["shoreline_dist_m"],
        vel=v_end,
        v_ew=v_ew_end,
        classification=classification,
        fused_h_m=fused_h,
        periods=periods,
        seed=seed ^ 0xC0FFEE,
    )

    composite_pct, shear_pct, cohort_size = _cohort_percentiles(
        composite_latest.astype(np.float32),
        np.abs(v_ew_end).astype(np.float32),
        fused_h.astype(np.float32),
        env["soil_class"].tolist(),
    )

    # ---- ARCHITECTURE_THREE C1/C4 — block membership + block-relative cohort -
    from scripts.aois import bbox as _aoi_bbox
    block_id, _block_meta = assign_blocks(c_lon, c_lat, _aoi_bbox(aoi))
    n_blocks = _block_meta["nx"] * _block_meta["ny"]
    cohort_block_pct = _rank_within_groups(
        composite_latest.astype(np.float32), block_id.astype(np.int64), n_blocks
    )

    # ---- ARCHITECTURE_THREE B1/B3 — diagnostic per-pixel rasters ----------
    # All extractors return NaN-filled shape-matched arrays when the source
    # HDF5 is unavailable, so the join still ships on old MintPy outputs.
    H, W = stack_shape
    rows_np = np.fromiter((i for i, _ in pixel_index), dtype=np.int64, count=N)
    cols_np = np.fromiter((j for _, j in pixel_index), dtype=np.int64, count=N)

    closure_grid = extract_closure_rms(run_dir, (H, W))      # (H, W) rad
    closure_per_b = closure_grid[rows_np, cols_np].astype(np.float32, copy=False)
    closure_per_b[dead] = np.nan

    dem_err_grid = extract_dem_err(run_dir, (H, W))           # (H, W) m
    dem_err_per_b = dem_err_grid[rows_np, cols_np].astype(np.float32, copy=False)
    dem_err_per_b[dead] = np.nan
    # |residual| > threshold → flag. NaN never trips the comparison, which is
    # what we want — buildings without DEM data are not flagged.
    dem_err_flag_per_b = np.abs(dem_err_per_b) > DEM_ERR_FLAG_M

    # ---- ARCHITECTURE_THREE B2 — per-epoch coherence stack → packed blob --
    coh_stack = extract_coh_per_epoch(run_dir, (T, H, W))     # (T, H, W) float32
    if np.isnan(coh_stack).all():
        # No source for per-epoch coherence; emit one blob of T zeros per
        # building so the parquet column is still present and the bundle
        # builder can read uniformly. UI hides the sparkline when the entire
        # series is zero.
        coh_blobs = np.array(
            [np.zeros(T, dtype=np.float32).tobytes()] * N,
            dtype=object,
        )
        n_epochs_used = T
    else:
        coh_blobs, n_epochs_used = pack_coh_series_per_building(coh_stack, rows_np, cols_np)

    # ---- Build buildings table (schema must match BUILDINGS_SCHEMA exactly) -
    # built_year override: footprints sometimes carry a value, otherwise the
    # synthetic env block provides a plausible one. Use OSM if present and >0.
    built_year = np.where(built_yr_fp > 0, built_yr_fp, env["built_year"]).astype(np.int32)
    reclaimed_set = env["reclaimed_land_set"]
    reclaimed_arr = [bool(env["reclaimed_land"][i]) if reclaimed_set[i] else None for i in range(N)]

    rows_b = [{
        "building_id":             int(keep_ids[i]),
        "aoi_code":                aoi.code,
        "footprint_source":        aoi.footprint_source,
        "osm_id":                  None if osm_id_fp[i] is None else int(osm_id_fp[i]),
        "open_buildings_id":       None if ob_id_fp[i] is None else str(ob_id_fp[i]),
        "geom_wkb":                bytes(keep_geoms[i]),
        "centroid_lon":            float(c_lon[i]),
        "centroid_lat":            float(c_lat[i]),
        "height_m":                float(height_fp[i]),
        "insar_height_m":          None if not np.isfinite(insar_h[i]) else float(insar_h[i]),
        "insar_height_sigma_m":    None if not np.isfinite(insar_h_sigma[i]) else float(insar_h_sigma[i]),
        "fused_height_m":          float(fused_h[i]),
        "n_floors":                int(n_floors_fp[i]),
        "soil_class":              str(env["soil_class"][i]),
        "riparian_dist_m":         None if not np.isfinite(env["riparian_dist_m"][i])  else float(env["riparian_dist_m"][i]),
        "shoreline_dist_m":        None if not np.isfinite(env["shoreline_dist_m"][i]) else float(env["shoreline_dist_m"][i]),
        "reclaimed_land":          reclaimed_arr[i],
        "built_year":              int(built_year[i]),
        "classification":          int(classification[i]),
        "velocity_accel_mm_yr2":   None if not np.isfinite(accel[i]) else float(accel[i]),
        "trend_slope_mm_yr":       None if not np.isfinite(trend_slope_m[i]) else float(trend_slope_m[i]),
        "seasonal_amplitude_mm":   None if not np.isfinite(seasonal_amp_m[i]) else float(seasonal_amp_m[i]),
        "trend_r2":                None if not np.isfinite(trend_r2_m[i]) else float(trend_r2_m[i]),
        "failure_mode":            int(failure_mode_m[i]),
        "velocity_sigma_mm_yr":    None if not np.isfinite(v_sigma[i]) else float(v_sigma[i]),
        "velocity_ew_sigma_mm_yr": None if not np.isfinite(v_ew_sigma[i]) else float(v_ew_sigma[i]),
        "cohort_composite_pct":    int(composite_pct[i]),
        "cohort_shear_pct":        int(shear_pct[i]),
        "cohort_size":             int(cohort_size[i]),
        "block_id":                int(block_id[i]),
        "cohort_block_pct":        int(cohort_block_pct[i]),
        "closure_rms_rad":         None if not np.isfinite(closure_per_b[i]) else float(closure_per_b[i]),
        "dem_err_m":               None if not np.isfinite(dem_err_per_b[i]) else float(dem_err_per_b[i]),
        "dem_err_flag":            bool(dem_err_flag_per_b[i]),
    } for i in range(N)]

    b_dir = PARQUET_ROOT / "buildings" / f"aoi={aoi.code}"
    b_dir.mkdir(parents=True, exist_ok=True)
    b_tbl = pa.Table.from_pylist(rows_b, schema=BUILDINGS_SCHEMA)
    # Clean any stale file from prior runs (different filename → would coexist).
    for stale in b_dir.glob("part-0.parquet"):
        stale.unlink()
    pq.write_table(b_tbl, b_dir / "data.parquet", compression="zstd")
    print(f"  ✓ wrote {N} buildings → {b_dir/'data.parquet'}")

    # ---- Build subsidence table (long-format, schema-matched) -------------
    bid_col = np.repeat(keep_ids, T)
    dates_parsed = [date.fromisoformat(d) for d in dates]
    date_col = np.tile(dates_parsed, N)
    # disp shape (N, T); v_per_month (N, T); trend_disp (N, T) — flatten in row-major.
    disp_flat = disp.reshape(-1).astype(np.float64)
    trend_disp_flat = trend_disp_masked.reshape(-1).astype(np.float64)
    vpm_flat = v_per_month.reshape(-1).astype(np.float64)
    coh_flat = np.repeat(coh.astype(np.float64), T)
    s_tbl = pa.table({
        "building_id":                  pa.array(bid_col,          type=pa.int64()),
        "aoi_code":                     pa.array([aoi.code] * (N * T), type=pa.string()),
        "observation_date":             pa.array(date_col.tolist(), type=pa.date32()),
        "displacement_mm":              pa.array(disp_flat,        type=pa.float64()),
        "trend_displacement_mm":        pa.array(trend_disp_flat,  type=pa.float64()),
        "velocity_mm_yr":               pa.array(vpm_flat,         type=pa.float64()),
        "velocity_horizontal_ew_mm_yr": pa.array(np.full(N * T, np.nan, dtype=np.float64), type=pa.float64()),
        "coherence":                    pa.array(coh_flat,         type=pa.float64()),
    }, schema=SUBSIDENCE_SCHEMA)
    s_dir = PARQUET_ROOT / "subsidence" / f"aoi={aoi.code}"
    s_dir.mkdir(parents=True, exist_ok=True)
    for stale in s_dir.glob("part-0.parquet"):
        stale.unlink()
    pq.write_table(s_tbl, s_dir / "data.parquet", compression="zstd")
    print(f"  ✓ wrote {N*T} subsidence rows → {s_dir/'data.parquet'}")

    # ---- env_index parquet -------------------------------------------------
    e_dir = PARQUET_ROOT / "env_index" / f"aoi={aoi.code}"
    e_dir.mkdir(parents=True, exist_ok=True)
    for stale in e_dir.glob("part-0.parquet"):
        stale.unlink()
    pq.write_table(env_tbl, e_dir / "data.parquet", compression="zstd")
    print(f"  ✓ wrote {env_tbl.num_rows} env_index rows → {e_dir/'data.parquet'}")

    # ---- ARCHITECTURE_THREE B2 — coherence sparkline partition ------------
    # One row per building, one binary blob of T × 4 bytes (Float32) per row.
    # Frontend reads it as zero-copy Float32Array. n_epochs is stored as
    # parquet table metadata so the reader doesn't need to infer from
    # blob length.
    coh_tbl = pa.table(
        {
            "building_id": pa.array(keep_ids.tolist(), type=pa.int64()),
            "aoi_code":    pa.array([aoi.code] * N,    type=pa.string()),
            "coh_series":  pa.array(coh_blobs.tolist(), type=pa.binary()),
        },
        schema=COH_SERIES_SCHEMA,
    ).replace_schema_metadata({b"n_epochs": str(n_epochs_used).encode()})
    cs_dir = PARQUET_ROOT / "coh_series" / f"aoi={aoi.code}"
    cs_dir.mkdir(parents=True, exist_ok=True)
    for stale in cs_dir.glob("*.parquet"):
        stale.unlink()
    pq.write_table(coh_tbl, cs_dir / "data.parquet", compression="zstd")
    print(f"  ✓ wrote {N} coh_series rows (T={n_epochs_used}) → {cs_dir/'data.parquet'}")


def rebuild_demo_db() -> None:
    """Build data/demo.duckdb.new from init_db.sql, then atomic-replace
    data/demo.duckdb. Safe to call while FastAPI holds a read-only handle on
    the live file — the existing handle stays bound to the unlinked inode;
    new connections see the new data.
    """
    import duckdb

    sql_path = BACKEND_DIR / "scripts" / "init_db.sql"
    sql = sql_path.read_text().replace("${PARQUET_ROOT}", str(PARQUET_ROOT.resolve()))

    db_new = BACKEND_DIR / "data" / "demo.duckdb.new"
    db_new.unlink(missing_ok=True)
    con = duckdb.connect(str(db_new))
    con.execute(sql)
    # Smoke-check: every (view, aoi) must be non-empty.
    for view in ("buildings", "subsidence_time_series", "environmental_index"):
        for aoi in REGISTRY:
            n = con.execute(
                f"SELECT COUNT(*) FROM {view} WHERE aoi_code = ?", [aoi.code]
            ).fetchone()[0]
            if n == 0:
                con.close()
                raise RuntimeError(f"{view} empty for aoi_code={aoi.code}")
    con.close()
    os.replace(str(db_new), str(DB_PATH))
    print(f"  ✓ rebuilt {DB_PATH}")


def run_join(aoi: AOI, track: str) -> None:
    track_safe = track.replace("/", "_")
    run_dir = MINTPY_DIR / f"{aoi.code}_{track_safe}"
    fp_path = FOOTPRINT_DIR / f"{aoi.code}.parquet"
    if not run_dir.exists():
        sys.exit(f"missing MintPy run dir: {run_dir}. Run scripts/mintpy_run.py first.")
    if not fp_path.exists():
        sys.exit(f"missing footprints: {fp_path}. Run scripts/osm_footprints.py first.")

    print(f"  loading MintPy stack from {run_dir}…")
    stack = load_mintpy_stack(run_dir)
    print(f"    grid {stack.coherence.shape}, {len(stack.dates)} dates, "
          f"coherence median={np.nanmedian(stack.coherence):.2f}")

    print(f"  mapping footprints to InSAR pixels from {fp_path}…")
    pixel_index, keep_ids, keep_geoms = rasterize_footprints(fp_path, stack)
    print(f"    {len(keep_ids)} footprints inside AOI grid")

    print("  per-building lookup (coherence-gated)…")
    agg = aggregate(stack, pixel_index)
    n_dead = int(np.isnan(agg["velocity_mm_yr"]).sum())
    print(f"    {len(keep_ids) - n_dead} buildings retained, {n_dead} dropped (no coherent pixels)")

    print("  writing parquet…")
    emit_parquet(
        aoi, fp_path, keep_ids, keep_geoms, agg, stack.dates,
        run_dir=run_dir,
        stack_shape=stack.coherence.shape,
        pixel_index=pixel_index,
    )
    # Real InSAR products produced these partitions — flip this AOI's provenance
    # so the bundle/disclaimer stop calling it synthetic. (We only reach here if
    # the placeholder guard in load_mintpy_stack passed.)
    set_provenance(aoi.code, "insar")
    print(f"  ✓ provenance[{aoi.code}] = insar")


def main() -> None:
    p = argparse.ArgumentParser(description="Join MintPy outputs to footprints → GeoParquet")
    p.add_argument("--aoi", required=True, help="AOI code, e.g. huruma")
    p.add_argument("--track", default="ASCENDING/57",
                   help='"<flight>/<path>", default ASCENDING/57')
    p.add_argument("--rebuild-db", action="store_true",
                   help="after writing parquet, rebuild data/demo.duckdb via atomic swap")
    args = p.parse_args()
    run_join(by_code(args.aoi), args.track)
    if args.rebuild_db:
        rebuild_demo_db()


if __name__ == "__main__":
    main()
