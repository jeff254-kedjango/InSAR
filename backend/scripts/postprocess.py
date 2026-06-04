"""
Shared post-processing for the buildings/subsidence/env_index pipeline.

This module is the bridge between the synthetic-data generator (`phenomena.py`)
and the real-InSAR join (`join_insar.py`). Every helper here is AOI-agnostic —
take in numpy arrays of velocity/coherence/displacement, return derived columns
that match the `init_db.sql` schema.

The one place this module *does* dispatch on AOI is `synthesize_env_context`,
which fabricates per-building environmental context (soil class, riparian or
shoreline distance, reclaimed-land flag, built year) for the real-InSAR path.

  IMPORTANT — honesty note. The fields produced by `synthesize_env_context` and
  `synthesize_env_index_rows` (`soil_class`, `riparian_dist_m`,
  `shoreline_dist_m`, `reclaimed_land`, `built_year`, `groundwater_anom`,
  `rainfall_anom_mm`, `ndvi_proxy`) are PLAUSIBLE-SYNTHETIC, not real. They
  reflect the same statistical distributions phenomena.py uses for the fully
  synthetic seed, but the values for any specific real building are made up —
  there is no soil map being consulted. Anything user-facing that quotes them
  must say so. (Once a real soil/river data source lands, replace
  `synthesize_env_context` with a real-data loader; the rest of this module is
  untouched.)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pyarrow as pa
from statsmodels.tsa.seasonal import STL

from .aois import AOI


# ============================================================================
# Classification + failure-mode codes (mirror frontend bundle.ts)
# ============================================================================

CLASS_INDETERMINATE         = 0
CLASS_CONFIRMED_THREAT      = 1
CLASS_ENV_NOISE             = 2
CLASS_STABLE_ANCHOR         = 3
CLASS_MIXED_SIGNAL          = 4
# A building whose displacement time series is not a court-defensible linear
# trend (low trend_r2 and/or high velocity σ). High coherence is necessary but
# not sufficient — only ~16% of high-γ Huruma pixels have R²≥0.7. We refuse to
# issue a confident safety verdict on the rest rather than fabricate one.
CLASS_INSUFFICIENT_EVIDENCE = 5

FAILURE_ELASTIC = 0
FAILURE_PLASTIC = 1

SOIL_CLASSES = ["black_cotton", "red_clay", "alluvial", "weathered_basalt", "coral_rag", "reclaim_fill"]


# ============================================================================
# Geometry helpers
# ============================================================================

def _meters_to_deg(at_lat: float) -> tuple[float, float]:
    dlat_per_m = 1.0 / 111_320.0
    dlon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(at_lat)))
    return dlon_per_m, dlat_per_m


def _deg_to_local_meters(lons: np.ndarray, lats: np.ndarray, aoi: AOI) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 centroids into AOI-local metres centred on `aoi.center_*`.

    This is the inverse of the local→WGS84 step phenomena.py's `_make_polygon`
    uses, so a centroid at the AOI centre maps to (0, 0) and the corners land
    at (±side_m/2, ±side_m/2). It lets `synthesize_env_context` reuse the same
    riparian-line / shoreline / soil-band physics phenomena.py wrote for the
    synthetic seed.
    """
    dlon_per_m, dlat_per_m = _meters_to_deg(aoi.center_lat)
    x_m = (lons - aoi.center_lon) / dlon_per_m
    y_m = (lats - aoi.center_lat) / dlat_per_m
    return x_m.astype(np.float64), y_m.astype(np.float64)


# ============================================================================
# InSAR-derived height (with footprint-area scaling)
# ============================================================================

def _insar_height(
    true_h: float,
    footprint_area_m2: float,
    noise_floor_m: float,
    rng: random.Random,
) -> tuple[float, float]:
    """Return (insar_height_m, sigma_m) for one footprint.

    Sentinel-1's coarse spatial resolution (~5 m × 20 m) means small
    footprints sample only a handful of phase pixels — the height inversion
    is noisy. A footprint that's, say, 8×8 m (64 m²) covers fewer than one
    full pixel and σ is high. A 30×20 m (600 m²) footprint covers many
    pixels and σ collapses toward the noise floor.

    Empirical model: σ = max(noise_floor, 30 / sqrt(area)).
    """
    sigma = max(noise_floor_m, 30.0 / math.sqrt(max(footprint_area_m2, 1.0)))
    return true_h + rng.gauss(0.0, sigma), sigma


def _fused_height(footprint_h: float, insar_h: float, insar_sigma: float) -> float:
    """Inverse-variance-weighted blend of the two estimates.

    Floor-count estimate has σ ≈ 1.5 m (uncertainty in floor height + counting
    floors from aerials). InSAR estimate has the per-building σ computed above.
    """
    sigma_floor = 1.5
    w_floor = 1.0 / (sigma_floor ** 2)
    w_insar = 1.0 / (insar_sigma ** 2)
    return (w_floor * footprint_h + w_insar * insar_h) / (w_floor + w_insar)


# ============================================================================
# Coherence-velocity classification (Principle 2 in ARCHITECTURE_TWO)
# ============================================================================

def _classify(
    v_subs: float,
    v_ew: float,
    gamma: float,
    trend_r2: float,
    sigma: float,
    r2_min: float,
    sigma_max: float,
) -> int:
    """Coherence-gated classification from end-of-series velocities.

    The framework's matrix: high velocity + low coherence = environmental
    surface noise (suppress). High velocity + high coherence = confirmed
    structural threat. Low velocity + high coherence = stable reference
    anchor.

    The V4 framework's table is incomplete: it doesn't say what to do with
    moderate velocity at borderline coherence (γ in 0.35–0.60), or with
    moderate velocity at high coherence. Those buildings would silently
    fall through to INDETERMINATE — a confidently-blank cell that gives a
    non-technical reader no way to distinguish "no signal" from "real
    signal we can't yet name." We add a 5th class, MIXED_SIGNAL, so the
    badge can honestly say "something's moving, watch this one."

    Integrity boundary (checked FIRST): a building whose displacement series is
    not a court-defensible linear trend — `trend_r2 < r2_min` and/or
    `sigma > sigma_max` — returns CLASS_INSUFFICIENT_EVIDENCE rather than any
    confident class. High coherence is necessary but not sufficient. The `not
    (x >= y)` form makes NaN (dead / failed-STL rows) fail the gate too.
    """
    if (not (trend_r2 >= r2_min)) or (not (sigma <= sigma_max)):
        return CLASS_INSUFFICIENT_EVIDENCE
    v_abs  = abs(v_subs)
    ew_abs = abs(v_ew)
    if gamma < 0.35 and (v_abs > 10.0 or ew_abs > 5.0):
        return CLASS_ENV_NOISE
    if v_abs < 1.5 and gamma > 0.60:
        return CLASS_STABLE_ANCHOR
    if (v_abs > 10.0 or ew_abs > 2.5) and gamma > 0.60:
        return CLASS_CONFIRMED_THREAT
    if (
        (0.35 <= gamma <= 0.60 and (v_abs > 5.0 or ew_abs > 2.5))
        or (gamma > 0.60 and (3.0 < v_abs <= 10.0 or 1.5 < ew_abs <= 2.5))
    ):
        return CLASS_MIXED_SIGNAL
    return CLASS_INDETERMINATE


# ============================================================================
# Composite risk (shear-weighted, classification-gated)
# ============================================================================

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def composite_risk(
    *,
    soil_class: str,
    riparian_dist_m: float | None,
    shoreline_dist_m: float | None,
    vel: float,
    v_ew: float,
    classification: int,
    fused_h_m: float,
    rng: random.Random,
) -> float:
    """Four-term composite. Drift contributes via a sigmoid with midpoint at
    the framework's 2.5 mm/yr threshold (we substitute sigmoid for the
    framework's "exponential" because exponentials in a [0,1] score blow up).

    Weights re-normalized from 0.55/0.25/0.20 → 0.35/0.25/0.20/0.20 so
    shear lands as a first-class term.

    Soil contribution is scaled by a load factor (Tier 2 of ARCHITECTURE_TWO,
    Principle 4): a 4-story and a 16-story building on the same alluvial clay
    no longer get the same soil score. The factor is `1 + h/10`, clamped at
    the input to soil score so the term stays in [0, 0.20].

    Gated by classification: ENV_NOISE buildings are dampened to 20% of
    their composite (kept findable, not hidden); STABLE_ANCHOR buildings
    are capped at 0.15.

    Sign convention: `vel` is in mm/yr with **negative = subsidence**, matching
    phenomena.py. Pass `0.0` (not NaN) when v_ew is unavailable — `abs(NaN)`
    is always False and would silently mis-classify the building.
    """
    subs_score  = min(1.0, max(0.0, -vel / 25.0))
    shear_score = _sigmoid((abs(v_ew) - 2.5) / 2.0)
    # Proximity decay — see phenomena.py for the full λ-calibration narrative.
    # riparian λ=400 m, shoreline λ=300 m. At most one of these is non-None per
    # AOI; if both happen to be None, the term contributes 0.
    if riparian_dist_m is not None:
        proximity_score = math.exp(-riparian_dist_m / 400.0)
    elif shoreline_dist_m is not None:
        proximity_score = math.exp(-shoreline_dist_m / 300.0)
    else:
        proximity_score = 0.0
    soil_lut = {
        "black_cotton": 0.9, "alluvial": 0.7, "red_clay": 0.4, "weathered_basalt": 0.1,
        "coral_rag": 0.15, "reclaim_fill": 0.85,
    }
    soil_score = soil_lut.get(soil_class, 0.3)
    load_factor = 1.0 + max(0.0, fused_h_m) / 10.0
    soil_score_loaded = min(1.0, soil_score * load_factor)
    composite = (
        0.35 * subs_score
        + 0.25 * shear_score
        + 0.20 * proximity_score
        + 0.20 * soil_score_loaded
    )
    composite = max(0.0, min(1.0, composite + rng.gauss(0, 0.03)))
    if classification == CLASS_ENV_NOISE:
        composite *= 0.2
    elif classification == CLASS_STABLE_ANCHOR:
        composite = min(composite, 0.15)
    elif classification == CLASS_MIXED_SIGNAL:
        composite *= 0.7
    elif classification == CLASS_INSUFFICIENT_EVIDENCE:
        # Not court-defensible: keep findable but don't let it top the threat
        # ranking on an undefendable velocity.
        composite *= 0.5
    return composite


# ============================================================================
# STL trend decoupling
# ============================================================================

def _stl_decompose(
    displacement: np.ndarray,
    period: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-building STL decomposition of the cumulative-displacement series.

    Returns:
        trend          (n_buildings, n_months) — robust LOESS trend component
        trend_slope    (n_buildings,)         — annualized slope of the trend, mm/yr
        seasonal_amp   (n_buildings,)         — peak-to-peak seasonal amplitude, mm
        trend_r2       (n_buildings,)         — 1 - var(resid)/var(displacement)
        failure_mode   (n_buildings,)         — uint8 ELASTIC/PLASTIC per row

    STL needs at least 2 full periods (n_months ≥ 2 × period). With period=12
    and n_months=24 this is the bare statistical floor; confidence intervals on
    trend slope are wide, and the UI is responsible for surfacing that caveat.
    """
    n, m = displacement.shape
    trend         = np.zeros((n, m), dtype=np.float32)
    trend_slope   = np.zeros(n,      dtype=np.float32)
    seasonal_amp  = np.zeros(n,      dtype=np.float32)
    trend_r2      = np.zeros(n,      dtype=np.float32)
    failure_mode  = np.zeros(n,      dtype=np.uint8)

    xs = np.arange(m, dtype=np.float64)
    x_mean = xs.mean()
    x_var = ((xs - x_mean) ** 2).sum()

    for i in range(n):
        y = displacement[i].astype(np.float64)
        try:
            res = STL(y, period=period, robust=True).fit()
        except Exception:
            continue
        t = res.trend
        s = res.seasonal
        r = res.resid
        trend[i, :] = t.astype(np.float32)
        y_mean_t = t.mean()
        slope_per_month = ((t - y_mean_t) * (xs - x_mean)).sum() / x_var if x_var > 0 else 0.0
        trend_slope[i] = float(slope_per_month * period)
        seasonal_amp[i] = float(s.max() - s.min())
        y_var = float(y.var())
        trend_r2[i] = float(1.0 - r.var() / y_var) if y_var > 1e-9 else 0.0
        if trend_slope[i] < -5.0 and trend_r2[i] > 0.85:
            failure_mode[i] = FAILURE_PLASTIC
        else:
            failure_mode[i] = FAILURE_ELASTIC

    return trend, trend_slope, seasonal_amp, trend_r2, failure_mode


# ============================================================================
# Velocity uncertainty propagation
# ============================================================================

# ============================================================================
# ARCHITECTURE_THREE B1/B2/B3 — extractors for MintPy diagnostic HDF5s
# ============================================================================
#
# Every extractor here is pure-numpy, vectorised, and tolerant of missing
# inputs. The MintPy run dir may not contain closurePhase.h5 / demErr.h5 /
# coherence_series.h5 on every config (e.g. when an older MintPy version
# names them differently, or the step was disabled). When the source is
# absent we return shape-matched NaN/zero arrays so the join step still
# ships — the UI will badge those columns as "not available" rather than
# the pipeline failing.

# DEM-residual threshold above which we flag a building "DEM-uncertain". Set
# to 15 m per ARCHITECTURE_THREE B3 — the typical magnitude at which the 30 m
# SRTM error meaningfully bleeds into apparent velocity at our 80 m looks.
DEM_ERR_FLAG_M: float = 15.0


def _h5_first_existing(path_candidates: Iterable, dataset_candidates: Iterable[str]):
    """Open the first HDF5 file in `path_candidates` that exists, returning
    (h5py.File, dataset_name) for the first dataset that exists inside it.
    Caller owns the file handle.

    Returns (None, None) when nothing matches — extractors then return
    safe NaN arrays.
    """
    import h5py
    for p in path_candidates:
        if p is None or not p.exists():
            continue
        try:
            f = h5py.File(p, "r")
        except Exception:
            continue
        for ds in dataset_candidates:
            if ds in f:
                return f, ds
        f.close()
    return None, None


def extract_closure_rms(run_dir, shape: tuple[int, int]) -> np.ndarray:
    """B1 — per-pixel closure-phase RMS (rad), shape (H, W).

    MintPy's `closure_phase_bias.py` writes `closurePhase.h5` with a
    `closurePhase` dataset (per-triplet residual stack) or, in newer
    versions, a precomputed `closurePhaseRMS` 2-D layer. We prefer the
    precomputed layer when present; otherwise we collapse the triplet
    stack to RMS along the triplet axis ourselves (vectorised, no Python
    loops).

    Returns float32, NaN where unavailable.
    """
    from pathlib import Path
    rd = Path(run_dir)
    candidates = [
        rd / "geo" / "geo_closurePhase.h5",
        rd / "closurePhase.h5",
        rd / "inputs" / "closurePhase.h5",
    ]
    f, ds = _h5_first_existing(candidates, ("closurePhaseRMS", "closurePhase"))
    if f is None:
        return np.full(shape, np.nan, dtype=np.float32)
    try:
        arr = np.asarray(f[ds], dtype=np.float32)
    finally:
        f.close()
    if arr.ndim == 2:
        rms = arr
    elif arr.ndim == 3:
        # (n_triplets, H, W) → RMS along axis 0. nanmean handles missing triplets.
        rms = np.sqrt(np.nanmean(arr * arr, axis=0)).astype(np.float32)
    else:
        return np.full(shape, np.nan, dtype=np.float32)
    if rms.shape != shape:
        # Shape mismatch (e.g. closurePhase in radar coords while velocity in
        # geo). Refuse to silently misalign — return NaN, log nothing here
        # (the join logs the column as missing).
        return np.full(shape, np.nan, dtype=np.float32)
    return rms


def extract_dem_err(run_dir, shape: tuple[int, int]) -> np.ndarray:
    """B3 — per-pixel DEM residual (m), shape (H, W). NaN where unavailable.

    MintPy's `correct_topography` step writes `demErr.h5` with a single
    `dem_error` dataset (some versions: `demError`). Sign convention is
    `actual - reference DEM` in metres.
    """
    from pathlib import Path
    rd = Path(run_dir)
    candidates = [
        rd / "geo" / "geo_demErr.h5",
        rd / "demErr.h5",
    ]
    f, ds = _h5_first_existing(candidates, ("dem_error", "demError"))
    if f is None:
        return np.full(shape, np.nan, dtype=np.float32)
    try:
        arr = np.asarray(f[ds], dtype=np.float32)
    finally:
        f.close()
    if arr.shape != shape:
        return np.full(shape, np.nan, dtype=np.float32)
    return arr


def extract_coh_per_epoch(run_dir, shape: tuple[int, int, int]) -> np.ndarray:
    """B2 — per-epoch spatial coherence stack, shape (T, H, W).

    MintPy's `temporalCoherence.h5` is a single 2-D map (post-inversion fit
    quality). The actual per-epoch coherence lives in the pre-inversion
    interferogram stack as `ifgramStack.h5::coherence` (shape
    (n_pairs, H, W)). To turn that into a per-epoch series we average the
    coherence of every interferogram that *touches* a given acquisition
    date. That's the standard per-epoch coherence definition used in
    InSAR review papers.

    Returns float32, NaN where unavailable.

    Cost: one h5py read + one numpy mean per epoch — O(T × n_pairs / T)
    in practice, no Python loop over pixels.
    """
    import h5py
    from pathlib import Path

    rd = Path(run_dir)
    T, H, W = shape
    # ifgramStack.h5 is always in radar coords pre-inversion; if MintPy
    # geocoded later, we still want the pre-inversion coherence here.
    candidates = [
        rd / "inputs" / "ifgramStack.h5",
        rd / "ifgramStack.h5",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return np.full(shape, np.nan, dtype=np.float32)

    try:
        f = h5py.File(src, "r")
    except Exception:
        return np.full(shape, np.nan, dtype=np.float32)
    try:
        if "coherence" not in f or "date" not in f:
            return np.full(shape, np.nan, dtype=np.float32)
        # date is (n_pairs, 2) bytes: each row [ref_date, sec_date] as YYYYMMDD.
        pair_dates = np.asarray(f["date"])  # bytes or str
        coh_stack = f["coherence"]  # h5py dataset; we'll slice per-pair below
        n_pairs = coh_stack.shape[0]
        h_pre, w_pre = coh_stack.shape[1], coh_stack.shape[2]
        if (h_pre, w_pre) != (H, W):
            # Shape mismatch — pre/post-geocode dims differ. Refuse.
            return np.full(shape, np.nan, dtype=np.float32)
        # Decode pair_dates to ISO strings → match against per-epoch dates.
        def _to_iso(b) -> str:
            s = b.decode() if isinstance(b, (bytes, bytearray)) else str(b)
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else s
        # Build (n_pairs, 2) of ISO strings, vectorised.
        ref = np.array([_to_iso(b) for b in pair_dates[:, 0]])
        sec = np.array([_to_iso(b) for b in pair_dates[:, 1]])
        # We need the caller's epoch ordering — pull it from the timeseries
        # dates. Read the matching `date` dataset from timeseries.h5 next
        # door, in the same conventions.
        ts_path = (rd / "geo" / "geo_timeseries.h5" if (rd / "geo" / "geo_timeseries.h5").exists()
                   else rd / "timeseries.h5")
        if not ts_path.exists():
            return np.full(shape, np.nan, dtype=np.float32)
        with h5py.File(ts_path, "r") as fts:
            ts_dates_raw = np.asarray(fts["date"])
        ts_dates = np.array([_to_iso(b) for b in ts_dates_raw])
        if ts_dates.size != T:
            return np.full(shape, np.nan, dtype=np.float32)
        # For each epoch t, mask of pairs that touch it.
        out = np.full((T, H, W), np.nan, dtype=np.float32)
        # Slurp the coherence stack in one read (float32; for our 2 km AOIs
        # n_pairs × H × W is well under 200 MB).
        coh_arr = np.asarray(coh_stack, dtype=np.float32)
        for t in range(T):
            d = ts_dates[t]
            mask = (ref == d) | (sec == d)
            if not mask.any():
                continue
            out[t] = np.nanmean(coh_arr[mask], axis=0)
        return out
    finally:
        f.close()


def pack_coh_series_per_building(
    coh_per_epoch_stack: np.ndarray,  # (T, H, W) float32
    pixel_rows: np.ndarray,           # (N,) int64
    pixel_cols: np.ndarray,           # (N,) int64
) -> tuple[np.ndarray, int]:
    """Gather (T,) per-epoch coherence at each building's pixel, pack into a
    flat float32 stream, return (binary_blobs, n_epochs).

    `binary_blobs` is an object array of length N where each element is a
    `bytes` of length 4 × T — exactly what the parquet `binary` column wants.
    The frontend reads `n_epochs` from the bundle header and slices each
    blob as a Float32Array.

    Vectorised gather over the (T,H,W) stack; no Python per-building loop
    over the time axis.
    """
    if coh_per_epoch_stack.size == 0 or pixel_rows.size == 0:
        return np.empty(0, dtype=object), int(coh_per_epoch_stack.shape[0]) if coh_per_epoch_stack.ndim == 3 else 0
    T = int(coh_per_epoch_stack.shape[0])
    # gather → shape (T, N), then transpose → (N, T)
    series = coh_per_epoch_stack[:, pixel_rows, pixel_cols].astype(np.float32, copy=False).T
    # All-NaN rows can come from dead pixels — keep them; the frontend
    # already knows to hide series for buildings with classification=0.
    contig = np.ascontiguousarray(series, dtype=np.float32)
    # Slice per row → bytes. `tobytes()` per row is the cheapest path in
    # numpy; no further packing helpers needed.
    blobs = np.empty(series.shape[0], dtype=object)
    row_nbytes = T * 4
    raw = contig.tobytes()
    for i in range(series.shape[0]):
        blobs[i] = raw[i * row_nbytes:(i + 1) * row_nbytes]
    return blobs, T


def _velocity_sigma_from_coherence(gamma: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Per-building σ on the velocity estimate, derived from end-of-series
    coherence.

    Empirical model: σ ≈ k * (1 - γ), calibrated so γ=0.9 → σ≈0.5 mm/yr
    (clean InSAR pixels) and γ=0.3 → σ≈3.5 mm/yr (decorrelated rooftops).
    """
    return np.clip(k * (1.0 - gamma), 0.05, 10.0).astype(np.float32)


# ============================================================================
# Court-defensibility gate (Tier-1 integrity boundary)
# ============================================================================

# R² floor is an ABSOLUTE constant: linear-fit quality is stack-independent, and
# 0.7 is the most court-defensible cut (keeps ~16% of real Huruma pixels — the
# only ones whose trend genuinely dominates the ~5 mm atmospheric noise floor).
DEFENSIBLE_R2_FLOOR: float = 0.7
# σ cap is DATA-DERIVED per-AOI: the noise floor differs per stack (1.2-1.9 mm/yr
# on the current Huruma run), so a hardcoded cap like 1.0 would gate everything.
DEFENSIBLE_SIGMA_PERCENTILE: float = 75.0


def defensibility_thresholds(
    v_sigma: np.ndarray,
    *,
    r2_floor: float = DEFENSIBLE_R2_FLOOR,
    sigma_pct: float = DEFENSIBLE_SIGMA_PERCENTILE,
) -> tuple[float, float]:
    """Return ``(r2_min, sigma_max)`` for this AOI's stack.

    ``r2_min`` is the absolute floor (fit quality is stack-independent).
    ``sigma_max`` is derived from THIS AOI's own σ distribution (its p75), so the
    gate adapts to the stack's noise floor instead of a constant that would zero
    everything. NaN-safe; falls back to ``+inf`` (σ-gate disabled) when no finite
    σ exists.
    """
    finite = np.isfinite(v_sigma)
    sigma_max = float(np.percentile(v_sigma[finite], sigma_pct)) if finite.any() else float("inf")
    return r2_floor, sigma_max


# ============================================================================
# Cohort percentile context
# ============================================================================

def _avg_rank_pct(vals: np.ndarray) -> np.ndarray:
    """Tie-aware average-rank percentile (0..100) of a 1-D array.

    Equal values share the mean of the ranks they span — so a cluster of
    identical scores all land on the same percentile rather than being split
    arbitrarily by sort order. Single-element input → [50].
    """
    k = vals.size
    if k == 1:
        return np.array([50], dtype=np.uint8)
    order = np.argsort(vals, kind="stable")
    ranked = vals[order]
    ranks = np.empty(k, dtype=np.float64)
    i0 = 0
    while i0 < k:
        i1 = i0 + 1
        while i1 < k and ranked[i1] == ranked[i0]:
            i1 += 1
        avg_rank = (i0 + i1 - 1) / 2.0
        ranks[order[i0:i1]] = avg_rank
        i0 = i1
    return (ranks / (k - 1) * 100.0).round().astype(np.uint8)


def _rank_within_groups(
    values: np.ndarray,
    group_id: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Per-element tie-aware percentile rank (0..100, uint8) computed *within*
    each integer group. Singleton groups → 50. Vectorized per group; the total
    work is O(n log n) across all groups.

    `group_id` is an int array in [0, n_groups). Empty groups are skipped.
    """
    n = values.size
    out = np.full(n, 50, dtype=np.uint8)
    # Bucket member indices by group in one pass.
    members: list[list[int]] = [[] for _ in range(n_groups)]
    gid = group_id.astype(np.int64, copy=False)
    for i in range(n):
        members[gid[i]].append(i)
    for grp in members:
        if len(grp) <= 1:
            continue  # singleton/empty already 50
        idx = np.asarray(grp, dtype=np.int64)
        out[idx] = _avg_rank_pct(values[idx])
    return out


def _cohort_percentiles(
    composite: np.ndarray,
    shear_abs: np.ndarray,
    fused_h_m: np.ndarray,
    soil_class: list[str],
    band_width_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-building percentile rank (0..100) of composite_risk and |v_ew|,
    computed within `height_band × soil_class` cohorts.

    Returns (composite_pct, shear_pct, cohort_size) — all uint8 except size,
    which is uint16 because a cohort can exceed 255 in dense AOIs.

    Singleton cohorts get percentile 50 (neither extreme justified with n=1).
    """
    n = len(composite)
    composite_pct = np.zeros(n, dtype=np.uint8)
    shear_pct     = np.zeros(n, dtype=np.uint8)
    cohort_size   = np.zeros(n, dtype=np.uint16)

    band = np.floor(np.clip(fused_h_m, 0.0, None) / band_width_m).astype(np.int32)
    soil_arr = np.asarray(soil_class, dtype=object)
    keys: dict[tuple[int, str], list[int]] = {}
    for i in range(n):
        keys.setdefault((int(band[i]), soil_arr[i]), []).append(i)

    for members in keys.values():
        idx = np.asarray(members, dtype=np.int64)
        if idx.size == 1:
            composite_pct[idx[0]] = 50
            shear_pct[idx[0]]     = 50
            cohort_size[idx[0]]   = 1
            continue
        composite_pct[idx] = _avg_rank_pct(composite[idx])
        shear_pct[idx]     = _avg_rank_pct(shear_abs[idx])
        cohort_size[idx]   = idx.size

    return composite_pct, shear_pct, cohort_size


# ============================================================================
# ARCHITECTURE_THREE C1/C4 — fixed-grid block aggregation
# ============================================================================
#
# Blocks tile each AOI into ~`target_m` squares in *degree* space (the AOIs are
# 2 km on a side, so a simple equirectangular grid is exact enough — no
# projection needed). A block is identified by a flat id `iy*nx + ix`. This is a
# pure function of building centroids: no new dependency, deterministic, and the
# whole grid for an AOI fits in a few hundred cells, so aggregation is trivial.

# Sentinel block id for a centroid that somehow falls outside the bbox (clamped
# in practice, so this is defensive only).
BLOCK_ID_NONE = np.uint16(0xFFFF)


def _block_grid_meta(bbox: tuple[float, float, float, float], target_m: float = 170.0) -> dict:
    """Grid descriptor for an AOI bbox: cell size in degrees + column/row counts.

    `target_m` is the desired block edge in metres; we convert to degrees at the
    bbox centre latitude (lon degrees shrink with cos(lat)). nx/ny are chosen so
    the grid covers the bbox with at least one cell.
    """
    minlon, minlat, maxlon, maxlat = bbox
    mid_lat = (minlat + maxlat) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(mid_lat))
    cell_lat_deg = target_m / m_per_deg_lat
    cell_lon_deg = target_m / m_per_deg_lon
    nx = max(1, int(math.ceil((maxlon - minlon) / cell_lon_deg)))
    ny = max(1, int(math.ceil((maxlat - minlat) / cell_lat_deg)))
    return {
        "minlon": float(minlon), "minlat": float(minlat),
        "cell_lon_deg": float(cell_lon_deg), "cell_lat_deg": float(cell_lat_deg),
        "nx": int(nx), "ny": int(ny),
    }


def assign_blocks(
    c_lon: np.ndarray,
    c_lat: np.ndarray,
    bbox: tuple[float, float, float, float],
    target_m: float = 170.0,
) -> tuple[np.ndarray, dict]:
    """Assign each (lon,lat) centroid to a fixed-grid block.

    Returns (block_id uint16 [n], grid_meta). `block_id = iy*nx + ix`, with ix/iy
    clamped into [0,nx)/[0,ny) so edge points land in the boundary cell rather
    than overflowing. n_blocks = nx*ny is always < 65535 for these AOIs, so
    uint16 is safe.
    """
    meta = _block_grid_meta(bbox, target_m)
    nx, ny = meta["nx"], meta["ny"]
    ix = np.floor((c_lon - meta["minlon"]) / meta["cell_lon_deg"]).astype(np.int64)
    iy = np.floor((c_lat - meta["minlat"]) / meta["cell_lat_deg"]).astype(np.int64)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    block_id = (iy * nx + ix).astype(np.uint16)
    return block_id, meta


def aggregate_blocks(
    block_id: np.ndarray,
    n_blocks: int,
    vel_end: np.ndarray,
    composite: np.ndarray,
    classification: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-block rollups, indexed [0, n_blocks).

    Returns dict of dense per-block arrays:
      count            (int32)   buildings in the block
      worst_velocity   (float32) most-negative end velocity (mm/yr); 0 if empty
      mean_risk        (float32) mean composite_risk; 0 if empty
      max_risk         (float32) max composite_risk; 0 if empty
      confirmed        (int32)   # buildings classified CONFIRMED_THREAT

    All vectorized with np.add/minimum.at — O(n_buildings), no Python per-block
    loop.
    """
    gid = block_id.astype(np.int64, copy=False)
    count = np.zeros(n_blocks, dtype=np.int32)
    np.add.at(count, gid, 1)

    sum_risk = np.zeros(n_blocks, dtype=np.float64)
    np.add.at(sum_risk, gid, composite.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_risk = np.where(count > 0, sum_risk / np.maximum(count, 1), 0.0).astype(np.float32)

    max_risk = np.zeros(n_blocks, dtype=np.float32)
    np.maximum.at(max_risk, gid, composite.astype(np.float32))

    # Worst (most-negative) velocity. Seed with +inf so minimum.at picks the real
    # min, then replace untouched (+inf) cells with 0.
    worst = np.full(n_blocks, np.inf, dtype=np.float32)
    np.minimum.at(worst, gid, vel_end.astype(np.float32))
    worst[~np.isfinite(worst)] = 0.0

    confirmed = np.zeros(n_blocks, dtype=np.int32)
    is_conf = (classification == CLASS_CONFIRMED_THREAT).astype(np.int32)
    np.add.at(confirmed, gid, is_conf)

    return {
        "count": count,
        "worst_velocity": worst,
        "mean_risk": mean_risk,
        "max_risk": max_risk,
        "confirmed": confirmed,
    }


def _acceleration_mm_yr2(velocity_per_month: np.ndarray, lookback: int = 6) -> np.ndarray:
    """Annualized change in trailing-12mo velocity over the trailing `lookback`
    months. Shape: [n_buildings]. Sign: negative = accelerating subsidence.
    """
    n, m = velocity_per_month.shape
    if m <= lookback:
        return np.zeros(n, dtype=np.float32)
    v_now   = velocity_per_month[:, -1]
    v_prior = velocity_per_month[:, -lookback - 1]
    return ((v_now - v_prior) * (12.0 / lookback)).astype(np.float32)


def _trailing_velocity(cumulative: np.ndarray, window: int = 12) -> np.ndarray:
    """For each (building, month_t), linear slope of displacement over the trailing `window` months,
    annualized to mm/yr. Vectorized; O(n_buildings × n_months)."""
    n, m = cumulative.shape
    out = np.zeros_like(cumulative)
    for t in range(m):
        lo = max(0, t - window + 1)
        seg = cumulative[:, lo:t + 1]
        k = seg.shape[1]
        if k < 2:
            out[:, t] = 0.0
            continue
        xs = np.arange(k)
        x_mean = xs.mean()
        y_mean = seg.mean(axis=1)
        num = ((seg - y_mean[:, None]) * (xs - x_mean)).sum(axis=1)
        den = ((xs - x_mean) ** 2).sum()
        slope_per_month = num / den
        out[:, t] = slope_per_month * 12.0
    return out


# ============================================================================
# Environmental-context synthesis for the real-InSAR path
# ============================================================================

def _huruma_env_for_xy(x_m: float, y_m: float, rng: random.Random) -> dict:
    """Mirror of `_huruma_buildings`'s env block: same riparian-line model,
    same soil-class distribution by distance band, same built-year range."""
    ripa = abs(0.3 * x_m - y_m + 200.0) / math.sqrt(0.3 ** 2 + 1.0)
    if ripa < 200:
        soil = rng.choices(["black_cotton", "alluvial", "red_clay"], weights=[5, 4, 1])[0]
    elif ripa < 600:
        soil = rng.choices(["black_cotton", "alluvial", "red_clay", "weathered_basalt"], weights=[3, 2, 3, 1])[0]
    else:
        soil = rng.choices(["black_cotton", "alluvial", "red_clay", "weathered_basalt"], weights=[1, 1, 3, 4])[0]
    return {
        "soil_class":       soil,
        "riparian_dist_m":  ripa,
        "shoreline_dist_m": None,
        "reclaimed_land":   None,
        "built_year":       rng.randint(1995, 2023),
    }


def _mombasa_env_for_xy(x_m: float, y_m: float, rng: random.Random) -> dict:
    """Mirror of `_mombasa_buildings`'s env block: same shoreline model
    (north-south at x = -700 m) for the proximity band, but `reclaimed_land` is
    keyed on the drawn soil (`reclaim_fill`), matching the real-data rule in
    phenomena.py — engineered fill IS the reclaimed-land signal. This keeps the
    synthetic env path internally consistent with the SoilGrids-keyed seed path,
    so the reclaim cohort is non-empty regardless of where the shoreline line
    falls relative to real footprints."""
    shoreline_dist = x_m - (-700.0)
    if shoreline_dist < 250.0:
        soil = rng.choices(["reclaim_fill", "alluvial"], weights=[7, 1])[0]
    elif shoreline_dist < 800:
        soil = rng.choices(["coral_rag", "alluvial", "red_clay"], weights=[5, 3, 2])[0]
    else:
        soil = rng.choices(["coral_rag", "red_clay"], weights=[7, 3])[0]
    reclaimed = soil == "reclaim_fill"
    return {
        "soil_class":       soil,
        "riparian_dist_m":  None,
        "shoreline_dist_m": max(0.0, shoreline_dist),
        "reclaimed_land":   reclaimed,
        "built_year":       rng.randint(1960, 2022),
    }


_ENV_DISPATCH = {
    "informal_settlement_subsidence": _huruma_env_for_xy,
    "coastal_subsidence":             _mombasa_env_for_xy,
}


def synthesize_env_context(
    aoi: AOI,
    centroid_lons: np.ndarray,
    centroid_lats: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Per-building env context (soil, distances, reclaimed-land, built year).

    Outputs are **plausible-synthetic** — see module docstring. The function
    re-projects real WGS84 centroids into AOI-local metres so the same
    riparian-line / shoreline geometry phenomena.py uses for the synthetic
    seed applies to real footprints too.

    Determinism: keyed on `seed` and building index. Stable across reruns of
    join_insar; will reshuffle if the footprint set changes between runs.
    """
    pheno_fn = _ENV_DISPATCH.get(aoi.phenomenon)
    if pheno_fn is None:
        raise ValueError(f"unknown phenomenon for env synthesis: {aoi.phenomenon}")

    x_m, y_m = _deg_to_local_meters(np.asarray(centroid_lons), np.asarray(centroid_lats), aoi)
    n = x_m.size
    rng = random.Random(seed)

    soil_class       = np.empty(n, dtype=object)
    riparian_dist_m  = np.full(n, np.nan, dtype=np.float64)
    shoreline_dist_m = np.full(n, np.nan, dtype=np.float64)
    reclaimed_land   = np.full(n, False, dtype=np.bool_)
    reclaimed_mask   = np.full(n, False, dtype=np.bool_)
    built_year       = np.zeros(n, dtype=np.int32)

    for i in range(n):
        env = pheno_fn(float(x_m[i]), float(y_m[i]), rng)
        soil_class[i] = env["soil_class"]
        if env["riparian_dist_m"] is not None:
            riparian_dist_m[i] = env["riparian_dist_m"]
        if env["shoreline_dist_m"] is not None:
            shoreline_dist_m[i] = env["shoreline_dist_m"]
        if env["reclaimed_land"] is not None:
            reclaimed_land[i] = bool(env["reclaimed_land"])
            reclaimed_mask[i] = True
        built_year[i] = int(env["built_year"])

    return {
        "soil_class":         soil_class,
        "riparian_dist_m":    riparian_dist_m,
        "shoreline_dist_m":   shoreline_dist_m,
        "reclaimed_land":     reclaimed_land,
        # Mask: which buildings actually have a meaningful reclaimed_land value.
        # Used downstream to decide whether to write True/False vs NULL.
        "reclaimed_land_set": reclaimed_mask,
        "built_year":         built_year,
    }


def synthesize_env_index_rows(
    *,
    aoi_code: str,
    building_ids: np.ndarray,
    soil_class: np.ndarray,
    riparian_dist_m: np.ndarray,
    shoreline_dist_m: np.ndarray,
    vel: np.ndarray,
    v_ew: np.ndarray,
    classification: np.ndarray,
    fused_h_m: np.ndarray,
    periods: list[date],
    seed: int,
) -> tuple[pa.Table, np.ndarray]:
    """Build the env_index parquet rows: one row per (building × quarter).

    Returns (table, composite_latest) where `composite_latest` is the
    per-building composite_risk from the final period — used downstream by
    `_cohort_percentiles` to compute peer rankings.

    Synthetic columns (`groundwater_anom`, `rainfall_anom_mm`, `ndvi_proxy`)
    use the same noise distributions phenomena.py uses; document this in any
    UI that surfaces them.
    """
    n = len(building_ids)
    rng = random.Random(seed)
    rows: list[dict] = []
    composite_latest = np.zeros(n, dtype=np.float32)

    for i in range(n):
        soil_i  = str(soil_class[i])
        ripa_i  = None if not np.isfinite(riparian_dist_m[i])  else float(riparian_dist_m[i])
        shor_i  = None if not np.isfinite(shoreline_dist_m[i]) else float(shoreline_dist_m[i])
        vel_i   = float(vel[i])   if np.isfinite(vel[i])   else 0.0
        vew_i   = float(v_ew[i])  if np.isfinite(v_ew[i])  else 0.0
        cls_i   = int(classification[i])
        fh_i    = float(fused_h_m[i]) if np.isfinite(fused_h_m[i]) else 0.0
        last_q = 0.0
        for q in periods:
            comp = composite_risk(
                soil_class=soil_i,
                riparian_dist_m=ripa_i,
                shoreline_dist_m=shor_i,
                vel=vel_i,
                v_ew=vew_i,
                classification=cls_i,
                fused_h_m=fh_i,
                rng=rng,
            )
            last_q = comp
            rows.append({
                "building_id":      int(building_ids[i]),
                "aoi_code":         aoi_code,
                "period_start":     q,
                "groundwater_anom": rng.gauss(0, 1),
                "rainfall_anom_mm": rng.gauss(0, 20),
                "ndvi_proxy":       rng.uniform(0.1, 0.5),
                "composite_risk":   comp,
            })
        composite_latest[i] = last_q

    return pa.Table.from_pylist(rows, schema=ENV_SCHEMA), composite_latest


# ============================================================================
# Arrow schemas — single source of truth shared by phenomena.py and join_insar.py
# ============================================================================

BUILDINGS_SCHEMA = pa.schema([
    ("building_id",            pa.int64()),
    ("aoi_code",               pa.string()),
    ("footprint_source",       pa.string()),
    ("osm_id",                 pa.int64()),
    ("open_buildings_id",      pa.string()),
    ("geom_wkb",               pa.binary()),
    ("centroid_lon",           pa.float64()),
    ("centroid_lat",           pa.float64()),
    ("height_m",               pa.float64()),
    ("insar_height_m",         pa.float64()),
    ("insar_height_sigma_m",   pa.float64()),
    ("fused_height_m",         pa.float64()),
    ("n_floors",               pa.int32()),
    ("soil_class",             pa.string()),
    ("riparian_dist_m",        pa.float64()),
    ("shoreline_dist_m",       pa.float64()),
    ("reclaimed_land",         pa.bool_()),
    ("built_year",             pa.int32()),
    # Tier 1: coherence-velocity classification + accel.
    ("classification",         pa.uint8()),
    ("velocity_accel_mm_yr2",  pa.float64()),
    # Tier 2: STL trend decomposition outputs.
    ("trend_slope_mm_yr",      pa.float64()),
    ("seasonal_amplitude_mm",  pa.float64()),
    ("trend_r2",               pa.float64()),
    ("failure_mode",           pa.uint8()),
    # Tier 3: velocity σ and cohort percentile context.
    ("velocity_sigma_mm_yr",   pa.float64()),
    ("velocity_ew_sigma_mm_yr", pa.float64()),
    ("cohort_composite_pct",   pa.uint8()),
    ("cohort_shear_pct",       pa.uint8()),
    ("cohort_size",            pa.uint16()),
    # ARCHITECTURE_THREE C1/C4 — fixed-grid block membership + block-relative
    # cohort percentile. block_id = iy*nx + ix (uint16; nx/ny from the AOI bbox,
    # see assign_blocks). cohort_block_pct = percentile rank of this building's
    # latest composite_risk *within its own block* (singletons → 50).
    ("block_id",               pa.uint16()),
    ("cohort_block_pct",       pa.uint8()),
    # ARCHITECTURE_THREE B1 — closure-phase RMS per pixel (rad). Surfaces as
    # "atmospheric noise: low/med/high" badge. High = residual tropo/decorr,
    # NOT geometry change.
    ("closure_rms_rad",        pa.float32()),
    # ARCHITECTURE_THREE B3 — DEM residual estimate (m) from MintPy's joint
    # velocity+DEM solve. |residual| > 15 m → flag this building "DEM-uncertain"
    # in the UI; the apparent velocity is partly DEM artefact, not deformation.
    ("dem_err_m",              pa.float32()),
    ("dem_err_flag",           pa.bool_()),
])

# ARCHITECTURE_THREE B2 — coherence sparkline. One row per building, holding a
# fixed-length packed Float32 binary of `n_epochs` values. Stored as a
# `binary` column rather than `list<float32>` so the read path is a single
# zero-copy memcpy into a JS Float32Array. `n_epochs` is the count for the AOI;
# the frontend reads it from the bundle header and reshapes accordingly.
COH_SERIES_SCHEMA = pa.schema([
    ("building_id",  pa.int64()),
    ("aoi_code",     pa.string()),
    ("coh_series",   pa.binary()),   # raw little-endian Float32, length = 4 × n_epochs
])

SUBSIDENCE_SCHEMA = pa.schema([
    ("building_id",                  pa.int64()),
    ("aoi_code",                     pa.string()),
    ("observation_date",             pa.date32()),
    ("displacement_mm",              pa.float64()),
    ("trend_displacement_mm",        pa.float64()),
    ("velocity_mm_yr",               pa.float64()),
    ("velocity_horizontal_ew_mm_yr", pa.float64()),
    ("coherence",                    pa.float64()),
])

ENV_SCHEMA = pa.schema([
    ("building_id",      pa.int64()),
    ("aoi_code",         pa.string()),
    ("period_start",     pa.date32()),
    ("groundwater_anom", pa.float64()),
    ("rainfall_anom_mm", pa.float64()),
    ("ndvi_proxy",       pa.float64()),
    ("composite_risk",   pa.float64()),
])
