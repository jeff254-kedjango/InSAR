"""
Stage 3 driver: MintPy SBAS time-series inversion.

Takes the HyP3 GAMMA products downloaded in Stage 2 and runs MintPy's
`smallbaselineApp.py` to produce a coherence-weighted time-series of vertical
displacement per pixel.

Why this is its own module rather than living in hyp3_pipeline.py:
  - MintPy needs its own conda environment (heavy GDAL/ISCE2/h5py deps).
  - We invoke it as a subprocess from the standard backend venv. Crossing the
    process boundary keeps the dependency graphs separate.

Run from backend/ (regular venv):
    python -m scripts.mintpy_run --aoi huruma --track ASCENDING/57

The wrapper:
  1. Renders `mintpy_config.tmpl` → `data/mintpy/<aoi>-<track>/config.cfg`
  2. Calls the mintpy conda env's smallbaselineApp.py as a subprocess
  3. Streams stdout/stderr to a log file and the terminal
  4. Returns the path to the geocoded velocity + timeseries HDF5

The conda env name and prefix are configurable via env vars (MINTPY_ENV,
MINTPY_PREFIX). Default is the env created by `setup_mintpy_env.sh`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.aois import AOI, by_code, processing_bbox

BACKEND_DIR = Path(__file__).resolve().parents[1]
# Where Stage 2 extracted the HyP3 products MintPy loads. Overridable via
# HYP3_WORK_DIR so we can point at the reprojected 4326 mirror tree
# (scripts/reproject_hyp3.py) without touching the originals — GACOS needs a
# geographic stack. Relative values resolve under BACKEND_DIR.
_work_env = os.environ.get("HYP3_WORK_DIR")
WORK_DIR = (
    (Path(_work_env) if Path(_work_env).is_absolute() else BACKEND_DIR / _work_env)
    if _work_env else BACKEND_DIR / "data" / "hyp3_work"
)
MINTPY_DIR = BACKEND_DIR / "data" / "mintpy"
GACOS_DIR = BACKEND_DIR / "data" / "raw" / "env" / "gacos"
TEMPLATE_PATH = Path(__file__).resolve().parent / "mintpy_config.tmpl"

MINTPY_PREFIX = Path(os.environ.get("MINTPY_PREFIX", str(Path.home() / "miniforge3"))).resolve()
MINTPY_ENV = os.environ.get("MINTPY_ENV", "mintpy")


@dataclass(frozen=True)
class MintpyOutputs:
    """Paths to the geocoded products MintPy emits. Stage 4 consumes these."""
    run_dir: Path
    velocity_h5: Path        # geo_velocity.h5 — per-pixel LOS velocity (m/yr → convert to mm/yr)
    timeseries_h5: Path      # geo_timeseries.h5 — per-pixel cumulative disp (m → mm)
    coherence_h5: Path       # geo_temporalCoherence.h5 — fit quality 0-1
    geometry_h5: Path        # geo_geometryRadar.h5 — lat/lon/inc/azi for each pixel


def _sar_dates(aoi: AOI) -> set[str]:
    """Distinct YYYYMMDD acquisition dates present in this AOI's HyP3 products.

    Each pair dir is named like `h-A57-240606240618` and emits
    `*_unw_phase.tif`; the two 6-digit halves are ref+sec acquisition dates.
    We read the filenames (O(#pairs), no raster opens) and expand to 8-digit
    dates. Used to diff against GACOS grids so we exclude only dates that truly
    lack a correction grid.
    """
    dates: set[str] = set()
    for tif in (WORK_DIR / aoi.code).glob("*/*_unw_phase.tif"):
        for token in re.findall(r"\d{8}", tif.name):
            dates.add(token)
    return dates


def _gacos_dates(aoi_gacos: Path) -> set[str]:
    """YYYYMMDD dates that have a GACOS `.ztd.tif` zenith-delay grid on disk."""
    return {
        m.group(0)
        for f in aoi_gacos.glob("*.ztd.tif")
        if (m := re.match(r"\d{8}", f.name))
    }


def _missing_gacos_dates(aoi: AOI, aoi_gacos: Path) -> list[str]:
    """SAR dates with no matching GACOS grid — must be excluded so the gacos
    troposphere step doesn't error on the gap. Empty list ⇒ render `auto`."""
    return sorted(_sar_dates(aoi) - _gacos_dates(aoi_gacos))


def _reference_lalo(aoi: AOI, minlon: float, minlat: float,
                    maxlon: float, maxlat: float) -> str:
    """Reference-point value for the template.

    Emits the explicit `lat,lon` anchor only when it lies INSIDE the subset
    bbox (so MintPy can actually pin to it); otherwise `auto` — an out-of-subset
    anchor would make MintPy warn and silently fall back to auto anyway, so we
    pick auto explicitly rather than ship a contradictory line. Phase B widens
    the subset to contain the anchor, at which point this returns the explicit
    coordinate with no other change.
    """
    inside = (minlat <= aoi.reference_lat <= maxlat
              and minlon <= aoi.reference_lon <= maxlon)
    if inside:
        return f"{aoi.reference_lat:.5f},{aoi.reference_lon:.5f}"
    return "auto"


def _snap_reference_lalo(
    run_dir: Path,
    minlon: float, minlat: float, maxlon: float, maxlat: float,
    *,
    coh_floor: float = 0.7,
    vel_abs_max_m_yr: float = 0.002,   # 2 mm/yr
) -> str | None:
    """Snap the reference point to a pixel that is coherent AND low-velocity AND
    inside a stable cluster, reading `avgSpatialCoh.h5` + `velocity.h5` from a
    MintPy run dir. Returns ``"lat,lon"`` or ``None``.

    Two-pass by design: on the FIRST run these products don't exist yet, so this
    returns ``None`` and the caller falls back to the AOI's fixed anchor. On a
    RE-RUN (smallbaselineApp is idempotent) the products are present and we pin a
    data-derived reference: high coherence is necessary but NOT sufficient — a
    γ=0.96 pixel can still be moving +17 mm/yr — so we additionally require
    |velocity| ≤ `vel_abs_max_m_yr`, and require the pixel to sit inside a
    contiguous stable patch (4-neighbour erosion) rather than be a lone lucky
    pixel. Picks max-coherence among survivors (tie-break min |velocity|).
    """
    import numpy as np  # local: keeps the module importable without numpy present
    try:
        import h5py
    except Exception:
        return None

    coh_path = run_dir / "avgSpatialCoh.h5"
    vel_path = run_dir / "velocity.h5"
    if not (coh_path.exists() and vel_path.exists()):
        return None
    try:
        with h5py.File(coh_path, "r") as f:
            key = "coherence" if "coherence" in f else list(f.keys())[0]
            coh = f[key][:]
        with h5py.File(vel_path, "r") as f:
            vel = f["velocity"][:]
            a = f.attrs
            y0 = float(a["Y_FIRST"]); x0 = float(a["X_FIRST"])
            dy = float(a["Y_STEP"]); dx = float(a["X_STEP"])
    except Exception:
        return None
    if coh.shape != vel.shape:
        return None

    mask = (coh >= coh_floor) & (np.abs(vel) <= vel_abs_max_m_yr) & np.isfinite(vel) & np.isfinite(coh)
    # Stable-cluster requirement: keep only pixels whose 4-neighbours are also in
    # the mask, so the anchor sits inside a contiguous stable patch.
    core = mask.copy()
    core[1:, :]  &= mask[:-1, :]
    core[:-1, :] &= mask[1:, :]
    core[:, 1:]  &= mask[:, :-1]
    core[:, :-1] &= mask[:, 1:]
    if not core.any():
        return None

    # Among clustered survivors: max coherence, tie-break min |velocity|.
    ys, xs = np.where(core)
    ry, rx = max(zip(ys, xs), key=lambda yx: (float(coh[yx]), -abs(float(vel[yx]))))
    ry, rx = int(ry), int(rx)
    lat = y0 + ry * dy
    lon = x0 + rx * dx
    if not (minlat <= lat <= maxlat and minlon <= lon <= maxlon):
        return None
    return f"{lat:.5f},{lon:.5f}"


def render_config(aoi: AOI, track: str, dest: Path, *, multilook: int = 1) -> Path:
    """Render `mintpy_config.tmpl` into the run directory with substitutions.

    The template intentionally uses {{double-brace}} tokens — Python str.format
    is too eager (would choke on legitimate `{` in MintPy syntax).

    `multilook` is the N×N load-time downsample factor (1 = full-res). It maps to
    mintpy.multilook.ystep/xstep in the template.
    """
    if multilook < 1:
        raise ValueError(f"multilook must be >= 1, got {multilook}")
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"missing template: {TEMPLATE_PATH}")
    # Use the PROCESSING box, not the display tile: it's wide enough to contain
    # the reference anchor (so reference.lalo resolves to the real coordinate
    # rather than `auto`) and it matches the Stage A clip extent the rasters were
    # cropped to. The 2 km display tile (bbox) stays for the UI/bundle only.
    minlon, minlat, maxlon, maxlat = processing_bbox(aoi)

    # Stage 2 extracts each HyP3 product into WORK_DIR/<aoi>/<job_name>/.
    # Job names are prefixed with the track letter+path (e.g. "h-A57-…"), so
    # when --all-tracks is off there's exactly one track flat under the AOI dir
    # and a simple wildcard is correct. For multi-track runs, MintPy will refuse
    # to mix geometries and we'd need to add a per-track subset step.
    text = TEMPLATE_PATH.read_text()
    # ARCHITECTURE_THREE A1+A3 — GACOS dir + explicit reference plumbed in.
    # Ensure the GACOS dir exists so MintPy's gacos step can list it without
    # error even when no grids are present yet (the template's fallback
    # comment covers the empty-dir case).
    aoi_gacos = GACOS_DIR / aoi.code
    aoi_gacos.mkdir(parents=True, exist_ok=True)

    # GACOS dates with no grid → excludeDate; "auto" when every date has one.
    missing = _missing_gacos_dates(aoi, aoi_gacos)
    exclude_dates = ",".join(missing) if missing else "auto"
    if missing:
        print(f"  excluding {len(missing)} date(s) with no GACOS grid: "
              f"{','.join(missing)}")

    # Prefer a data-derived stable anchor snapped from this run's own products
    # (coherent AND low-velocity AND clustered). On the first pass the products
    # don't exist yet → None → fall back to the AOI's fixed anchor; a re-run then
    # pins the snapped reference (smallbaselineApp re-fit is idempotent).
    snapped = _snap_reference_lalo(dest, minlon, minlat, maxlon, maxlat)
    if snapped is not None:
        reference_lalo = snapped
        print(f"  reference.lalo = {reference_lalo}  (snapped: coherent+stable+clustered)")
    else:
        reference_lalo = _reference_lalo(aoi, minlon, minlat, maxlon, maxlat)
        print(f"  reference.lalo = {reference_lalo}  (fixed anchor; no run products yet)")

    subs = {
        "work": str(WORK_DIR),
        "aoi": aoi.code,
        "minlat": f"{minlat:.5f}",
        "maxlat": f"{maxlat:.5f}",
        "minlon": f"{minlon:.5f}",
        "maxlon": f"{maxlon:.5f}",
        "reference_lalo": reference_lalo,
        "exclude_dates": exclude_dates,
        "gacos_dir": str(aoi_gacos),
        "multilook": str(multilook),
    }
    for k, v in subs.items():
        text = text.replace("{{" + k + "}}", v)

    dest.mkdir(parents=True, exist_ok=True)
    cfg = dest / "config.cfg"
    cfg.write_text(text)
    return cfg


def _conda_run_cmd(args: list[str]) -> list[str]:
    """Wrap a command to run inside the mintpy conda env without sourcing
    activate scripts (which require a login shell)."""
    conda_bin = MINTPY_PREFIX / "bin" / "conda"
    if not conda_bin.exists():
        raise FileNotFoundError(
            f"conda not found at {conda_bin}. Run scripts/setup_mintpy_env.sh first."
        )
    return [str(conda_bin), "run", "-n", MINTPY_ENV, "--no-capture-output", *args]


def run_mintpy(
    aoi: AOI, track: str = "ASCENDING/57", *, multilook: int = 1, run_suffix: str = ""
) -> MintpyOutputs:
    """Run smallbaselineApp end-to-end for one (aoi, track).

    Idempotent: MintPy re-uses cached intermediate HDF5s if the config and
    inputs haven't changed. Safe to interrupt and re-run.

    `multilook` (N) downsamples the stack N×N at load time for fast iteration.
    `run_suffix`, when set, isolates this run into a sibling dir
    `<aoi>_<track>_<suffix>` so a trial run never overwrites the canonical
    products (the live join/provenance key off the un-suffixed dir).
    """
    track_safe = track.replace("/", "_")
    base = f"{aoi.code}_{track_safe}"
    run_dir = MINTPY_DIR / (f"{base}_{run_suffix}" if run_suffix else base)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = render_config(aoi, track, run_dir, multilook=multilook)
    print(f"  config: {cfg}")
    if multilook > 1:
        print(f"  multilook: {multilook}×{multilook} downsample")

    log_path = run_dir / "smallbaselineApp.log"
    print(f"  running smallbaselineApp.py in conda env '{MINTPY_ENV}' (log: {log_path})")

    cmd = _conda_run_cmd(["smallbaselineApp.py", str(cfg), "--dir", str(run_dir)])
    # Stream output to both terminal and log file. tee-style behaviour without
    # spawning an extra process.
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"smallbaselineApp.py exited {rc}; see {log_path}")

    # MintPy writes geocoded outputs to two different places depending on input
    # coordinate system: a `geo/` subdir when geocoding from radar coords (e.g.
    # ISCE2 inputs), or flat in the run dir when inputs were already in GEO
    # (HyP3 GAMMA — what we use). Probe both, prefer whichever exists.
    geo_subdir = run_dir / "geo"
    if (geo_subdir / "geo_velocity.h5").exists():
        out = MintpyOutputs(
            run_dir=run_dir,
            velocity_h5=geo_subdir / "geo_velocity.h5",
            timeseries_h5=geo_subdir / "geo_timeseries.h5",
            coherence_h5=geo_subdir / "geo_temporalCoherence.h5",
            geometry_h5=geo_subdir / "geo_geometryRadar.h5",
        )
    else:
        out = MintpyOutputs(
            run_dir=run_dir,
            velocity_h5=run_dir / "velocity.h5",
            timeseries_h5=run_dir / "timeseries.h5",
            coherence_h5=run_dir / "temporalCoherence.h5",
            geometry_h5=run_dir / "inputs" / "geometryGeo.h5",
        )
    missing = [p for p in (out.velocity_h5, out.timeseries_h5) if not p.exists()]
    if missing:
        raise RuntimeError(f"MintPy finished but expected outputs missing: {missing}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run MintPy SBAS for one AOI/track")
    p.add_argument("--aoi", required=True, help="AOI code, e.g. huruma")
    p.add_argument("--track", default="ASCENDING/57",
                   help='"<flight>/<path>", e.g. "ASCENDING/57"')
    p.add_argument("--multilook", type=int, default=1,
                   help="N×N load-time downsample for fast iteration; "
                        "default 1 = full-res")
    p.add_argument("--run-suffix", default="",
                   help="append to the run-dir name to isolate a trial run "
                        "(e.g. 'clipped'); default writes the canonical dir")
    args = p.parse_args()

    aoi = by_code(args.aoi)
    out = run_mintpy(aoi, args.track, multilook=args.multilook,
                     run_suffix=args.run_suffix)
    print(f"\n  ✓ velocity:   {out.velocity_h5}")
    print(f"  ✓ timeseries: {out.timeseries_h5}")
    print(f"  ✓ coherence:  {out.coherence_h5}")
    print(f"  ✓ geometry:   {out.geometry_h5}")


if __name__ == "__main__":
    main()
