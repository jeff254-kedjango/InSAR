"""
Build-time InSAR pipeline.

This runs OFFLINE relative to the demo: kick it off ahead of time, let it bake,
commit the resulting GeoParquet/DuckDB seed into the repo. The demo app never
reaches out to ASF.

Pipeline:
    1. asf_search    → discover S1 SLC scenes covering the AOI over the 24-month window
    2. ASF HyP3      → submit InSAR_GAMMA jobs for sequential scene pairs
    3. poll          → wait for completion, download zips
    4. MintPy SBAS   → smallbaselineApp.py on the stack to get velocities & cumulative displ.
    5. join          → spatial-join MintPy points to OSM building footprints
    6. emit          → write GeoParquet + load into DuckDB

This file is a SKELETON. Each step contains the calls you need; comments mark the
spots that require credentials, real disk space, and time. Don't run this on a
laptop you also need for slides — MintPy is RAM-hungry.

Credentials needed (set via env or ~/.netrc):
    EARTHDATA_USER, EARTHDATA_PASS   (NASA Earthdata Login)
    HYP3 uses the same credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from scripts.aois import AOI, REGISTRY, by_code, bbox

START = date(2024, 6, 1)
END   = date(2026, 5, 31)
WORK_DIR = Path(__file__).resolve().parents[1] / "data" / "hyp3_work"
PARQUET_DIR = Path(__file__).resolve().parents[1] / "data" / "parquet"

# HyP3 enforces a 20-character cap on `name`. Job names are the only resume
# handle, so they MUST be deterministic from (aoi, flight, path, scene-pair).
# We pack: <aoi[:4]>-<dir[0]><path>-<ref8>-<sec8>, e.g. "huru-A57-20240606-20240618"
# would be 28 chars — too long. Compress to ref/sec as YYMMDD: "huru-A57-240606240618" = 21.
# Drop the inner dash → "huru-A57-240606-240618" = 22. Still over.
# Final format: "h-A57-240606240618" = 18 chars. Single-letter AOI code prefix.
_AOI_LETTER = {"huruma": "h", "mombasa": "m"}


def _job_name(aoi_code: str, flight: str, path: int, ref: datetime, sec: datetime) -> str:
    letter = _AOI_LETTER.get(aoi_code, aoi_code[:1])
    d = flight[:1]  # "A" or "D"
    return f"{letter}-{d}{path}-{ref:%y%m%d}{sec:%y%m%d}"


def _aoi_wkt(aoi: AOI) -> str:
    """Closed-ring WKT polygon for the AOI bounding box."""
    minlon, minlat, maxlon, maxlat = bbox(aoi)
    return (
        f"POLYGON(({minlon} {minlat}, {maxlon} {minlat}, {maxlon} {maxlat}, "
        f"{minlon} {maxlat}, {minlon} {minlat}))"
    )


@dataclass(frozen=True)
class Scene:
    name: str
    start_time: datetime
    path: int
    flight_direction: str  # "ASCENDING" or "DESCENDING"


@dataclass(frozen=True)
class ScenePair:
    reference: str
    secondary: str
    # Carried through so the rest of the pipeline can keep ASC and DESC stacks
    # separate when running MintPy.
    path: int
    flight_direction: str
    # Start times let us build deterministic, resume-safe HyP3 job names
    # without re-querying ASF on restart.
    reference_time: datetime
    secondary_time: datetime


def search_scenes(aoi: AOI, start: date = START, end: date = END) -> list[Scene]:
    """Step 1. Find every S1 SLC scene over the AOI in the window.

    Returns scenes with enough metadata to build per-track pair chains in
    `make_pairs`. We intentionally keep both ascending and descending tracks
    here — vector decomposition of LOS into vertical + east-west requires both.
    """
    import asf_search as asf

    results = asf.geo_search(
        intersectsWith=_aoi_wkt(aoi),
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.SLC,
        beamMode=asf.BEAMMODE.IW,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    scenes: list[Scene] = []
    for r in results:
        p = r.properties
        ts = p.get("startTime")
        # asf_search returns ISO 8601 with trailing Z
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.min
        scenes.append(
            Scene(
                name=p["sceneName"],
                start_time=dt,
                path=int(p.get("pathNumber", -1)),
                flight_direction=p.get("flightDirection", "?"),
            )
        )
    scenes.sort(key=lambda s: (s.flight_direction, s.path, s.start_time))
    return scenes


def make_pairs(
    scenes: list[Scene],
    max_temporal_baseline_days: int = 24,
) -> list[ScenePair]:
    """Step 2a. Build sequential pairs grouped by (flight_direction, path).

    Mixing ascending and descending in a single InSAR pair is nonsense — the LOS
    geometry differs. Mixing two different relative orbits over the same AOI is
    equally bad: the imaging geometry shifts and the interferogram falls apart.
    So we partition by (flight_direction, path) and chain within each group.

    `max_temporal_baseline_days` protects against decorrelation: pairs longer
    than ~24 days lose coherence quickly in vegetated areas.
    """
    by_track: dict[tuple[str, int], list[Scene]] = defaultdict(list)
    for s in scenes:
        by_track[(s.flight_direction, s.path)].append(s)

    pairs: list[ScenePair] = []
    for (flight, path), group in by_track.items():
        group.sort(key=lambda s: s.start_time)
        for i in range(len(group) - 1):
            ref, sec = group[i], group[i + 1]
            dt_days = (sec.start_time - ref.start_time).days
            if dt_days <= 0 or dt_days > max_temporal_baseline_days:
                continue
            pairs.append(
                ScenePair(
                    reference=ref.name,
                    secondary=sec.name,
                    path=path,
                    flight_direction=flight,
                    reference_time=ref.start_time,
                    secondary_time=sec.start_time,
                )
            )
    return pairs


# Default track selection per AOI. Picked from Stage-1 plan: longest ASC chain
# per AOI gives a clean LOS series back to 2024-06; DESC tracks only go back to
# 2025-04 so we leave them as opt-in via --all-tracks.
DEFAULT_TRACKS: dict[str, list[tuple[str, int]]] = {
    "huruma":  [("ASCENDING", 57)],
    "mombasa": [("ASCENDING", 159)],
}


def filter_to_tracks(pairs: list[ScenePair], tracks: list[tuple[str, int]]) -> list[ScenePair]:
    """Keep only pairs matching one of the given (flight_direction, path) tracks."""
    keep = set(tracks)
    return [p for p in pairs if (p.flight_direction, p.path) in keep]


def _connect_hyp3():
    """Build an authenticated HyP3 client. Reads .env on first call."""
    from dotenv import load_dotenv
    from hyp3_sdk import HyP3
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    user = os.environ.get("EARTHDATA_USER", "").strip()
    pw = os.environ.get("EARTHDATA_PASS", "").strip()
    if not user or not pw:
        raise RuntimeError("EARTHDATA_USER / EARTHDATA_PASS missing — populate backend/.env")
    return HyP3(username=user, password=pw)


def submit_hyp3(
    aoi_code: str,
    pairs: list[ScenePair],
    hyp3=None,
):
    """Step 2b. Submit InSAR_GAMMA jobs idempotently.

    HyP3 *is* the manifest: each pair gets a deterministic name derived from
    (aoi, flight, path, ref-date, sec-date). On restart we ask HyP3 which job
    names already exist and submit only the missing ones. This is O(1) network
    calls regardless of pair count — far cheaper than a local JSON manifest
    that can drift out of sync with the server.

    Returns a hyp3_sdk.Batch containing every job for this AOI (resumed +
    newly submitted), suitable for handing to `wait_and_download`.
    """
    from hyp3_sdk import Batch

    if hyp3 is None:
        hyp3 = _connect_hyp3()

    # Build the target name set once.
    name_to_pair: dict[str, ScenePair] = {
        _job_name(aoi_code, p.flight_direction, p.path, p.reference_time, p.secondary_time): p
        for p in pairs
    }

    # `find_jobs` returns every InSAR job we've ever submitted. Filter to this
    # AOI by name prefix, then index by exact name. We don't filter server-side
    # by name because HyP3's `name=` filter is exact-match per call.
    letter = _AOI_LETTER.get(aoi_code, aoi_code[:1])
    existing = hyp3.find_jobs(job_type="INSAR_GAMMA")
    existing_by_name = {j.name: j for j in existing if j.name and j.name.startswith(f"{letter}-")}

    resumed: list = []
    todo: list[ScenePair] = []
    for name, pair in name_to_pair.items():
        if name in existing_by_name:
            resumed.append(existing_by_name[name])
        else:
            todo.append(pair)

    print(f"  resumed {len(resumed)} existing jobs, submitting {len(todo)} new pairs")

    # Submit the missing pairs. We use the batched submit so HyP3 only roundtrips
    # once per pair — there's no public bulk endpoint, and parallel HTTP from
    # the client would only hammer the rate limiter. The SDK serializes well
    # because each call is sub-second.
    new_batch = Batch()
    for attempt in range(3):
        try:
            print(f"   -> Submitting pair {i+1}/{len(todo)}: {name} (Attempt {attempt+1}/3)")
            b = hyp3.submit_insar_job(
                granule1=pair.reference,
                granule2=pair.secondary,
                name=name,
                looks="20x4",                      # 80 m × 80 m pixels — right size for building-scale InSAR
                include_displacement_maps=True,    # LOS + vertical displacement, headline product
                include_inc_map=True,              # needed for ASC/DESC vector decomposition
                include_dem=True,                  # MintPy needs the SRTM clip
                apply_water_mask=True,             # silences ocean phase noise (Mombasa)
                phase_filter_parameter=0.6,        # SDK default; right for mixed urban/vegetation
            )
            new_batch += b
            break  # Success! Break out of the retry loop
        except Exception as e:
            if attempt == 2:  # Hard crash if all 3 attempts fail
                print(f"❌ Failed to submit {name} after 3 attempts.")
                raise e
            print(f"⚠️ Server hiccup ({e}). Pausing 5 seconds before retrying...")
            time.sleep(5)

    full = Batch()
    for j in resumed:
        full += j
    full += new_batch
    return full


def wait_and_download(batch, dest: Path, hyp3=None):
    """Step 3. Block on completion, then download finished products in parallel.

    `hyp3.watch()` uses adaptive polling (60-300s) and blocks until every job in
    the batch reaches a terminal state. Cheaper and faster than a hand-rolled
    loop. We parallelize the downloads with a small thread pool — products are
    50-200 MB each, and a single sequential download would take all day.
    """
    if hyp3 is None:
        hyp3 = _connect_hyp3()

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  watching {len(batch)} jobs (this can take hours)…")
    batch = hyp3.watch(batch)
    succeeded = [j for j in batch if j.succeeded()]
    failed = [j for j in batch if j.failed()]
    print(f"  {len(succeeded)} succeeded, {len(failed)} failed")
    for j in failed:
        # Surface the failure reason so we can decide whether to resubmit.
        print(f"    FAILED {j.name}: {getattr(j, 'status_code', '?')}")

    # Parallel download. HyP3 product URLs are pre-signed S3, so the bottleneck
    # is bandwidth, not the server — 4-8 concurrent connections saturates a
    # typical home line without provoking rate limiting.
    to_fetch = [j for j in succeeded if not _already_extracted(j, dest)]
    print(f"  downloading {len(to_fetch)} products (skipping {len(succeeded) - len(to_fetch)} already on disk)…")

    def _fetch(job):
        files = job.download_files(dest, create=True)
        for f in files:
            _unzip(Path(f), dest / job.name)
            Path(f).unlink(missing_ok=True)
        return job.name

    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed(pool.submit(_fetch, j) for j in to_fetch):
            name = fut.result()
            print(f"    ✓ {name}")

    return [dest / j.name for j in succeeded]


def _already_extracted(job, dest: Path) -> bool:
    """Skip-download check: the per-job product dir exists and has the GAMMA
    outputs MintPy will need."""
    d = dest / job.name
    if not d.is_dir():
        return False
    # GAMMA InSAR output contains <product>_unw_phase.tif and <product>_corr.tif
    return any(d.glob("*_unw_phase.tif")) and any(d.glob("*_corr.tif"))


def _unzip(zip_path: Path, into: Path) -> None:
    """Extract a HyP3 product zip, flattening the single top-level directory."""
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        # HyP3 zips wrap everything in <product_id>/ — strip that prefix.
        top = os.path.commonpath(members).rstrip("/") + "/" if members else ""
        for m in members:
            if m.endswith("/"):
                continue
            rel = m[len(top):] if top and m.startswith(top) else m
            out = into / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(out, "wb") as dst:
                # 1 MiB chunks — small enough to keep memory flat across the
                # thread pool, large enough to keep syscall overhead negligible.
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)


def run_mintpy(stack_dir: Path) -> Path:
    """Step 4. Run MintPy headlessly. Writes velocity.h5 and timeseries.h5."""
    raise NotImplementedError(
        "Write a smallbaselineApp config (smallbaselineApp.cfg) pointing at:\n"
        "    mintpy.load.processor      = hyp3\n"
        "    mintpy.load.unwFile        = <stack>/*/*unw_phase.tif\n"
        "    mintpy.load.corFile        = <stack>/*/*corr.tif\n"
        "    mintpy.load.demFile        = <stack>/*/*dem.tif\n"
        "Then: subprocess.run(['smallbaselineApp.py', 'smallbaselineApp.cfg'], cwd=stack_dir)\n"
        "Output: velocity.h5, timeseries.h5"
    )


def join_to_footprints(mintpy_out: Path, footprints_geojson: Path) -> Path:
    """Step 5. Spatial-join MintPy points to OSM building footprints. Emit GeoParquet.

    The deliverables here match the seeder's schema exactly:
      - buildings.parquet            (one row per footprint)
      - subsidence.parquet           (building_id, observation_date, displacement_mm, ...)
    """
    raise NotImplementedError(
        "Approach:\n"
        "  1. Use h5py to read mintpy_out/timeseries.h5 → array of (n_dates, n_y, n_x)\n"
        "  2. Read mintpy_out/geo/geo_velocity.h5 lat/lon arrays for georeferencing\n"
        "  3. For each footprint polygon, take the mean displacement over intersecting pixels\n"
        "     weighted by coherence (drop pixels with coherence < 0.3)\n"
        "  4. Write GeoParquet via geopandas.to_parquet(..., schema_version='1.0.0')"
    )


def load_into_duckdb(parquet_dir: Path, db_path: Path):
    """Step 6. Replace the demo DuckDB with real data using the same schema."""
    import duckdb
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute((parquet_dir.parent / "init_db.sql").read_text())  # init_db.sql is reused
    con.execute(f"INSERT INTO buildings SELECT * FROM read_parquet('{parquet_dir}/buildings.parquet');")
    con.execute(f"INSERT INTO subsidence_time_series SELECT * FROM read_parquet('{parquet_dir}/subsidence.parquet');")
    con.close()


def plan(aoi: AOI) -> dict:
    """Stage-1 entry: discover scenes and report what HyP3 will be asked to do.

    Does NOT submit jobs. Use this to sanity-check track coverage and pair
    counts before committing compute. The dict is JSON-serializable so the
    caller can stash it as a manifest before Stage 2.
    """
    scenes = search_scenes(aoi)
    pairs = make_pairs(scenes)

    per_track: dict[str, dict] = {}
    for s in scenes:
        key = f"{s.flight_direction}/path={s.path}"
        per_track.setdefault(key, {"scenes": 0, "first": None, "last": None})
        per_track[key]["scenes"] += 1
        ts = s.start_time.date().isoformat()
        if per_track[key]["first"] is None or ts < per_track[key]["first"]:
            per_track[key]["first"] = ts
        if per_track[key]["last"] is None or ts > per_track[key]["last"]:
            per_track[key]["last"] = ts
    for key in per_track:
        per_track[key]["pairs"] = sum(
            1 for p in pairs
            if f"{p.flight_direction}/path={p.path}" == key
        )

    return {
        "aoi": aoi.code,
        "window": [START.isoformat(), END.isoformat()],
        "n_scenes": len(scenes),
        "n_pairs": len(pairs),
        "per_track": per_track,
    }


def _resolve_pairs(aoi: AOI, all_tracks: bool) -> list[ScenePair]:
    scenes = search_scenes(aoi)
    pairs = make_pairs(scenes)
    if not all_tracks:
        tracks = DEFAULT_TRACKS.get(aoi.code, [])
        if not tracks:
            raise RuntimeError(f"no default track configured for {aoi.code}; pass --all-tracks")
        pairs = filter_to_tracks(pairs, tracks)
    return pairs


def _cmd_plan(args: argparse.Namespace) -> None:
    """Stage 1: print scene plan for the requested AOIs without submitting."""
    aois = [by_code(c) for c in args.aoi] if args.aoi else REGISTRY
    for aoi in aois:
        print(f"\n=== {aoi.code} ({aoi.name}) ===")
        p = plan(aoi)
        print(json.dumps(p, indent=2))


def _cmd_submit(args: argparse.Namespace) -> None:
    """Stage 2: submit pairs to HyP3 (idempotent), optionally watch + download."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    aois = [by_code(c) for c in args.aoi] if args.aoi else REGISTRY
    hyp3 = _connect_hyp3()

    # Estimate cost before we submit so the user can abort.
    plans: list[tuple[AOI, list[ScenePair]]] = []
    total = 0
    for aoi in aois:
        pairs = _resolve_pairs(aoi, all_tracks=args.all_tracks)
        plans.append((aoi, pairs))
        total += len(pairs)
        print(f"  {aoi.code}: {len(pairs)} pairs")
    info = hyp3.my_info()
    remaining = info.get("remaining_credits")
    print(f"\n  total pairs to submit: {total}; HyP3 credits remaining: {remaining}")
    if remaining is not None and total > remaining:
        sys.exit(f"insufficient credits: need {total}, have {remaining}")
    if not args.yes:
        resp = input("  proceed? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit("aborted")

    batches = []
    for aoi, pairs in plans:
        print(f"\n--- submitting {aoi.code} ---")
        b = submit_hyp3(aoi.code, pairs, hyp3=hyp3)
        batches.append((aoi, b))

    if args.watch:
        for aoi, b in batches:
            print(f"\n--- watching + downloading {aoi.code} ---")
            wait_and_download(b, WORK_DIR / aoi.code, hyp3=hyp3)


def _cmd_status(args: argparse.Namespace) -> None:
    """Quick status report: count of jobs per AOI in each terminal state."""
    hyp3 = _connect_hyp3()
    jobs = hyp3.find_jobs(job_type="INSAR_GAMMA")
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for j in jobs:
        if not j.name:
            continue
        aoi_letter = j.name[:1]
        aoi = next((c for c, l in _AOI_LETTER.items() if l == aoi_letter), aoi_letter)
        status = "SUCCEEDED" if j.succeeded() else "FAILED" if j.failed() else "RUNNING/PENDING"
        buckets[aoi][status] += 1
    for aoi, counts in sorted(buckets.items()):
        print(f"  {aoi}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main():
    """CLI entry. Run from backend/:

        python -m scripts.hyp3_pipeline plan
        python -m scripts.hyp3_pipeline submit --watch
        python -m scripts.hyp3_pipeline status
    """
    p = argparse.ArgumentParser(description="InSAR pipeline driver")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="print scene plan without submitting")
    p_plan.add_argument("--aoi", action="append", help="AOI code (repeatable); default = all")
    p_plan.set_defaults(func=_cmd_plan)

    p_sub = sub.add_parser("submit", help="submit pairs to HyP3 (idempotent)")
    p_sub.add_argument("--aoi", action="append", help="AOI code (repeatable); default = all")
    p_sub.add_argument("--all-tracks", action="store_true",
                       help="include every track. Default: single best ASC track per AOI")
    p_sub.add_argument("--watch", action="store_true",
                       help="block until jobs finish and download products")
    p_sub.add_argument("-y", "--yes", action="store_true", help="skip confirmation prompt")
    p_sub.set_defaults(func=_cmd_submit)

    p_stat = sub.add_parser("status", help="report HyP3 job state per AOI")
    p_stat.set_defaults(func=_cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
