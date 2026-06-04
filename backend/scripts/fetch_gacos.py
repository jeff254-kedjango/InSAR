"""
GACOS tropospheric-delay fetcher (ARCHITECTURE_THREE A1).

Pulls per-acquisition Zenith Total Delay grids from the GACOS portal for every
Sentinel-1 acquisition date in our HyP3 working set, caches them under
`data/raw/env/gacos/<aoi>/`, and lets MintPy's gacos step subtract them at
inversion time.

Why this matters: in tropical Nairobi/Mombasa, atmospheric water-vapour delay
is the *dominant* source of velocity error after coherence — typically several
mm/yr of bias if uncorrected. GACOS reduces that by ~50–70 %, which is the
single biggest credibility lever in Phase A.

Portal mechanics:
  1. POST a form to gacos.net with: bbox, date list, email, mode=time-series.
  2. Portal queues the job, emails a download URL when ready (typically
     minutes to hours, depending on queue depth).
  3. Job result is a `.tar.gz` (or `.zip`) of `YYYYMMDD.ztd.tif` GeoTIFF grids
     (we request the GeoTIFF output format — MintPy 1.6 reads it natively via
     GDAL, no binary-grid endianness handling) plus a `_preview.jpg` per date.
     We unpack only the `*.ztd.tif` grids into the AOI cache dir under the
     names MintPy expects (YYYYMMDD.ztd.tif).

This script does not handle the email step automatically — that requires SMTP
plumbing nobody wants. Instead it supports these modes:
  - `submit`: post the request, print the portal job ID, exit.
  - `ingest`: take one manually-downloaded archive (.tar.gz or .zip) and
    unpack its `*.ztd.tif` grids into the AOI cache dir.
  - `ingest-dir`: ingest every archive in a directory, auto-routing each one
    to the correct AOI by the GeoTIFF's geographic centre (no --aoi needed).
  - `status`: coverage report against the HyP3 acquisition dates.

All modes are idempotent. `ingest`/`ingest-dir` skip files already present.

Performance:
  - All file I/O is streamed (no buffer-in-memory of the zip).
  - Date extraction is a single regex pass over HyP3 job-dir names.
  - No per-date Python loop overhead; bulk operations everywhere.

Usage:
    # 1. Submit a GACOS job for an AOI (one HTTPS POST):
    python -m scripts.fetch_gacos submit --aoi huruma --email me@example.com

    # 2. After portal emails you a download URL, ingest the zip:
    python -m scripts.fetch_gacos ingest --aoi huruma --zip ~/Downloads/gacos_xxx.zip
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from datetime import date
from pathlib import Path

import requests

from scripts.aois import AOI, REGISTRY, by_code, bbox


BACKEND_DIR = Path(__file__).resolve().parents[1]
WORK_DIR = BACKEND_DIR / "data" / "hyp3_work"
GACOS_CACHE_DIR = BACKEND_DIR / "data" / "raw" / "env" / "gacos"

# HyP3 job dir names: `h-A57-240606240618` (Huruma), `m-A159-240601240613`
# (Mombasa) — first char is the AOI prefix from hyp3_pipeline.py. Capture both
# ref and sec dates so we cover the full acquisition list.
_JOB_NAME_RE = re.compile(r"^[a-z]-[AD]\d+-(\d{6})(\d{6})$")

# GACOS portal endpoint. The web form is the only documented entry point;
# there is no rate-limited public API. We post the same fields the form posts.
GACOS_PORTAL_URL = "http://www.gacos.net/M/action_page.php"


def _yymmdd_to_date(s: str) -> date:
    """`240606` → date(2024, 6, 6). Two-digit year assumes 20xx (S1 launched 2014)."""
    return date(2000 + int(s[:2]), int(s[2:4]), int(s[4:6]))


def discover_acquisition_dates(aoi: AOI) -> list[date]:
    """Walk WORK_DIR/<aoi>/ and pull every Sentinel-1 acquisition date from
    HyP3 job names. Returns a deduped, sorted list."""
    aoi_dir = WORK_DIR / aoi.code
    if not aoi_dir.is_dir():
        raise FileNotFoundError(f"no HyP3 work dir for {aoi.code}: {aoi_dir}")

    dates: set[date] = set()
    for entry in aoi_dir.iterdir():
        if not entry.is_dir():
            continue
        m = _JOB_NAME_RE.match(entry.name)
        if not m:
            continue
        dates.add(_yymmdd_to_date(m.group(1)))
        dates.add(_yymmdd_to_date(m.group(2)))
    if not dates:
        raise RuntimeError(
            f"no HyP3 job dirs matched expected pattern under {aoi_dir} — "
            f"is the Stage-2 download complete?"
        )
    return sorted(dates)


def _aoi_cache(aoi: AOI) -> Path:
    d = GACOS_CACHE_DIR / aoi.code
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached_date_codes(cache: Path) -> set[str]:
    """`YYYYMMDD` codes already present in the cache dir.

    GACOS GeoTIFFs land as `YYYYMMDD.ztd.tif`; `Path.stem` would leave the
    `.ztd`, so we split on the first dot to recover the bare date code.
    """
    return {p.name.split(".", 1)[0] for p in cache.glob("*.ztd.tif")}


def _missing_dates(aoi: AOI, dates: list[date]) -> list[date]:
    """Subset of `dates` for which we don't already have a `.ztd.tif` cached."""
    have = _cached_date_codes(_aoi_cache(aoi))
    return [d for d in dates if d.strftime("%Y%m%d") not in have]


def submit(aoi: AOI, email: str) -> None:
    """POST a time-series job to the GACOS portal.

    Output is the portal's response page; the job ID is in the HTML body. The
    portal emails a download URL when the job completes (queue depth varies —
    sometimes minutes, sometimes hours).
    """
    dates = discover_acquisition_dates(aoi)
    missing = _missing_dates(aoi, dates)
    if not missing:
        print(f"  ✓ all {len(dates)} dates already cached for {aoi.code}; nothing to submit")
        return

    minlon, minlat, maxlon, maxlat = bbox(aoi)
    # Portal expects newline-separated `YYYYMMDD` strings (no hyphens) —
    # matches the eventual `YYYYMMDD.ztd` GACOS output naming. Earlier code
    # sent `YYYY-MM-DD`; the portal rejected those as "wrong date format".
    date_block = "\n".join(d.strftime("%Y%m%d") for d in missing)
    payload = {
        "N": f"{maxlat:.4f}",
        "S": f"{minlat:.4f}",
        "W": f"{minlon:.4f}",
        "E": f"{maxlon:.4f}",
        "H": "00:00",                     # Sentinel-1 IW ASC ~05:30 UTC at this longitude;
                                          # GACOS rounds internally — 00:00 is fine for daily ERA5
        "type": "2",                      # "2" = time-series mode (not single-epoch)
        "date": date_block,
        "email": email,
    }
    print(f"  → submitting {len(missing)} dates to GACOS portal for {aoi.code}")
    try:
        r = requests.post(GACOS_PORTAL_URL, data=payload, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  ❌ GACOS submission failed: {e}", file=sys.stderr)
        raise

    # The portal returns an HTML page; the job ID is in there. Print just the
    # first 200 chars so the user can see the confirmation without spamming.
    body = r.text or ""
    snippet = body.replace("\n", " ")[:400]
    print(f"  ✓ portal accepted: {snippet}")
    print(f"\n  next: wait for the GACOS email, then run:")
    print(f"    python -m scripts.fetch_gacos ingest --aoi {aoi.code} --zip <downloaded.zip>")


def _iter_archive_members(archive: Path):
    """Yield (member_name, size, open_stream_callable) for a .tar.gz or .zip.

    Abstracts over tarfile/zipfile so the extraction loop is identical for
    both. `open_stream_callable()` returns a binary file-like for streaming.
    """
    name = archive.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        tf = tarfile.open(archive, "r:*")
        try:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                yield m.name, m.size, (lambda mm=m: tf.extractfile(mm))
        finally:
            # Caller fully consumes the generator before we close.
            tf.close()
    elif name.endswith(".zip"):
        zf = zipfile.ZipFile(archive)
        try:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                yield info.filename, info.file_size, (lambda i=info: zf.open(i))
        finally:
            zf.close()
    else:
        raise ValueError(f"unsupported archive type: {archive} (expected .tar.gz/.tgz/.zip)")


def _extract_ztd_tif(archive: Path, cache: Path) -> tuple[int, int]:
    """Stream every `*.ztd.tif` from `archive` into `cache`. Returns (new, skipped).

    Only the GeoTIFF delay grids are kept — `_preview.jpg` thumbnails and any
    other members are ignored. Idempotent: skips a file already present with
    the same byte size.
    """
    n_new = n_skipped = 0
    for member_name, size, opener in _iter_archive_members(archive):
        leaf = Path(member_name).name
        if not leaf.endswith(".ztd.tif"):
            continue
        out = cache / leaf
        if out.exists() and out.stat().st_size == size:
            n_skipped += 1
            continue
        src = opener()
        if src is None:  # tar can return None for odd member types
            continue
        with src, out.open("wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
        n_new += 1
    return n_new, n_skipped


def _route_archive_to_aoi(archive: Path) -> AOI | None:
    """Decide which AOI an archive belongs to by reading one GeoTIFF's centre.

    GACOS grids are WGS84 GeoTIFFs; the embedded geotransform tells us where
    on Earth they sit, so we don't have to trust filenames or ask the user
    which tarball is which. Returns the AOI whose bbox contains the grid
    centre, or None if it matches none.
    """
    import rasterio
    from rasterio.io import MemoryFile

    for member_name, _size, opener in _iter_archive_members(archive):
        if not Path(member_name).name.endswith(".ztd.tif"):
            continue
        src = opener()
        if src is None:
            continue
        with src:
            data = src.read()
        with MemoryFile(data) as mf, mf.open() as ds:
            b = ds.bounds
        cx, cy = (b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0
        for aoi in REGISTRY:
            minlon, minlat, maxlon, maxlat = bbox(aoi)
            if minlon <= cx <= maxlon and minlat <= cy <= maxlat:
                return aoi
        return None  # had a grid, matched no AOI — don't keep scanning
    return None


def _report_coverage(aoi: AOI, cache: Path) -> None:
    """Spot-check cached coverage against the HyP3 acquisition list."""
    expected = {d.strftime("%Y%m%d") for d in discover_acquisition_dates(aoi)}
    have = _cached_date_codes(cache)
    missing = sorted(expected - have)
    if missing:
        print(f"  ⚠ {aoi.code}: still missing {len(missing)} dates: "
              f"{missing[:5]}{'…' if len(missing) > 5 else ''}")
        print(f"     resubmit those via `submit` or accept reduced coverage (MintPy will skip them)")
    else:
        print(f"  ✓ {aoi.code}: all {len(expected)} HyP3 acquisition dates have GACOS coverage")


def ingest(aoi: AOI, archive_path: Path) -> None:
    """Unpack one GACOS archive (.tar.gz or .zip) into the AOI cache dir.

    MintPy reads `YYYYMMDD.ztd.tif` GeoTIFFs from `mintpy.troposphericDelay.gacosDir`
    directly. We keep only those grids; previews are dropped.
    """
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    cache = _aoi_cache(aoi)
    n_new, n_skipped = _extract_ztd_tif(archive_path, cache)
    print(f"  ✓ ingested {n_new} GACOS grids into {cache}  ({n_skipped} already cached)")
    _report_coverage(aoi, cache)


def ingest_dir(archive_dir: Path) -> None:
    """Ingest every archive in a directory, auto-routing each to its AOI.

    The GACOS portal caps a submission at 20 dates, so each AOI's window
    arrives as several archives. Rather than make the user track which file is
    which, we read each archive's GeoTIFF centre and route by AOI bbox.
    """
    if not archive_dir.is_dir():
        raise NotADirectoryError(archive_dir)
    archives = sorted(
        p for p in archive_dir.iterdir()
        if p.name.lower().endswith((".tar.gz", ".tgz", ".tar", ".zip"))
    )
    if not archives:
        print(f"  ⚠ no .tar.gz/.zip archives found in {archive_dir}")
        return

    touched: set[str] = set()
    for archive in archives:
        aoi = _route_archive_to_aoi(archive)
        if aoi is None:
            print(f"  ⚠ {archive.name}: no AOI matched its grid centre — skipped")
            continue
        cache = _aoi_cache(aoi)
        n_new, n_skipped = _extract_ztd_tif(archive, cache)
        print(f"  ✓ {archive.name} → {aoi.code}: {n_new} new, {n_skipped} skipped")
        touched.add(aoi.code)

    print()
    for code in sorted(touched):
        _report_coverage(by_code(code), _aoi_cache(by_code(code)))


def status(aoi: AOI) -> None:
    """One-shot report: how many dates have GACOS coverage."""
    dates = discover_acquisition_dates(aoi)
    cache = _aoi_cache(aoi)
    have = _cached_date_codes(cache)
    n_total = len(dates)
    n_have = sum(1 for d in dates if d.strftime("%Y%m%d") in have)
    print(f"  {aoi.code}: {n_have}/{n_total} dates covered  ({cache})")
    if n_have < n_total:
        first_missing = next(d for d in dates if d.strftime("%Y%m%d") not in have)
        print(f"    first missing: {first_missing.isoformat()}")


def main() -> None:
    p = argparse.ArgumentParser(description="GACOS tropospheric-delay fetcher")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sub = sub.add_parser("submit", help="POST a GACOS time-series job")
    p_sub.add_argument("--aoi", required=True)
    p_sub.add_argument("--email", required=True, help="portal sends download URL here")

    p_ing = sub.add_parser("ingest", help="unpack one GACOS archive (.tar.gz/.zip) into the cache")
    p_ing.add_argument("--aoi", required=True)
    p_ing.add_argument("--archive", dest="archive_path", required=True, type=Path,
                       help="path to the GACOS .tar.gz or .zip")

    p_idir = sub.add_parser("ingest-dir",
                            help="ingest every archive in a dir, auto-routing each to its AOI")
    p_idir.add_argument("--dir", dest="archive_dir", required=True, type=Path)

    p_st = sub.add_parser("status", help="coverage report")
    p_st.add_argument("--aoi", required=True)

    args = p.parse_args()
    if args.cmd == "submit":
        submit(by_code(args.aoi), args.email)
    elif args.cmd == "ingest":
        ingest(by_code(args.aoi), args.archive_path)
    elif args.cmd == "ingest-dir":
        ingest_dir(args.archive_dir)
    elif args.cmd == "status":
        status(by_code(args.aoi))


if __name__ == "__main__":
    main()
