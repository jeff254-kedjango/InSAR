"""
Generate plausible synthetic datasets for every AOI in the registry and emit them
as Hive-partitioned GeoParquet:

    backend/data/parquet/
        aoi_registry.parquet
        buildings/aoi=<code>/data.parquet
        subsidence/aoi=<code>/data.parquet
        env_index/aoi=<code>/data.parquet
        demo.duckdb                       (built last from the Parquet files)

The data shape (columns, dtypes) matches what the real HyP3+MintPy pipeline will
emit per AOI, so swapping in real data downstream changes no app code.

Per-AOI physics live in `phenomena.py`. New AOIs are added by registering them
in `aois.py` and (if needed) writing a new phenomenon function.

Run:
    cd backend
    python -m scripts.seed_synthetic
"""

from __future__ import annotations

import math
import random
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import aois
from .phenomena import generate_aoi_dataset
from .provenance import set_provenance

N_MONTHS = 24
START_DATE = date(2024, 6, 1)

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = ROOT / "data" / "parquet"
DB_PATH = ROOT / "data" / "demo.duckdb"
SQL_PATH = Path(__file__).resolve().parent / "init_db.sql"


def monthly_dates() -> list[date]:
    return [
        date.fromordinal(START_DATE.toordinal() + int(30.44 * i))
        for i in range(N_MONTHS)
    ]


def write_aoi_registry() -> None:
    rows = [
        {
            "aoi_code":          a.code,
            "name":              a.name,
            "center_lon":        a.center_lon,
            "center_lat":        a.center_lat,
            "side_m":            a.side_m,
            "phenomenon":        a.phenomenon,
            "footprint_source":  a.footprint_source,
            "narrative":         a.narrative,
            "bbox_minlon":       aois.bbox(a)[0],
            "bbox_minlat":       aois.bbox(a)[1],
            "bbox_maxlon":       aois.bbox(a)[2],
            "bbox_maxlat":       aois.bbox(a)[3],
            # ARCHITECTURE_THREE B4 — InSAR reference point. Rendered as the ⚓
            # pin on the map. Every velocity in the bundle is measured
            # relative to this lat/lon.
            "reference_lon":     a.reference_lon,
            "reference_lat":     a.reference_lat,
            "reference_note":    a.reference_note,
        }
        for a in aois.REGISTRY
    ]
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), PARQUET_DIR / "aoi_registry.parquet")


def write_partition(table: str, aoi_code: str, rows: pa.Table) -> None:
    out_dir = PARQUET_DIR / table / f"aoi={aoi_code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(rows, out_dir / "data.parquet", compression="zstd")


def write_synthetic_coh_series(
    aoi_code: str,
    building_ids: np.ndarray,
    subsidence_tbl: pa.Table,
) -> None:
    """ARCHITECTURE_THREE B2 — emit a coh_series partition from the synthetic
    subsidence table. One row per building, packed Float32 binary of length
    `n_months × 4`. We reuse the per-month synthetic coherence already present
    in the subsidence table, so the sparkline matches the time-series.

    Vectorised: pull the coherence column once, reshape to (n_buildings,
    n_months), build all blobs in one tobytes() call.
    """
    from scripts.postprocess import COH_SERIES_SCHEMA

    n = building_ids.size
    coh_col = np.asarray(subsidence_tbl.column("coherence").to_numpy(), dtype=np.float32)
    if coh_col.size % n != 0:
        raise RuntimeError(f"coh column not divisible by n_buildings: {coh_col.size} % {n}")
    n_epochs = coh_col.size // n
    coh_matrix = np.ascontiguousarray(coh_col.reshape(n, n_epochs), dtype=np.float32)
    raw = coh_matrix.tobytes()
    row_nbytes = n_epochs * 4
    blobs = [raw[i * row_nbytes:(i + 1) * row_nbytes] for i in range(n)]
    tbl = pa.table(
        {
            "building_id": pa.array(building_ids.tolist(), type=pa.int64()),
            "aoi_code":    pa.array([aoi_code] * n,        type=pa.string()),
            "coh_series":  pa.array(blobs,                 type=pa.binary()),
        },
        schema=COH_SERIES_SCHEMA,
    ).replace_schema_metadata({b"n_epochs": str(n_epochs).encode()})
    write_partition("coh_series", aoi_code, tbl)


def build_duckdb() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    con.execute("INSTALL spatial; LOAD spatial;")
    sql = SQL_PATH.read_text().replace("${PARQUET_ROOT}", str(PARQUET_DIR))
    con.execute(sql)
    # Smoke check: counts per AOI
    for a in aois.REGISTRY:
        nb = con.execute("SELECT COUNT(*) FROM buildings WHERE aoi_code = ?", [a.code]).fetchone()[0]
        nt = con.execute("SELECT COUNT(*) FROM subsidence_time_series WHERE aoi_code = ?", [a.code]).fetchone()[0]
        print(f"  {a.code}: {nb} buildings, {nt} time-series rows")
    con.close()


def main() -> None:
    print("Seeding synthetic data for all AOIs...")
    write_aoi_registry()
    dates = monthly_dates()

    for aoi in aois.REGISTRY:
        print(f"\nAOI: {aoi.code} ({aoi.phenomenon})")
        rng = random.Random(hash(aoi.code) & 0xFFFF_FFFF)
        np_rng = np.random.default_rng(hash(aoi.code) & 0xFFFF_FFFF)

        buildings, subsidence, env_index = generate_aoi_dataset(
            aoi=aoi,
            dates=dates,
            rng=rng,
            np_rng=np_rng,
        )

        print(f"  {buildings.num_rows} buildings, {subsidence.num_rows} ts rows, {env_index.num_rows} env rows")
        write_partition("buildings", aoi.code, buildings)
        write_partition("subsidence", aoi.code, subsidence)
        write_partition("env_index", aoi.code, env_index)
        # ARCHITECTURE_THREE B2 — packed Float32 coherence sparkline per building.
        bids_arr = np.asarray(buildings.column("building_id").to_numpy(), dtype=np.int64)
        write_synthetic_coh_series(aoi.code, bids_arr, subsidence)
        # Record provenance. When real footprints + real static terrain were
        # used (the parquet exists), this AOI is 'partial' — real geometry/soil/
        # proximity, synthetic velocity until the MintPy join. Otherwise it's a
        # fully-synthetic fallback. A later real join (scripts/join_insar.py)
        # flips just this AOI to 'insar'.
        has_real_footprints = (ROOT / "data" / "footprints" / f"{aoi.code}.parquet").exists()
        set_provenance(aoi.code, "partial" if has_real_footprints else "synthetic")

    print("\nBuilding DuckDB demo.duckdb from Parquet...")
    build_duckdb()
    print(f"\nDone. DB at: {DB_PATH}")


if __name__ == "__main__":
    main()
