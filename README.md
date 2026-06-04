# infra-proptech

Local-only laptop demo: structural deformation monitor for a 2km × 2km Nairobi neighborhood, built on Sentinel-1 InSAR. Designed to run with **zero internet** during a boardroom pitch.

## Honest framing

Per-building InSAR velocities in dense, informal Nairobi neighborhoods are **noisy** (decorrelation on corrugated-iron roofs, sub-pixel buildings, construction churn). Treat the numbers as a **block-level deformation indicator**, fused with riparian distance and soil class into an explainable composite score. This is **not** a predictive collapse model — that needs construction-quality data the pipeline can't see.

## Architecture (decided)

| Layer | Choice | Why |
|---|---|---|
| Basemap | MapLibre GL JS + PMTiles | Truly offline, no Mapbox token, single-file tiles |
| 3D viz | deck.gl `PolygonLayer` over MapLibre | Data-driven extrusion, animates per-feature attrs |
| Backend | Single FastAPI process | Laptop demo — microservices add nothing |
| Store | DuckDB + `spatial` ext + GeoParquet | Zero-install, fast at this scale, ships as files |
| Pipeline | `asf_search` → HyP3 → MintPy, **build-time only** | Demo never calls the network |

PostGIS is the right answer at city scale; not for this MVP.

## Layout

```
backend/
  app/main.py                 FastAPI service (read-only DuckDB)
  scripts/init_db.sql         DuckDB schema + RTree indexes + v_building_latest view
  scripts/seed_synthetic.py   Plausible 24-month synthetic dataset (run this first)
  scripts/hyp3_pipeline.py    Skeleton for the real InSAR pipeline (build-time)
  data/                       demo.duckdb + parquet/ (git-ignored or LFS-tracked)
frontend/
  src/components/RiskMap.tsx  MapLibre + deck.gl + threat sidebar + time slider
  public/tiles/               PMTiles basemap goes here
docs/
  risk_model.md               Composite-score weights and caveats
```

## Quickstart

```bash
# 1. Seed synthetic data (≈10s)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m backend.scripts.seed_synthetic

# 2. Serve API
uvicorn backend.app.main:app --port 8000

# 3. Frontend
cd ../frontend
npm install
npm run dev   # http://localhost:5173
```

The basemap is optional — the UI boots on a flat dark canvas without it. To install a PMTiles extract covering both AOIs (Huruma + Mombasa, ~40-60 MB):

```bash
cd frontend
npm run fetch:pmtiles
```

The script installs the `pmtiles` CLI locally (`frontend/scripts/.bin/`), extracts the Kenya bbox from the Protomaps daily build, and writes `frontend/public/tiles/nairobi.pmtiles`. Re-runs are no-ops; pass `--force` to overwrite. The MapLibre style probes for the file at boot and lights up road/building/water layers automatically when present.

## Swapping in the real pipeline

When `backend/scripts/hyp3_pipeline.py` is fleshed out and run, it writes Parquet files with the same schema as the seeder. Replace the seeder's outputs and `init_db.sql` re-runs identically. No app code changes.

## Frontend dependencies

```
react react-dom
maplibre-gl pmtiles
deck.gl @deck.gl/layers @deck.gl/mapbox
tailwindcss
```

## What's left

- [ ] Implement the real `hyp3_pipeline.py` steps (each is `NotImplementedError` with the call shape inlined)
- [ ] PMTiles basemap extract checked into LFS or fetched at install time
- [ ] Calibrate composite-risk weights against any historical incident reports
- [ ] Add an "uncertainty visible" mode that downweights / hashes low-coherence buildings on the map
