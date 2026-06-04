"""
AOI registry. One module-level constant per area of interest, holding everything
the seeder, pipeline, and UI need to know about it. New AOIs go here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Phenomenon = Literal["informal_settlement_subsidence", "coastal_subsidence"]
FootprintSource = Literal["open_buildings", "osm", "synthetic"]


@dataclass(frozen=True)
class AOI:
    code: str
    name: str
    center_lon: float
    center_lat: float
    side_m: float                       # AOI is a square of this side around the centroid
    phenomenon: Phenomenon
    footprint_source: FootprintSource   # which real source we'll use when we leave synthetic
    n_synthetic_buildings: int          # for the seeder
    narrative: str                      # short copy shown in the UI sidebar
    # ARCHITECTURE_THREE A3 — explicit InSAR reference point.
    # Chosen *outside* the AOI on stable, high-coherence terrain so all
    # velocities are anchored to a documented zero. Surfaced in the UI as the
    # ⚓ pin so viewers can see what their measurements are relative to.
    reference_lon: float
    reference_lat: float
    reference_note: str                 # ≤120 chars, shown as the pin tooltip
    # ARCHITECTURE_THREE A3 — processing geometry decoupled from display geometry.
    # `side_m` is the 2 km tile the UI/bundle shows; the MintPy load subset can be
    # wider so it (a) is identical across all interferograms and (b) contains the
    # reference anchor, which sits outside the display tile. None → processing box
    # == display box. Defaulted, so it must stay last among the fields.
    processing_side_m: float | None = None


HURUMA = AOI(
    code="huruma",
    name="Huruma, Nairobi",
    center_lon=36.8740,
    center_lat=-1.2510,
    side_m=2000.0,
    # Karura anchor is ~4396 m NW of centre — outside the 2 km display tile. A
    # 10 km processing box (half-side 5000 m) contains it with ~600 m margin.
    processing_side_m=10000.0,
    phenomenon="informal_settlement_subsidence",
    footprint_source="open_buildings",
    n_synthetic_buildings=1500,
    narrative=(
        "Dense informal settlement on mixed black-cotton and alluvial soils, "
        "with a tributary running diagonally through the block. Footprints are "
        "ML-derived (Google Open Buildings) because OSM coverage here is sparse. "
        "InSAR signal is noisy on corrugated-iron roofs — coherence is surfaced "
        "in the UI so users can see where to trust the velocity."
    ),
    # Karura Forest granite edge — stable Precambrian basement, high temporal
    # coherence in S1 stacks, no construction, ~4 km NW of Huruma centroid.
    reference_lon=36.8345,
    reference_lat=-1.2391,
    reference_note=(
        "Karura Forest bedrock outcrop — stable Precambrian basement, no "
        "construction. Anchors all Huruma velocities to a documented zero."
    ),
)

MOMBASA = AOI(
    code="mombasa",
    name="Mombasa Old Town / Kilindini",
    center_lon=39.6680,
    center_lat=-4.0610,
    side_m=2000.0,
    # Changamwe anchor is ~3840 m W of centre — outside the 2 km display tile. A
    # 9 km processing box (half-side 4500 m) contains it with ~660 m margin.
    processing_side_m=9000.0,
    phenomenon="coastal_subsidence",
    footprint_source="open_buildings",
    n_synthetic_buildings=1100,
    narrative=(
        "Coastal urban tile spanning Old Town and reclaimed land near Kilindini. "
        "Footprints are ML-derived (Google Open Buildings) for dense, uniform "
        "coverage. Concrete and bare surfaces yield higher InSAR coherence than "
        "Huruma. Watch the reclaimed-land cohort on engineered fill (real "
        "SoilGrids reclaim_fill ground): subsidence there is real, slow, and "
        "well-measured by Sentinel-1."
    ),
    # Changamwe Hill coral-platform exposure — far from coastline + reclaim,
    # high coherence on both ASC/DESC tracks, ~3 km W of Mombasa centroid.
    reference_lon=39.6395,
    reference_lat=-4.0265,
    reference_note=(
        "Changamwe Hill coral platform — inland, stable, far from reclaim fill. "
        "Anchors all Mombasa velocities to a documented zero."
    ),
)


REGISTRY: list[AOI] = [HURUMA, MOMBASA]


def by_code(code: str) -> AOI:
    for a in REGISTRY:
        if a.code == code:
            return a
    raise KeyError(f"unknown AOI: {code}")


def _square_bbox(center_lon: float, center_lat: float, side_m: float) -> tuple[float, float, float, float]:
    """(minlon, minlat, maxlon, maxlat) for a square of `side_m` — equirect approx."""
    import math
    half = side_m / 2.0
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * math.cos(math.radians(center_lat)))
    return (
        center_lon - dlon, center_lat - dlat,
        center_lon + dlon, center_lat + dlat,
    )


def bbox(aoi: AOI) -> tuple[float, float, float, float]:
    """Display bbox — the 2 km tile the UI and bundle show (`side_m`)."""
    return _square_bbox(aoi.center_lon, aoi.center_lat, aoi.side_m)


def processing_bbox(aoi: AOI) -> tuple[float, float, float, float]:
    """MintPy load/clip bbox — `processing_side_m` if set, else the display bbox.

    Wider than `bbox` so it can contain the reference anchor (which sits outside
    the display tile) and give every interferogram one identical clip extent.
    """
    side = aoi.processing_side_m if aoi.processing_side_m is not None else aoi.side_m
    return _square_bbox(aoi.center_lon, aoi.center_lat, side)
