#!/usr/bin/env python3
"""
mangrove_candidate_mask_from_land.py
------------------------------------

Create a “candidate mangrove” layer:
    • within ±30° latitude
    • within 20 km of the present-day coastline

Input  : Natural Earth 1:10 m Land  –  data/ne_10m_land/ne_10m_land.shp
Output : mangrove_candidate_mask.gpkg (EPSG:4326)

Dependencies
------------
pip install geopandas shapely tqdm
"""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm

# ----------------------------------------------------------------------
LAND_SHAPEFILE = Path("data/ne_10m_land/ne_10m_land.shp")

BUFFER_KM  = 30     # width of coastal strip
LAT_LIMIT  = 30     # ± latitude band
OUTFILE    = "mangrove_candidate_mask.gpkg"
METRIC_CRS = 8857   # Equal Earth – distances in metres
# ----------------------------------------------------------------------


def build_lat_band(limit: int) -> gpd.GeoDataFrame:
    """Return a ±limit-degree latitude band."""
    band = box(-180, -limit, 180, limit)
    return gpd.GeoDataFrame(geometry=[band], crs="EPSG:4326")


def main() -> None:
    if not LAND_SHAPEFILE.exists():
        raise FileNotFoundError(f"{LAND_SHAPEFILE} not found")

    # 1.  Load Natural Earth land polygons (WGS-84)
    land = gpd.read_file(LAND_SHAPEFILE)
    lat_band = build_lat_band(LAT_LIMIT)

    # 2.  Get the coastline as a multiline – polygon boundaries
    coast_lines = land.boundary
    coast_gdf   = gpd.GeoDataFrame(geometry=coast_lines, crs=land.crs)

    # 3.  Re-project to metric CRS for buffering
    coast_m   = coast_gdf.to_crs(METRIC_CRS)
    band_m    = lat_band.to_crs(METRIC_CRS)

    # 4.  Buffer and union
    print(f"Buffering {BUFFER_KM} km strip …")
    buf_union = coast_m.buffer(BUFFER_KM * 1_000).unary_union
    buf_gdf   = gpd.GeoDataFrame(geometry=[buf_union], crs=METRIC_CRS)

    # 5.  Clip to ±30° band
    print("Clipping to latitude band …")
    candidate = gpd.overlay(buf_gdf, band_m, how="intersection")

    # 6.  Back to WGS-84 and save
    candidate = candidate.to_crs(4326)
    candidate.to_file(OUTFILE, driver="GPKG")
    print(f"✓ wrote {OUTFILE} with {len(candidate)} polygon part(s)")


if __name__ == "__main__":
    main()
