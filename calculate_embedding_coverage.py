#!/usr/bin/env python3
"""
calculate_embedding_coverage_fixed.py
-------------------------------------

Calculate the percent coverage of mangroves in each embedding chip by intersecting
embedding tiles with Global Mangrove Watch (GMW) raster data.

FIXED VERSION: Uses actual raster bounds from metadata instead of filename parsing.

Input:
- Embedding parquet files (e.g., mangrove-test_landsat-c2l2-sr_128_2020_tile_embeddings_final_1467.parquet)
- GMW raster tiles (e.g., GMW_N00E008_2020_v3.tif)

Output:
- Enhanced parquet files with mangrove coverage statistics

Dependencies:
pip install geopandas pandas rasterio shapely tqdm numpy
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class MangroveeCoverageCalculator:
    """Calculate mangrove coverage for embedding tiles using GMW raster data."""
    
    def __init__(self, gmw_directory: Path, verbose: bool = True):
        """
        Initialize the coverage calculator.
        
        Args:
            gmw_directory: Path to directory containing GMW raster files
            verbose: Whether to print progress information
        """
        self.gmw_directory = Path(gmw_directory)
        self.verbose = verbose
        self.gmw_files_cache = {}
        self._build_gmw_index()
    
    def _build_gmw_index(self) -> None:
        """Build an index of available GMW files by reading their actual bounds from metadata."""
        self.gmw_index = {}
        
        if self.verbose:
            print("Building GMW file index using actual raster bounds...")
        
        gmw_files = list(self.gmw_directory.glob("GMW_*.tif"))
        
        for gmw_file in tqdm(gmw_files, desc="Reading GMW metadata", disable=not self.verbose):
            try:
                with rasterio.open(gmw_file) as src:
                    bounds = src.bounds
                    # Store as (minx, miny, maxx, maxy) tuple
                    bounds_tuple = (bounds.left, bounds.bottom, bounds.right, bounds.top)
                    self.gmw_index[bounds_tuple] = gmw_file
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not read bounds for {gmw_file.name}: {e}")
                continue
        
        if self.verbose:
            print(f"Successfully indexed {len(self.gmw_index)} GMW tiles")
            
            if len(self.gmw_index) > 0:
                # Show bounds summary
                all_bounds = list(self.gmw_index.keys())
                min_lon = min(b[0] for b in all_bounds)
                max_lon = max(b[2] for b in all_bounds)
                min_lat = min(b[1] for b in all_bounds)
                max_lat = max(b[3] for b in all_bounds)
                print(f"GMW coverage: {min_lon:.3f} to {max_lon:.3f}°E, {min_lat:.3f} to {max_lat:.3f}°N")
    
    def _find_intersecting_gmw_tiles(self, bounds: Tuple[float, float, float, float]) -> List[Path]:
        """
        Find GMW tiles that intersect with the given bounds using actual raster extents.
        
        Args:
            bounds: (minx, miny, maxx, maxy) in WGS84
            
        Returns:
            List of paths to intersecting GMW files
        """
        minx, miny, maxx, maxy = bounds
        intersecting_files = []
        
        for gmw_bounds, gmw_file in self.gmw_index.items():
            gmw_minx, gmw_miny, gmw_maxx, gmw_maxy = gmw_bounds
            
            # Check if bounding boxes intersect
            # Two rectangles intersect if they overlap in both X and Y dimensions
            x_overlap = not (maxx <= gmw_minx or minx >= gmw_maxx)
            y_overlap = not (maxy <= gmw_miny or miny >= gmw_maxy)
            
            if x_overlap and y_overlap:
                intersecting_files.append(gmw_file)
        
        return intersecting_files
    
    def _load_gmw_tile(self, gmw_file: Path) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
        """
        Load a GMW raster tile.
        
        Args:
            gmw_file: Path to GMW raster file
            
        Returns:
            Tuple of (data array, transform, crs)
        """
        if gmw_file in self.gmw_files_cache:
            return self.gmw_files_cache[gmw_file]
        
        with rasterio.open(gmw_file) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
        
        # Cache the result for reuse
        self.gmw_files_cache[gmw_file] = (data, transform, crs)
        return data, transform, crs
    
    def calculate_coverage(self, embedding_polygon, embedding_bounds: Tuple[float, float, float, float]) -> Dict:
        """
        Calculate mangrove coverage for a single embedding tile.
        
        Args:
            embedding_polygon: Shapely polygon of the embedding tile
            embedding_bounds: (minx, miny, maxx, maxy) bounds of the embedding tile
            
        Returns:
            Dictionary with coverage statistics
        """
        intersecting_gmw_files = self._find_intersecting_gmw_tiles(embedding_bounds)
        
        if not intersecting_gmw_files:
            return {
                'mangrove_coverage_percent': 0.0,
                'mangrove_pixel_count': 0,
                'total_valid_pixels': 0,
                'gmw_tiles_intersected': [],
                'gmw_bounds_checked': len(self.gmw_index)
            }
        
        total_mangrove_pixels = 0
        total_valid_pixels = 0
        gmw_tiles_used = []
        
        for gmw_file in intersecting_gmw_files:
            try:
                # Create a temporary GeoDataFrame for the embedding polygon
                temp_gdf = gpd.GeoDataFrame([1], geometry=[embedding_polygon], crs="EPSG:4326")
                
                # Mask the raster with the embedding polygon
                try:
                    with rasterio.open(gmw_file) as src:
                        # Reproject embedding polygon to match raster CRS if needed
                        if str(src.crs) != "EPSG:4326":
                            temp_gdf = temp_gdf.to_crs(src.crs)
                        
                        # FIX: Don't force nodata=255, use the raster's original nodata handling
                        # The GMW rasters have nodata=0, but 0 also means "no mangrove detected"
                        # All pixels within the polygon boundary should be considered valid data
                        masked_data, masked_transform = mask(
                            src,
                            temp_gdf.geometry, 
                            crop=True, 
                            all_touched=False  # Only pixels whose center is within the polygon
                        )
                    
                    if masked_data.size == 0:
                        continue
                    
                    # FIXED LOGIC: All pixels within the polygon are valid geographic data
                    # - Value 0: Non-mangrove area (land/water with no mangroves)
                    # - Value 1: Mangrove area
                    # The mask operation already handled the geographic boundaries
                    data_1d = masked_data[0]
                    total_pixels_in_polygon = data_1d.size
                    mangrove_pixels = (data_1d == 1)
                    
                    total_mangrove_pixels += np.sum(mangrove_pixels)
                    total_valid_pixels += total_pixels_in_polygon
                    gmw_tiles_used.append(gmw_file.name)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not mask {gmw_file.name}: {e}")
                    continue
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not process {gmw_file.name}: {e}")
                continue
        
        # Calculate coverage percentage
        coverage_percent = 0.0
        if total_valid_pixels > 0:
            coverage_percent = (total_mangrove_pixels / total_valid_pixels) * 100.0
        
        return {
            'mangrove_coverage_percent': coverage_percent,
            'mangrove_pixel_count': int(total_mangrove_pixels),
            'total_valid_pixels': int(total_valid_pixels),
            'gmw_tiles_intersected': gmw_tiles_used,
            'gmw_bounds_checked': len(self.gmw_index)
        }
    
    def process_embedding_file(self, embedding_file: Path, output_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Process an entire embedding file and calculate coverage for all tiles.
        
        Args:
            embedding_file: Path to input embedding parquet file
            output_file: Optional path for output file. If None, adds '_with_coverage_fixed' suffix
            
        Returns:
            DataFrame with coverage statistics added
        """
        if self.verbose:
            print(f"Processing {embedding_file.name}...")
        
        # Load embedding data as GeoDataFrame to preserve CRS info
        gdf = gpd.read_parquet(embedding_file)
        
        # Ensure we have geometry
        if 'geometry' not in gdf.columns:
            raise ValueError("Missing geometry column in embedding data")
        
        # Ensure CRS is WGS84
        if gdf.crs is None:
            if self.verbose:
                print("Warning: No CRS found, assuming WGS84")
            gdf = gdf.set_crs("EPSG:4326")
        elif str(gdf.crs) != "EPSG:4326":
            if self.verbose:
                print(f"Reprojecting from {gdf.crs} to WGS84")
            gdf = gdf.to_crs("EPSG:4326")
        
        # Show embedding bounds for verification
        if self.verbose:
            total_bounds = gdf.total_bounds
            print(f"Embedding bounds: {total_bounds[0]:.3f} to {total_bounds[2]:.3f}°E, {total_bounds[1]:.3f} to {total_bounds[3]:.3f}°N")
        
        # Initialize new columns
        gdf['mangrove_coverage_percent'] = 0.0
        gdf['mangrove_pixel_count'] = 0
        gdf['total_valid_pixels'] = 0
        gdf['gmw_tiles_intersected'] = None  # Will be filled with lists
        gdf['gmw_bounds_checked'] = 0
        
        # Initialize the list column properly
        gdf['gmw_tiles_intersected'] = gdf['gmw_tiles_intersected'].astype('object')
        for idx in gdf.index:
            gdf.at[idx, 'gmw_tiles_intersected'] = []
        
        # Process each embedding tile
        successful_intersections = 0
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Calculating coverage"):
            # Get polygon geometry and bounds
            polygon = row['geometry']
            bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            
            # Calculate coverage
            coverage_stats = self.calculate_coverage(polygon, bounds)
            
            # Update GeoDataFrame
            gdf.loc[idx, 'mangrove_coverage_percent'] = coverage_stats['mangrove_coverage_percent']
            gdf.loc[idx, 'mangrove_pixel_count'] = coverage_stats['mangrove_pixel_count']
            gdf.loc[idx, 'total_valid_pixels'] = coverage_stats['total_valid_pixels']
            gdf.loc[idx, 'gmw_bounds_checked'] = coverage_stats['gmw_bounds_checked']
            gdf.at[idx, 'gmw_tiles_intersected'] = coverage_stats['gmw_tiles_intersected']
            
            if len(coverage_stats['gmw_tiles_intersected']) > 0:
                successful_intersections += 1
        
        # Save results
        if output_file is None:
            stem = embedding_file.stem
            output_file = embedding_file.parent / f"{stem}_with_coverage_fixed.parquet"
        
        gdf.to_parquet(output_file)
        
        if self.verbose:
            print(f"Saved results to {output_file}")
            print(f"Coverage summary:")
            print(f"  - Total tiles processed: {len(gdf)}")
            print(f"  - Tiles with intersecting GMW data: {successful_intersections}")
            print(f"  - Tiles with mangroves: {(gdf['mangrove_coverage_percent'] > 0).sum()}")
            print(f"  - Mean coverage: {gdf['mangrove_coverage_percent'].mean():.2f}%")
            print(f"  - Max coverage: {gdf['mangrove_coverage_percent'].max():.2f}%")
            print(f"  - Coverage > 0: {(gdf['mangrove_coverage_percent'] > 0).sum()} tiles")
        
        return gdf


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Calculate mangrove coverage for embedding tiles (FIXED VERSION)")
    parser.add_argument("embedding_file", type=Path, help="Path to embedding parquet file")
    parser.add_argument("gmw_directory", type=Path, help="Path to directory containing GMW raster files")
    parser.add_argument("-o", "--output", type=Path, help="Output file path (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.embedding_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {args.embedding_file}")
    
    if not args.gmw_directory.exists():
        raise FileNotFoundError(f"GMW directory not found: {args.gmw_directory}")
    
    # Initialize calculator
    calculator = MangroveeCoverageCalculator(args.gmw_directory, verbose=args.verbose)
    
    # Process the embedding file
    result_df = calculator.process_embedding_file(args.embedding_file, args.output)
    
    print("Coverage calculation complete!")


if __name__ == "__main__":
    main()