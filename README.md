# Mangrove Estimation with Clay Foundation Model

This project explores mangrove predictions through time using [Clay](https://clay-foundation.github.io/model/), a geospatial foundation model. We generate embeddings from Landsat imagery and analyze their relationship to mangrove coverage.

## Workflow

### 1. Generate Embedding Areas
```bash
python generate_embedding_area.py
```
Creates spatial polygons for embedding extraction. These polygons define the geographic areas where Landsat embeddings will be generated.

### 2. Generate Landsat Embeddings
*Note: Embedding generation happens outside this repository using Clay foundation model*

Landsat imagery within the generated polygons is processed through Clay to produce high-dimensional embeddings that capture geospatial features.

### 3. Calculate Mangrove Coverage
```bash
python calculate_embedding_coverage_fixed.py \
    data/embedding_file.parquet \
    data/gmw_v3_2020_gtiff/gmw_v3_2020 \
    --verbose
```
Intersects embedding polygons with Global Mangrove Watch (GMW) raster data to calculate the percentage of mangrove coverage within each embedding area.

### 4. Quality Assurance Analysis
```bash
jupyter notebook mangrove_coverage_qa.ipynb
```
Performs comprehensive QA analysis including:
- **Visual QA**: Side-by-side comparison of GMW rasters and embedding polygons
- **Clustering Analysis**: K-means clustering of embeddings with spatial visualization  
- **Dimensionality Reduction**: t-SNE and PCA analysis colored by mangrove coverage
- **Predictive Modeling**: Neural network training to predict coverage from embeddings

## Key Files

- `generate_embedding_area.py` - Generate spatial polygons for embedding extraction
- `mangrove_candidate_mask.gpkg` - Generated polygon mask showing embedding extent (view in GIS tools for validation)
- `calculate_embedding_coverage.py` - Calculate mangrove coverage percentages
- `mangrove_coverage_qa.ipynb` - Quality assurance and analysis notebook

## Environment Setup

Create and activate the conda environment:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate mangrove-clay
```

The environment includes all necessary dependencies for geospatial analysis, machine learning, and the Clay foundation model.

## Data Sources

- **Landsat imagery**: Processed through Clay foundation model
- **Global Mangrove Watch (GMW)**: Mangrove extent raster data
- **Embedding polygons**: Generated spatial areas for analysis
