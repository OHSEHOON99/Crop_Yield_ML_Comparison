# Data and Artifact Scope

This repository is structured as a public research-code repository. It includes the compact processed tables and curated result CSVs needed to review the machine learning comparison without downloading the full raw geospatial stack.

## Included In Git

- `data/processed/combined_data.csv`: merged regional monthly MODIS and ERA5-Land feature table
- `data/processed/crop_yield.csv`: regional crop yield table
- `configs/search_spaces/*.yaml`: hyperparameter search-space definitions
- `results/`: curated metrics, predictions, and selected hyperparameters
- `notebooks/`: supplementary notebooks with execution outputs cleared

## External Or Regenerated Locally

The following artifacts are better kept in a local data workspace because they are large, source-derived, or intermediate:

- ERA5-Land NetCDF files
- MODIS raster products
- crop map rasters
- administrative boundary GeoJSON files
- raw yearly crop-yield spreadsheets
- generated sample-point CSVs and GeoTIFFs
- exploratory output folders from preprocessing runs

A recommended local layout is:

```text
data/external/
data/raw/
data/intermediate/
data/processed/
outputs/
```

## Data Sources

The workflow is based on public or research data sources such as ERA5-Land, MODIS-derived products, crop map rasters, administrative boundary data, and regional crop-yield statistics. When reproducing the preprocessing workflow, use the original providers' citation, license, and redistribution terms for the raw files.
