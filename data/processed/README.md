# Processed Data

This folder contains compact processed tables used by the modeling workflow.

## Files

- `combined_data.csv`
  - Shape: 2,464 rows x 16 columns
  - Contents: region, year, month, ERA5-Land weather variables, and MODIS-derived vegetation/temperature variables
- `crop_yield.csv`
  - Shape: 6,669 rows x 7 columns
  - Contents: region, year, crop identifier, area, production, and yield

These files are the recommended starting point for reproducing the model comparison without rebuilding the full raw geospatial preprocessing pipeline.
