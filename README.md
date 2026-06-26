# Ukraine Crop Yield Estimation with Machine Learning

This repository contains code, notebooks, processed tabular data, and curated experiment outputs for estimating regional wheat yield in Ukraine using MODIS-derived vegetation variables, ERA5-Land weather variables, and machine learning regressors.

The workflow adapts the county-level crop yield prediction methodology from:

Ju, S., Lim, H., Ma, J. W., Kim, S., Lee, K., Zhao, S., & Heo, J. (2021). Optimal county-level crop yield prediction using MODIS-based variables and weather data: A comparative study on machine learning models. Agricultural and Forest Meteorology, 307, 108530. https://doi.org/10.1016/j.agrformet.2021.108530

## What Is Included

- `code/`: reusable Python modules for data loading, Bayesian hyperparameter search, leave-one-year-out evaluation, scoring, and plotting
- `notebooks/`: supplementary preprocessing and modeling notebooks with outputs cleared
- `data/processed/`: compact processed CSV files used by the modeling workflow
- `configs/search_spaces/`: Bayesian optimization search spaces for each model
- `results/`: curated model metrics, predictions, and selected best hyperparameters
- `DATA_POLICY.md`: data provenance and artifact scope for public use

Large raw geospatial/weather files and intermediate extraction outputs are represented by the documented workflow rather than stored directly in this repository.

## Models

The comparison covers six regression models:

- Support Vector Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Decision Tree Regressor
- K-Nearest Neighbors Regressor

The default experiment uses 21 Ukrainian regions, March-to-October monthly features, and leave-one-year-out evaluation.

## Installation

Use a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For notebook work:

```bash
jupyter lab
```

Open notebooks from the repository root so the relative paths resolve as expected.

## Data

The repository includes two compact processed CSV files:

- `data/processed/combined_data.csv`: regional monthly MODIS and ERA5-Land features for 2010-2023
- `data/processed/crop_yield.csv`: regional crop yield table for 2010-2023

Raw and intermediate files used to produce these tables can be placed locally under:

```text
data/external/
data/raw/
data/intermediate/
outputs/
```

See `DATA_POLICY.md` for source and artifact details.

## Modeling Workflow

The main modeling workflow is in:

```text
notebooks/modeling/training.ipynb
```

The notebook loads:

```text
data/processed/combined_data.csv
data/processed/crop_yield.csv
configs/search_spaces/*.yaml
```

and writes refreshed local outputs to `results/` or `outputs/`, depending on whether you are reproducing the curated result tables or generating exploratory artifacts.

## Results

Curated result tables are available in:

```text
results/11_20/
results/12_20/
```

Each experiment folder contains:

- `*_metrics.csv`: yearly and aggregate metrics
- `*_results.csv`: true and predicted yield by region and year
- `hyperparams/*_best_params.csv`: selected model hyperparameters

## Notes

Preprocessing notebooks are kept as supplementary material because the raw geospatial workflow depends on external files and local compute environment details. The modeling code and compact processed data are the recommended entry point for reviewing or rerunning the machine learning comparison.
