# PA Housing Trend Predictor

This project predicts 12-month housing value change at the ZIP level in Pennsylvania using:

- Zillow Home Value Index (ZHVI)
- American Community Survey (ACS)
- EPA Toxics Release Inventory (TRI)
- FEMA Flood Hazard Data

## Features
- Housing momentum signals
- Socioeconomic indicators
- Environmental exposure context
- Flood hazard proxy

## Model
- Linear Regression (final model)
- Compared with Random Forest

## Results
- MAE: ~1.44
- RMSE: ~2.50
- R²: ~0.75

Environmental features added contextual insight but minimal predictive lift.

## Streamlit App

Run locally:

```bash
python -m streamlit run app.py