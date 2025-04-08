---
title: Statsforecast
emoji: 🔥
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
short_description: A Demo of statforecast methods
---

# StatsForecast Demo App

This demo application showcases various time series forecasting models from the [StatsForecast](https://github.com/Nixtla/statsforecast) package.

## Features

- Upload your own time series data in CSV format
- Choose from multiple forecasting models:
  - Historical Average
  - Naive
  - Seasonal Naive
  - Window Average
  - Seasonal Window Average
  - AutoETS
  - AutoARIMA
- Configure evaluation strategy:
  - Fixed Window
  - Cross Validation
- View performance metrics (ME, MAE, RMSE, MAPE)
- Visualize forecasts

## How to Use

1. Upload a CSV file with time series data containing:
   - `unique_id` column: Identifier for each time series
   - `ds` column: Date/timestamp
   - `y` column: Target values

2. Configure:
   - Frequency (D=daily, H=hourly, M=monthly, etc.)
   - Evaluation strategy and parameters
   - Select models and their parameters

3. Click "Run Forecast" to see results

## Sample Data Format

Your CSV should look like this:

```
unique_id,ds,y
series1,2023-01-01,100
series1,2023-01-02,105
series1,2023-01-03,98
...
```

## About StatsForecast

StatsForecast is a Python library that provides statistical forecasting algorithms for time series data. It is fast and scalable and offers many classical forecasting methods.

For more information, visit [Nixtla's StatsForecast repository](https://github.com/Nixtla/statsforecast).
