# TEST 2 - TIME SERIES - REGRESSION
**Author:** María Donoso  
**Dataset:** Vito, S. (2008). *Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F

## Objective

Build a fully reproducible pipeline to:
1. Prepare a **non-seasonal target** from the UCI Air Quality dataset.  
2. Verify **non-seasonality** with standard diagnostics.  
3. Train, backtest and compare forecasting models that **do not rely on future exogenous values**.  
4. Produce a **100-step ahead** forecast, save the outputs, and document results and conclusions.

### Why this dataset fits the brief
- **Non-financial** multivariate time series (air quality sensors, meteorological variables).  
- The target used here is the **daily first difference** of the daily-mean CO concentration from the reference analyzer:  
$$y_t = \mathrm{CO\_daily}_t - \mathrm{CO\_daily}_{t-1}$$


  This construction **removes seasonality** at the level component, and we **verify** non-seasonality using STL, periodogram, and Ljung–Box.
- During forecasting we **do not assume** access to any **future exogenous** values.

## Reproducibility
- **Python**: 3.11+  
- Key packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `statsmodels`, `scipy`, `pyarrow`

### Option A — `pip + venv`
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```


### Option B — conda (optional)
```bash
conda env create -f test2-ds.yml
conda activate test2-ds
```


## Repository Layout

```rb
MDI-TestDataScience-2/  
├── README.md
├── requirements.txt              
├── test2-ds.yml           # (optional - conda)
├── notebooks/
│   ├── 01_EDA.ipynb              # Data loading + cleaning + target construction + non-seasonality checks
│   ├── 02_modeling.ipynb         # Backtesting + final 100-step forecast + conclusions
├── data/
│   ├─ raw/                        # raw CSV placed here by user if needed
│   └─ processed/
├── src/                            # src code directory (example only; not required to run for this test)
├── reports/
│   ├─ forecast_100.csv           # 100-step forecast (deltas + reconstructed level)
├── docs/                 # notebooks on HTML
│   ├── 01_EDA.html             
│   ├── 02_modeling.html
├── deployment/           # FastAPI serving endpoints (example only; not required to run for this test)
│   ├── app.py            
│   ├── Dockerfile
│   └── requirements-deploy.txt  
```

## How To Run

1. Execute `01_EDA.ipynb` end-to-end → generates `data/processed/air_quality_daily.parquet`.  
3. Execute `02_modeling.ipynb` end-to-end → backtest tables + `reports/forecast_100.csv`.

To export the HTML notebooks:
```bash
jupyter nbconvert --output-dir "../docs" --to html 01_EDA.ipynb 02_modeling.ipynb
```

## Data Ingestion

The EDA notebook loads the dataset from `data/raw/AirQualityUCI.csv` or from UCI (note that some environments may block outbound connections). **If offline**, download the CSV from the UCI repository and place it at: `data/raw/AirQualityUCI.csv`.

**Attribute Documentation** 
- `Date`: (DD/MM/YYYY)
- `Time`: (HH.MM.SS)
- `CO(GT)`: True hourly averaged concentration CO in mg/m^3  (reference analyzer).
- `PT08.S1(CO)`: (tin oxide) hourly averaged sensor response (nominally  CO targeted).
- `NMHC(GT)`: True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer).
- `C6H6(GT)`: True hourly averaged Benzene concentration  in microg/m^3 (reference analyzer).
- `PT08.S2(NMHC)`: (titania) hourly averaged sensor response (nominally NMHC targeted).
- `NOx(GT)`: True hourly averaged NOx concentration  in ppb (reference analyzer).
- `PT08.S3(NOx)`: (tungsten oxide) hourly averaged sensor response (nominally NOx targeted).
- `NO2(GT)`: True hourly averaged NO2 concentration in microg/m^3 (reference analyzer).
- `PT08.S4(NO2)`: (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted).
- `PT08.S5(O3)`: (indium oxide) hourly averaged sensor response (nominally O3 targeted).
- `T`: Temperature in °C.
- `RH`: Relative Humidity (%).
- `AH`: Absolute Humidity.


## Exploratory Data Analysis (notebook `01_EDA.ipynb`)
- Loads data from `data/raw/AirQualityUCI.csv`or from UCI URL.
- Cleans:
  - Original CSV uses semicolons and decimal commas → parsed accordingly.
  - Missing values indicated with **-200** → converted to `NaN`.
  - Timestamp built from `Date` + `Time`, with an **hourly** frequency; later resampled to **daily means**.
- **Target construction & Non-seasonality verification**
  - **Target:** `y_nonseasonal = diff(daily_mean(CO_GT))`  
  - **Diagnostics:**
    - **STL** with weekly period → **seasonal strength** near 0 (non-seasonal).  
    - **Periodogram** → no pronounced weekly peak.  
    - **Ljung–Box** at lags 7/14/21 → no significant seasonal autocorrelation.

The EDA stores the processed dataset as `data/processed/air_quality_daily.parquet`.

## Modeling (notebook 02_modeling.ipynb)

### Univariate models (robust)
- **Baselines:** Naive (last value), Drift.  
- **ARIMA** (non-seasonal; small grid over (p,d,q)).  
- **ETS/SES** (trend-only; no seasonal component).  
- **ML univariate** with **lag features** (e.g., `HistGradientBoostingRegressor`) and **recursive rollout** for multi-step forecasting.

> These models **do not use exogenous features**, so the “no future regressors” constraint is naturally satisfied.



### Backtesting protocol & Metrics

- **Expanding window** cross-validation (`TimeSeriesSplit`) with ~6 folds.  
- In each fold, models are refit on the training window and evaluated on the hold-out window.  
- **Metrics**:  
  - **MAE** (robust)  
  - **RMSE** (penalizes large errors)  
  - **MASE** (scale-free; baseline = naive-1).  
- We **avoid MAPE** since the differenced target can be close to zero.

The notebook aggregates the metrics across folds, plots **boxplots** (stability) and **bar charts** (mean MASE), and selects the **best average MASE** model.

### Final 100-step forecast

- Fit the **selected model** on the **full history** of `y_nonseasonal`.  
- Forecast **100 days** ahead for the **delta target**.  
- **Reconstruct the level** series (`CO_daily`) by cumulatively adding predicted deltas to the last observed level.  
- Save outputs to:  
  - `reports/forecast_100.csv` (columns: `y_delta_hat`, `CO_daily_forecast`).


## Results and Conclusions 

- The differenced daily CO target behaves **non-seasonally** (as established in EDA), so **seasonal** baselines/models are intentionally omitted.
- In expanding-window backtesting, the notebook prints the **winning model** (lowest mean **MASE**) and its summary table.  
  > In our reference run, the best performer came from the **ARIMA family** (order selected on the grid; e.g., `ARIMA(0,0,2)`), edging out ETS/SES and the ML-lags approach on mean MASE. If your local run picks a different order, **trust the model name that appears in the notebook summary**.
- **Forecast behaviour.** On a non-seasonal **difference** target, long-horizon forecasts often **flatten** toward zero; after level reconstruction, the 100-day `CO_daily` path appears **flat to gently trending** from the last observed value—consistent with the data’s short memory.

