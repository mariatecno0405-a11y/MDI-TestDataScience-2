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
├─ README.md
├─ requirements.txt              
├─ test2-ds.yml           # (optional - conda)
├── notebooks/
│   ├── 01_EDA.ipynb              # Data loading, cleaning, target construction, non-seasonality checks
│   ├── 02_modeling.ipynb         # Backtesting + final 100-step forecast
├─ data/
│  ├─ raw/AirQualityUCI.csv          # raw CSV placed here by user if needed
│  └─ processed/air_quality_daily.parquet
├─ reports/
│  ├─ forecast_100.csv           # 100-step forecast (deltas + reconstructed level)
│  └─ figures/                   # (optional) figures saved by notebooks

```

## How To Run

1. Execute `01_EDA.ipynb` end-to-end → generates `data/processed/air_quality_daily.parquet`.  
3. Execute `02_modeling.ipynb` end-to-end → backtest tables + `reports/forecast_100.csv`.

To export the HTML notebooks:
```bash
jupyter nbconvert --to html 01_EDA.ipynb 02_modeling.ipynb
```

## Data Ingestion

The EDA notebook loads the dataset from `data/raw/AirQualityUCI.csv` or from UCI (note that some environments may block outbound connections).  
- **If offline**, download the CSV from the UCI repository and place it at: `data/raw/AirQualityUCI.csv`.

Cleaning steps:
- Original CSV uses semicolons and decimal commas → parsed accordingly.
- Missing values indicated with **-200** → converted to `NaN`.
- Timestamp built from `Date` + `Time`, with an **hourly** frequency; later resampled to **daily means**.

## Target construction & Non-seasonality verification

- **Target:** `y_nonseasonal = diff(daily_mean(CO_GT))`  
- **Diagnostics** (in `01_EDA.ipynb`):
  - **STL** with weekly period → **seasonal strength** near 0 (non-seasonal).  
  - **Periodogram** → no pronounced weekly peak.  
  - **Ljung–Box** at lags 7/14/21 → no significant seasonal autocorrelation.

The EDA stores the processed dataset as `data/processed/air_quality_daily.parquet`.


## Modeling strategies (constraint: no future exogenous)

### S1 — **Univariate models** (robust)
- **Baselines:** Naive (last value), Drift.  
- **ARIMA** (non-seasonal; small grid over (p,d,q)).  
- **ETS/SES** (trend-only; no seasonal component).  
- **ML univariate** with **lag features** (e.g., `HistGradientBoostingRegressor`) and **recursive rollout** for multi-step forecasting.

> These models **do not use exogenous features**, so the “no future regressors” constraint is naturally satisfied.

### S2 — **Multivariate (optional)** with **forecasted exogenous**
- Forecast each candidate exogenous series **individually** (e.g., temperature `T`, humidity `RH`, etc.).  
- Feed those **forecasted** trajectories to SARIMAX / ML as exogenous inputs for the 100-step target forecast.  
- This **still satisfies** the constraint, as **no ground-truth future exogenous values** are used.


## Backtesting protocol & Metrics

- **Expanding window** cross-validation (`TimeSeriesSplit`) with ~6 folds.  
- In each fold, models are refit on the training window and evaluated on the hold-out window.  
- **Metrics**:  
  - **MAE** (robust)  
  - **RMSE** (penalizes large errors)  
  - **MASE** (scale-free; baseline = naive-1).  
- We **avoid MAPE** since the differenced target can be close to zero.

The modeling notebook aggregates metrics **per model** and selects the **best average MASE** model.


## Final 100-step forecast

- Fit the **selected model** on the **full history** of `y_nonseasonal`.  
- Forecast **100 days** ahead for the **delta target**.  
- Reconstruct the **level** series (daily CO) by cumulatively adding predicted deltas to the last observed level.  
- Save outputs to:  
  - `reports/forecast_100.csv` (columns: `y_delta_hat`, `CO_daily_forecast`).

> Figures are displayed inline in the notebook; you may save them under `reports/figures/` if desired.




## Results and Conclusions 

- The differenced daily CO target behaves **non-seasonally**.  
- In backtesting, **[fill here: winning model]** shows the best **MASE** across folds.  
- The **100-day forecast** for CO (level reconstructed from deltas) is **[trend/variance observation here]**.  
- **Trade-offs:** univariate models are simpler and robust under the constraint; multivariate models can help **only** if exogenous forecasts carry additional signal and are themselves reliable.
