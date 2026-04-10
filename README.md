##Sales Forecasting

Predict weekly retail sales using time-series feature engineering and multiple regression models, based on a Walmart-style dataset.

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd sales_forecasting

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python sales_forecasting.py
```

Output plots are saved automatically to the working directory:
- `sales_forecasting_results.png`
- `seasonal_decomposition.png`

---

## 📁 Project Structure

```
sales_forecasting/
│
├── sales_forecasting.py        # Main script (all steps)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── outputs/                    # Auto-generated on run
    ├── sales_forecasting_results.png
    └── seasonal_decomposition.png
```

---

## 🔬 What the Script Does

| Step | Description |
|------|-------------|
| 1 | Simulates 143 weeks of Walmart-style sales (trend + seasonality + holidays + noise) |
| 2 | Engineers 22 time-based features (lags, rolling stats, calendar, exogenous) |
| 3 | Time-aware train/test split — last 20 weeks held out (no data leakage) |
| 4 | Trains 4 models + rolling average baseline |
| 5 | Evaluates with MAE, RMSE, R² |
| 6 | Plots actual vs predicted, model comparison, feature importance |
| 7 | BONUS: Seasonal decomposition (trend / seasonal / residual) |

---

## 🧠 Models Used

| Model | Notes |
|-------|-------|
| Linear Regression | Fast baseline; uses StandardScaler |
| Ridge Regression | L2-regularized linear model |
| Random Forest | Ensemble of 300 decision trees |
| Gradient Boosting (GBM) | sklearn's GBM ≈ XGBoost equivalent |
| Rolling Avg Baseline | 4-week rolling mean — naive benchmark |

---

## 📊 Features Engineered

- **Calendar**: Year, Month, Week of year, Quarter
- **Lag features**: 1, 2, 4, 8, 13, 26, 52 weeks back
- **Rolling mean**: 4, 8, 13, 26-week windows (lag-1 to avoid leakage)
- **Rolling std**: Same windows
- **Exogenous**: Temperature, Fuel Price, IsHoliday

---

## 🎯 Key Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression ⭐ | $3,046 | $3,579 | 0.554 |
| Ridge Regression | $2,982 | $3,737 | 0.514 |
| Random Forest | $2,947 | $3,830 | 0.490 |
| Gradient Boosting | $3,269 | $4,085 | 0.419 |
| Rolling Avg Baseline | $3,663 | $4,185 | 0.391 |

---

## 🔁 Using the Real Walmart Dataset (Kaggle)

1. Download from: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data
2. Place `train.csv`, `stores.csv`, `features.csv` in the project folder
3. Replace the data simulation block (Step 1) with:

```python
train_df   = pd.read_csv("train.csv",    parse_dates=["Date"])
stores_df  = pd.read_csv("stores.csv")
features_df = pd.read_csv("features.csv", parse_dates=["Date"])

df = train_df.merge(stores_df, on="Store").merge(features_df, on=["Store","Date","IsHoliday"])
df = df.sort_values("Date").set_index("Date")
# Then filter to a single store or aggregate, and continue from Step 2
```

---

## ⚙️ Python Version

Tested on **Python 3.9+**. Recommended: **Python 3.10 or 3.11**.
