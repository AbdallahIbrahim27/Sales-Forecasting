"""
Task 7: Sales Forecasting
Dataset: Simulated Walmart-style Sales Data
Tools: Python, Pandas, Matplotlib, Scikit-learn
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1.  SIMULATE WALMART-STYLE SALES DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("TASK 7: SALES FORECASTING")
print("=" * 60)

n_weeks = 143  # ~Walmart dataset length
dates = pd.date_range(start="2010-02-05", periods=n_weeks, freq="W")

# Build realistic sales with trend + seasonality + noise
t = np.arange(n_weeks)
trend = 40000 + 50 * t
seasonal = 8000 * np.sin(2 * np.pi * t / 52)          # yearly cycle
holiday_weeks = [50, 51, 100, 101]                     # holiday spikes
holiday_effect = np.zeros(n_weeks)
for w in holiday_weeks:
    if w < n_weeks:
        holiday_effect[w] = 25000

noise = np.random.normal(0, 3000, n_weeks)
sales = trend + seasonal + holiday_effect + noise

df = pd.DataFrame({
    "Date":        dates,
    "Weekly_Sales": sales,
    "Temperature": np.random.normal(60, 18, n_weeks),
    "Fuel_Price":  np.random.uniform(2.5, 4.5, n_weeks),
    "IsHoliday":   [1 if i in holiday_weeks else 0 for i in range(n_weeks)],
})
df = df.set_index("Date")

print(f"\nDataset: {len(df)} weeks  |  {df.index.min().date()} → {df.index.max().date()}")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# 2.  TIME-BASED FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[Feature Engineering]")

df["Year"]    = df.index.year
df["Month"]   = df.index.month
df["Week"]    = df.index.isocalendar().week.astype(int)
df["Quarter"] = df.index.quarter

# Lag features
for lag in [1, 2, 4, 8, 13, 26, 52]:
    df[f"Lag_{lag}"] = df["Weekly_Sales"].shift(lag)

# Rolling statistics
for window in [4, 8, 13, 26]:
    df[f"Roll_Mean_{window}"] = df["Weekly_Sales"].shift(1).rolling(window).mean()
    df[f"Roll_Std_{window}"]  = df["Weekly_Sales"].shift(1).rolling(window).std()

# Rolling average as a simple baseline
df["Rolling_Avg_Baseline"] = df["Weekly_Sales"].shift(1).rolling(4).mean()

# Seasonal decomposition (manual)
df["Sales_Diff1"]   = df["Weekly_Sales"].diff(1)
df["Sales_Diff52"]  = df["Weekly_Sales"].diff(52)

df.dropna(inplace=True)
print(f"After adding features & dropping NaN: {len(df)} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT  (last 20 weeks = test)
# ─────────────────────────────────────────────
TARGET    = "Weekly_Sales"
SKIP_COLS = [TARGET, "Rolling_Avg_Baseline", "Sales_Diff1", "Sales_Diff52"]
FEATURES  = [c for c in df.columns if c not in SKIP_COLS]

split = len(df) - 20
train, test = df.iloc[:split], df.iloc[split:]
X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"\nTrain: {len(train)} weeks | Test: {len(test)} weeks")
print(f"Features used: {len(FEATURES)}")

# ─────────────────────────────────────────────
# 4.  MODELS
# ─────────────────────────────────────────────
scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

models = {
    "Linear Regression":       LinearRegression(),
    "Ridge Regression":        Ridge(alpha=10),
    "Random Forest":           RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting (GBM)": GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
}

results = {}
preds   = {}

print("\n[Model Training & Evaluation]")
print(f"{'Model':<28} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
print("-" * 60)

for name, mdl in models.items():
    use_scaled = "Regression" in name
    Xtr = Xs_train if use_scaled else X_train
    Xte = Xs_test  if use_scaled else X_test

    mdl.fit(Xtr, y_train)
    pred = mdl.predict(Xte)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    preds[name]   = pred
    print(f"{name:<28} {mae:>10,.0f} {rmse:>10,.0f} {r2:>8.4f}")

# Rolling average baseline
base_pred = test["Rolling_Avg_Baseline"].values
mae_b  = mean_absolute_error(y_test, base_pred)
rmse_b = np.sqrt(mean_squared_error(y_test, base_pred))
r2_b   = r2_score(y_test, base_pred)
results["Rolling Avg Baseline"] = {"MAE": mae_b, "RMSE": rmse_b, "R2": r2_b}
preds["Rolling Avg Baseline"]   = base_pred
print(f"{'Rolling Avg Baseline':<28} {mae_b:>10,.0f} {rmse_b:>10,.0f} {r2_b:>8.4f}")

best_model = min(results, key=lambda k: results[k]["RMSE"])
print(f"\n★  Best model: {best_model}  (RMSE = {results[best_model]['RMSE']:,.0f})")

# ─────────────────────────────────────────────
# 5.  SEASONAL DECOMPOSITION (manual)
# ─────────────────────────────────────────────
period      = 52
sales_full  = df["Weekly_Sales"]
trend_comp  = sales_full.rolling(period, center=True).mean()
detrended   = sales_full - trend_comp
seasonal_comp = detrended.groupby(df.index.isocalendar().week.values).transform("mean")
residual    = sales_full - trend_comp - seasonal_comp

# ─────────────────────────────────────────────
# 6.  FEATURE IMPORTANCE  (best tree model)
# ─────────────────────────────────────────────
best_tree_name = "Gradient Boosting (GBM)" if results["Gradient Boosting (GBM)"]["R2"] >= results["Random Forest"]["R2"] else "Random Forest"
best_tree_mdl  = models[best_tree_name]
fi_series      = pd.Series(best_tree_mdl.feature_importances_, index=FEATURES).sort_values(ascending=False).head(12)

# ─────────────────────────────────────────────
# 7.  PLOTTING  (4 panels)
# ─────────────────────────────────────────────
plt.style.use("dark_background")
GOLD   = "#F5C518"
CYAN   = "#00D4FF"
RED    = "#FF4757"
GREEN  = "#2ED573"
PURPLE = "#A29BFE"
WHITE  = "#EFEFEF"
PANEL  = "#1A1A2E"
BG     = "#0D0D1A"

fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle("Task 7 — Walmart-Style Sales Forecasting",
             fontsize=22, fontweight="bold", color=WHITE, y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

palette = [GOLD, CYAN, GREEN, PURPLE, RED]

# ── Panel 1: Full historical sales + rolling mean ──
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(PANEL)
ax1.plot(df.index, df["Weekly_Sales"] / 1e3, color=CYAN, lw=1.2, alpha=0.7, label="Weekly Sales")
ax1.plot(df.index, df["Roll_Mean_13"] / 1e3, color=GOLD, lw=2.5, label="13-Week Rolling Mean")
ax1.axvline(test.index[0], color=RED, lw=1.5, ls="--", label="Train/Test Split")
ax1.fill_between(df.index, df["Weekly_Sales"] / 1e3, alpha=0.08, color=CYAN)
ax1.set_title("Historical Weekly Sales with Rolling Average", color=WHITE, fontsize=14)
ax1.set_ylabel("Sales ($K)", color=WHITE)
ax1.tick_params(colors=WHITE)
ax1.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=10)
ax1.spines[:].set_color("#333355")

# ── Panel 2: Actual vs Predicted (best model) ──
ax2 = fig.add_subplot(gs[1, :])
ax2.set_facecolor(PANEL)
ax2.plot(test.index, y_test / 1e3, color=WHITE, lw=2,  label="Actual", zorder=5)
for i, (name, pred) in enumerate(preds.items()):
    ls = "--" if name == "Rolling Avg Baseline" else "-"
    lw = 1.2 if name != best_model else 2.5
    ax2.plot(test.index, pred / 1e3, color=palette[i % len(palette)],
             lw=lw, ls=ls, alpha=0.85, label=name)
ax2.set_title("Actual vs Predicted Sales — Test Period (Last 20 Weeks)", color=WHITE, fontsize=14)
ax2.set_ylabel("Sales ($K)", color=WHITE)
ax2.tick_params(colors=WHITE)
ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9, ncol=3)
ax2.spines[:].set_color("#333355")

# ── Panel 3: Model Comparison Bar Chart ──
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_facecolor(PANEL)
model_names = list(results.keys())
rmse_vals   = [results[m]["RMSE"] / 1e3 for m in model_names]
colors_bar  = [GOLD if m == best_model else CYAN for m in model_names]
bars = ax3.barh(model_names, rmse_vals, color=colors_bar, edgecolor="#333355", height=0.55)
for bar, val in zip(bars, rmse_vals):
    ax3.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
             f"${val:.1f}K", va="center", color=WHITE, fontsize=9)
ax3.set_title("Model RMSE Comparison", color=WHITE, fontsize=13)
ax3.set_xlabel("RMSE ($K)", color=WHITE)
ax3.tick_params(colors=WHITE)
ax3.spines[:].set_color("#333355")
ax3.invert_yaxis()

# ── Panel 4: Feature Importance ──
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_facecolor(PANEL)
fi_colors = [GOLD if i == 0 else PURPLE for i in range(len(fi_series))]
ax4.barh(fi_series.index[::-1], fi_series.values[::-1],
         color=fi_colors[::-1], edgecolor="#333355", height=0.6)
ax4.set_title(f"Top Feature Importances ({best_tree_name})", color=WHITE, fontsize=13)
ax4.set_xlabel("Importance", color=WHITE)
ax4.tick_params(colors=WHITE)
ax4.spines[:].set_color("#333355")

plt.savefig("outputs/sales_forecasting_results.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("\nPlot saved → sales_forecasting_results.png")

# ─────────────────────────────────────────────
# 8.  BONUS – Seasonal Decomposition Plot
# ─────────────────────────────────────────────
fig2, axes = plt.subplots(4, 1, figsize=(18, 12), facecolor=BG)
fig2.suptitle("Seasonal Decomposition of Weekly Sales", fontsize=18,
              fontweight="bold", color=WHITE, y=0.99)

components = [
    (df["Weekly_Sales"] / 1e3, "Observed",  CYAN),
    (trend_comp / 1e3,          "Trend",     GOLD),
    (seasonal_comp / 1e3,       "Seasonal",  GREEN),
    (residual / 1e3,            "Residual",  RED),
]

for ax, (data, label, col) in zip(axes, components):
    ax.set_facecolor(PANEL)
    ax.plot(data.index, data.values, color=col, lw=1.5)
    ax.fill_between(data.index, data.values, alpha=0.1, color=col)
    ax.set_ylabel(label + "\n($K)", color=WHITE, fontsize=10)
    ax.tick_params(colors=WHITE)
    ax.spines[:].set_color("#333355")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("outputs/seasonal_decomposition.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Seasonal decomposition plot saved → seasonal_decomposition.png")

# ─────────────────────────────────────────────
# 9.  SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Dataset        : {n_weeks} weeks of simulated Walmart-style data")
print(f"  Features built : {len(FEATURES)} (lags, rolling stats, calendar, exogenous)")
print(f"  Models tested  : {len(models)} regression models + rolling-avg baseline")
print(f"  Best model     : {best_model}")
print(f"    MAE  = ${results[best_model]['MAE']:>10,.0f}")
print(f"    RMSE = ${results[best_model]['RMSE']:>10,.0f}")
print(f"    R²   = {results[best_model]['R2']:.4f}")
print("\n  Bonus completed:")
print("    ✓ Rolling averages used as features + baseline")
print("    ✓ Seasonal decomposition (trend / seasonal / residual)")
print("    ✓ Gradient Boosting used (XGBoost-equivalent built-in sklearn)")
print("    ✓ Time-aware train/test split (no data leakage)")
print("=" * 60)