import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("📊 Sales Forecasting")

# ─────────────────────────────
# Upload Data
# ─────────────────────────────
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure required columns
    df = df[['Date', 'Weekly_Sales']]

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    st.subheader("📌 Raw Data")
    st.write(df.head())

    # ─────────────────────────────
    # Feature Engineering
    # ─────────────────────────────
    df['Month'] = df.index.month
    df['Week']  = df.index.isocalendar().week.astype(int)

    lags = [1, 2, 4, 8]

    for lag in lags:
        df[f'Lag_{lag}'] = df['Weekly_Sales'].shift(lag)

    df['Rolling_Mean'] = df['Weekly_Sales'].shift(1).rolling(4).mean()

    df.dropna(inplace=True)

    # ─────────────────────────────
    # Train Model
    # ─────────────────────────────
    TARGET = "Weekly_Sales"
    FEATURES = [col for col in df.columns if col != TARGET]

    X = df[FEATURES]
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor()
    model.fit(X_scaled, y)

    st.success("✅ Model Trained Successfully")

    # ─────────────────────────────
    # Forecast Future
    # ─────────────────────────────
    st.subheader("🔮 Forecast")

    n_weeks = st.slider("Number of weeks to forecast", 1, 20, 5)

    # Last data
    last_row = df.iloc[-1:].copy()

    # Store last values to use in lag
    history = list(df['Weekly_Sales'].values[-8:])

    future_preds = []

    for i in range(n_weeks):
        row = last_row.copy()

        # ✅ Use history instead of Lag mistake
        for lag in lags:
            row[f'Lag_{lag}'] = history[-lag]

        row['Rolling_Mean'] = np.mean(history[-4:])

        # Update time
        next_date = row.index[0] + pd.Timedelta(weeks=1)
        row.index = [next_date]

        row['Month'] = next_date.month
        row['Week']  = next_date.isocalendar().week

        # Predict
        X_future = scaler.transform(row[FEATURES])
        pred = model.predict(X_future)[0]

        future_preds.append(pred)

        # Update history
        history.append(pred)

        # Update last_row
        last_row = row.copy()
        last_row['Weekly_Sales'] = pred

    # ─────────────────────────────
    # Visualization
    # ─────────────────────────────
    st.subheader("📈 Forecast Results")

    future_dates = pd.date_range(start=df.index[-1], periods=n_weeks+1, freq='W')[1:]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['Weekly_Sales'], label="Historical")
    ax.plot(future_dates, future_preds, label="Forecast", linestyle='--')

    ax.set_title("Sales Forecast")
    ax.legend()

    st.pyplot(fig)

    # ─────────────────────────────
    # Show Predictions Table
    # ─────────────────────────────
    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": future_preds
    })

    st.subheader("📊 Predictions Table")
    st.write(pred_df)
