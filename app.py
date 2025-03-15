import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Load Model & Scaler
@st.cache_resource
def load_trained_model():
    return load_model("stock_lstm_model.h5"), joblib.load("scaler.pkl")

model, scaler = load_trained_model()

# UI Design
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; color: pink;'>📈 Stock Price Prediction</h1>", unsafe_allow_html=True)

st.sidebar.header("⚙️ User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., ^NSEI, GOOGL)", "^NSEI").upper()
future_days = st.sidebar.slider("Days to Predict", 1, 60, 30)

# Fetch Data
df = yf.download(ticker, period="5y", interval="1d")
if df.empty:
    st.error(f"❌ Failed to fetch {ticker}. Try another.")
    st.stop()

# Preprocess Data
scaled_data = scaler.transform(df['Close'].values.reshape(-1, 1))
last_sequence = scaled_data[-60:]

# Prediction Function
def predict_future(model, last_sequence, scaler, future_days):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        pred = model.predict(current_sequence.reshape(1, -1, 1))[0, 0]
        future_predictions.append(pred)
        current_sequence = np.append(current_sequence[1:], pred).reshape(-1, 1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Generate Predictions
future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, future_days+1)]
predictions = predict_future(model, last_sequence, scaler, future_days)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Actual Prices", line=dict(color='cyan')))
fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', name="Predicted Prices", line=dict(color='red', dash="dash")))

fig.update_layout(title=f"{ticker} Stock Price Prediction", xaxis_title="Date", yaxis_title="Stock Price", template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

st.success(f"✅ Prediction Complete for {future_days} days!")
