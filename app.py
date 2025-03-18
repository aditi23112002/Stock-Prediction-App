import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Streamlit Page Setup
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction Dashboard")

# User Input: Ticker
ticker = st.text_input("Enter Stock Ticker (e.g., ^NSEI for NIFTY, GOOG,GODFRYPHLP.NS)", "^NSEI").upper()
model_file = f"{ticker}_lstm_model.h5"
scaler_file = f"{ticker}_scaler.pkl"

# Load Model and Scaler
if os.path.exists(model_file) and os.path.exists(scaler_file):
    model = load_model(model_file)
    scaler = joblib.load(scaler_file)
    st.success(f"✅ Model for {ticker} loaded successfully!")
else:
    st.error(f"❌ No trained model found for {ticker}! Please train it using model.py.")
    st.stop()

# Fetch stock data
def get_stock_data(ticker, start='2010-01-01', end='2025-03-10'):
    data = yf.download(ticker, start=start, end=end)
    return data[['Close']]

# Prepare data for prediction
def prepare_prediction_data(data, scaler, time_steps=60):
    scaled_data = scaler.transform(data)
    X_test = [scaled_data[-time_steps:]]
    return np.array(X_test)

# Predict future prices
def predict_future_prices(model, scaler, last_60_days, days=30):
    predictions = []
    last_input = last_60_days.copy()

    for _ in range(days):
        X_test = np.array([last_input])
        pred = model.predict(X_test)
        predictions.append(pred[0][0])
        last_input = np.vstack([last_input[1:], pred])
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Fetch Data and Display Chart
data = get_stock_data(ticker)
st.line_chart(data['Close'], use_container_width=True)

# Prepare last 60 days of data
time_steps = 60
last_60_days = data['Close'].values[-time_steps:].reshape(-1, 1)
last_60_scaled = scaler.transform(last_60_days)

# User Input: Future Prediction Days
future_days = st.slider("Select Future Days to Predict", 1, 30, 10)
future_predictions = predict_future_prices(model, scaler, last_60_scaled, future_days)

# Display Predictions
future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1)[1:]
prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})

st.write("### Future Predictions 📅")
st.dataframe(prediction_df)
st.line_chart(prediction_df.set_index("Date"), use_container_width=True)
