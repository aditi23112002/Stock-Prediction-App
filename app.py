import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from keras.models import load_model
import datetime

# Load the trained model and scaler
@st.cache_resource
def load_trained_model():
    model_path = "stock_model.h5"
    scaler_path = "scaler.pkl"

    # Check if the files exist before loading
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: {model_path}")
        st.stop()
    
    if not os.path.exists(scaler_path):
        st.error(f"⚠️ Scaler file not found: {scaler_path}")
        st.stop()
    
    return load_model(model_path), joblib.load(scaler_path)

# Function to predict future stock prices
def predict_future_prices(model, scaler, days=30):
    last_known_price = np.random.uniform(5000, 6000)  # Replace with real last stock price
    predictions = []

    for _ in range(days):
        pred_price = last_known_price + np.random.uniform(-50, 50)  # Simulate small changes
        predictions.append(pred_price)
        last_known_price = pred_price

    return np.array(predictions)

# App Title
st.title("📈 Stock Price Prediction")

# User Input Section
st.sidebar.header("⚙️ User Input")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., ^NSEI, GOOGL)", "^NSEI")
days_to_predict = st.sidebar.slider("Days to Predict", 1, 60, 30)

# Load the model
model, scaler = load_trained_model()

# Generate predictions
future_prices = predict_future_prices(model, scaler, days_to_predict)
dates = pd.date_range(start=datetime.date.today(), periods=days_to_predict)

# 🔹 Normal Graph: Actual vs Predicted Prices
st.subheader(f"📊 {stock_ticker} Stock Price Prediction")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates, future_prices, label="Predicted Prices", linestyle="dashed", color="red")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
st.pyplot(fig)

# 🔹 Future Price Range Display
st.subheader("📉 Future Price Prediction Range")
min_price = np.min(future_prices)
max_price = np.max(future_prices)
st.success(f"✅ **Expected price range in the next {days_to_predict} days:** ₹{min_price:.2f} - ₹{max_price:.2f}")

# 🔹 Time Series Graph for Trend Analysis
st.subheader("📅 Time Series Analysis")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(dates, future_prices, label="Predicted Trend", color="purple")
ax2.fill_between(dates, min_price, max_price, color="purple", alpha=0.2)
ax2.set_xlabel("Date")
ax2.set_ylabel("Stock Price")
ax2.legend()
st.pyplot(fig2)

