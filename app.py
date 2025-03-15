import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load trained model
MODEL_PATH = "stock_lstm_model.h5"

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to prepare input data
def prepare_input_data(data, time_steps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X = [scaled_data[-time_steps:]]
    return np.array(X), scaler

# Function to predict future prices
def predict_future(model, last_sequence, scaler, n_future):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(n_future):
        current_prediction = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        future_predictions.append(current_prediction[0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = current_prediction

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions

# Generate future dates
def generate_future_dates(last_date, num_days):
    future_dates = []
    current_date = last_date
    for _ in range(num_days):
        current_date += timedelta(days=1)
        while current_date.weekday() > 4:  # Skip weekends
            current_date += timedelta(days=1)
        future_dates.append(current_date)
    return pd.DatetimeIndex(future_dates)

# Streamlit UI
st.title("📈 Stock Price Prediction App")
st.sidebar.header("🔍 Select Stock & Dates")

# User Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, ^NSEI)", "^NSEI")
start_date = st.sidebar.date_input("Select Start Date", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("Select End Date", datetime(2025, 3, 1))
future_days = st.sidebar.slider("Select Future Prediction Days", 1, 60, 30)

# Load Data
if st.sidebar.button("📊 Load Data & Predict"):
    with st.spinner("Fetching stock data..."):
        data = fetch_stock_data(ticker, start_date, end_date)

    if data.empty:
        st.error("⚠️ No stock data found. Please check the ticker symbol.")
    else:
        st.success("✅ Data loaded successfully!")

        # Display historical data
        st.subheader(f"📉 {ticker} Stock Price History")
        st.line_chart(data['Close'])

        # Prepare Data for Prediction
        time_steps = 60
        last_sequence, scaler = prepare_input_data(data['Close'], time_steps)

        # Load Model & Predict
        st.subheader(f"📊 Predicting Future {future_days} Days...")
        model = load_model(MODEL_PATH)
        future_predictions = predict_future(model, last_sequence, scaler, future_days)

        # Generate Future Dates
        future_dates = generate_future_dates(data.index[-1], future_days)

        # Plot Predictions
        st.subheader("📈 Future Stock Price Prediction")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(future_dates, future_predictions.flatten(), marker='o', linestyle='--', color='purple', label="Future Predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price (USD)")
        ax.set_title(f"{ticker} Stock Price Prediction")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Display Future Predictions
        st.subheader("📅 Future Price Predictions")
        future_data = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})
        st.dataframe(future_data)

        # Download Predictions
        csv_data = future_data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv_data, "future_predictions.csv", "text/csv", key="download-csv")
