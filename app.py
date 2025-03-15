import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import streamlit as st
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# 🔹 Load Model & Scaler
model = load_model("stock_lstm_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 🔹 Streamlit UI
st.title("📈 Stock Price Prediction")
st.sidebar.header("⚙️ User Input")

# 🔹 User Input for Stock Ticker
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., ^NSEI, GOOGL)", "^NSEI").strip().upper()
future_days = st.sidebar.slider("Days to Predict", min_value=1, max_value=60, value=30)

# 🔹 Fetch Stock Data
try:
    df = yf.download(ticker, start="2010-01-01", end="2024-01-01")
    if df.empty:
        st.error("⚠️ Invalid Ticker or No Data Found! Try a different one.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error fetching data: {e}")
    st.stop()

df = df[['Close']]

# 🔹 Prepare Data for Prediction
scaled_data = scaler.transform(df['Close'].values.reshape(-1, 1))
last_sequence = scaled_data[-60:]

# 🔹 Function to Predict Future Prices
def predict_future(model, last_sequence, scaler, n_future):
    future_predictions = []
    current_sequence = last_sequence.copy().reshape(1, last_sequence.shape[0], 1)

    for _ in range(n_future):
        current_prediction = model.predict(current_sequence)
        future_predictions.append(current_prediction[0][0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = current_prediction[0][0]

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 🔹 Predict Future Prices
future_predictions = predict_future(model, last_sequence, scaler, future_days)

# 🔹 Create Date Range for Future Predictions
future_dates = pd.date_range(df.index[-1], periods=future_days+1)[1:]

# 🔹 Plotly Graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Actual Prices", line=dict(color="cyan")))
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode="lines", name="Predicted Prices", line=dict(dash="dash", color="red")))
fig.update_layout(title=f"{ticker} Stock Price Prediction", xaxis_title="Date", yaxis_title="Stock Price", template="plotly_dark")

st.plotly_chart(fig)

st.success(f"✅ Prediction Complete for {future_days} days!")
