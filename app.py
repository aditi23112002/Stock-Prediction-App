import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# 🔹 Set Page Configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# 🎯 Load LSTM Model with Caching to Optimize Performance
@st.cache_resource
def load_stock_model():
    return load_model("stock_lstm_model.h5")

model = load_stock_model()

# 🎯 Function to Fetch Stock Data (Cached)
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# 🎯 Function to Predict Future Prices
def predict_future(model, last_sequence, scaler, n_future):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(n_future):
        current_sequence = current_sequence.reshape(1, current_sequence.shape[0], 1)
        current_prediction = model.predict(current_sequence, verbose=0)
        future_predictions.append(current_prediction[0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = current_prediction

    return scaler.inverse_transform(np.array(future_predictions))

# 🎯 Streamlit Sidebar for User Inputs
st.sidebar.header("🔹 Settings")
ticker = st.sidebar.text_input("📌 Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("📅 Start Date", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("📅 End Date", datetime.today())
future_days = st.sidebar.slider("⏳ Days to Predict", 10, 60, 30)

# 🎯 Prediction Button
if st.sidebar.button("🔮 Predict"):
    st.info("⏳ Fetching Stock Data...")

    # 🔹 Fetch Stock Data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    if stock_data.empty:
        st.error("❌ Invalid Ticker Symbol or No Data Found!")
    else:
        st.success("✅ Data Loaded Successfully!")

        # 🔹 Extract Close Prices & Normalize Data
        close_prices = stock_data["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # 🔹 Prepare Last Sequence
        time_steps = 60  # Last 60 days for prediction
        last_sequence = scaled_data[-time_steps:]

        # 🔹 Predict Future Stock Prices
        st.info("📊 Predicting Future Prices...")
        future_predictions = predict_future(model, last_sequence, scaler, future_days)

        # 🔹 Generate Future Dates
        future_dates = [end_date + timedelta(days=i) for i in range(1, future_days + 1)]

        # 🔹 Create DataFrame for Visualization
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})

        # 🔹 Interactive Stock Price Graph using Plotly
        st.subheader(f"📈 {ticker} Stock Price Prediction for Next {future_days} Days")

        fig = go.Figure()

        # 🔹 Add Historical Data
        fig.add_trace(go.Scatter(
            x=stock_data.index, 
            y=stock_data["Close"], 
            mode="lines", 
            name="📉 Historical Prices",
            line=dict(color="blue")
        ))

        # 🔹 Add Future Predictions
        fig.add_trace(go.Scatter(
            x=future_df["Date"], 
            y=future_df["Predicted Price"], 
            mode="lines+markers", 
            name="🔮 Predicted Prices",
            line=dict(color="red", dash="dot")
        ))

        # 🔹 Customize Graph Layout
        fig.update_layout(
            title=f"{ticker} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Stock Price (USD)",
            legend=dict(x=0, y=1),
            template="plotly_dark",
            height=500
        )

        # 🔹 Display Graph in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # 🔹 Show Future Predictions Data
        st.subheader("📋 Future Predictions Data")
        st.dataframe(future_df)

# 🎯 Footer
st.markdown("---")
st.markdown("📌 Developed by **Aditi Garg** | 🚀 Powered by LSTM & Streamlit")

