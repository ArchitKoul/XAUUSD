
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import ta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(layout="wide")
st.title("📈 XAUUSD Strategy Dashboard")

# Load data
df = pd.read_csv("XAU_USD Historical Data.csv")
df.columns = [col.strip().replace(" ", "_").replace(".", "") for col in df.columns]
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date')
df['Price'] = df['Price'].str.replace(',', '').astype(float)

# Feature Engineering
df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
df['Lag1'] = df['Log_Returns'].shift(1)
df['MA5'] = df['Price'].rolling(window=5).mean()
df['MA20'] = df['Price'].rolling(window=20).mean()
df['Volatility'] = df['Log_Returns'].rolling(window=10).std()
df['Direction'] = (df['Log_Returns'] > 0).astype(int)

# Technical Indicators
df['RSI'] = ta.momentum.RSIIndicator(close=df['Price']).rsi()
df['MACD'] = ta.trend.MACD(close=df['Price']).macd()
df['ADX'] = ta.trend.ADXIndicator(high=df['High'].str.replace(',', '').astype(float),
                                  low=df['Low'].str.replace(',', '').astype(float),
                                  close=df['Price']).adx()
df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'].str.replace(',', '').astype(float),
                                           low=df['Low'].str.replace(',', '').astype(float),
                                           close=df['Price']).average_true_range()

# Drop missing values
features = ['Lag1', 'MA5', 'MA20', 'Volatility', 'RSI', 'MACD', 'ADX', 'ATR']
df_ml = df[features + ['Direction']].dropna()

# Model Training
X = df_ml[features]
y = df_ml['Direction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
df_ml['Predicted'] = model.predict(X)

# Strategy Simulation
df_ml['Strategy_Returns'] = df_ml['Predicted'].shift(1) * df['Log_Returns'].loc[df_ml.index]
df_ml['Market_Returns'] = df['Log_Returns'].loc[df_ml.index]
df_ml['Cumulative_Strategy'] = (1 + df_ml['Strategy_Returns']).cumprod()
df_ml['Cumulative_Market'] = (1 + df_ml['Market_Returns']).cumprod()
df_ml['Drawdown'] = df_ml['Cumulative_Strategy'] / df_ml['Cumulative_Strategy'].cummax() - 1

# Sharpe Ratio & Drawdown
strategy_returns = df_ml['Strategy_Returns'].dropna()
sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
max_drawdown = df_ml['Drawdown'].min()

# Dashboard Layout
st.subheader("📊 Strategy vs Market Performance")
fig1 = px.line(df_ml, y=['Cumulative_Strategy', 'Cumulative_Market'], title="Cumulative Returns")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📉 Drawdown Over Time")
fig2 = px.line(df_ml, y='Drawdown', title="Strategy Drawdown")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📈 Risk Metrics")
st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
st.metric("Max Drawdown", f"{max_drawdown:.2%}")
