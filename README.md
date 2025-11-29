#STOCKPRICEPREDICTION

 📊 Stock Price Prediction using Long Short Term Memory(LSTM 📈)<br>
This project leverages a Long Short-Term Memory (LSTM) neural network to predict Apple Inc.’s closing stock prices based on historical data.<br> 🧠 It utilizes Yahoo Finance API for real-time data fetching and MinMaxScaler for data normalization to improve accuracy.<br> 🔍 The model is trained on past 60-day stock prices to forecast future trends with high precision.<br> 📅 Visualization is done using Matplotlib to compare actual vs predicted values for performance evaluation.<br> 📉 This project demonstrates skills in deep learning, data preprocessing, and time-series forecasting using Python and Keras. 🚀

import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use ('fivethirtyeight')
df = yf.download('AAPL', start='2012-01-01', end='2019-12-17')
df.shape
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset) * .8)
training_data_len
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data
