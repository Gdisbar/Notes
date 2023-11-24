import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import talib

# Fetch historical stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to add additional features to sequences using TA-Lib
def add_ta_features(data):
    # Calculate technical analysis indicators
    data['SMA'] = talib.SMA(data['Close'], timeperiod=10)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], _, _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Drop rows with NaN values due to indicator calculation
    data = data.dropna()
    
    return data[['Close', 'Volume', 'SMA', 'RSI', 'MACD']].values

# Function to create temporal sequences for RBM training
def create_sequences(data, sequence_length):
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length, 0]  # Target is the next day's closing price
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

# Set parameters
ticker = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'
sequence_length = 10

# Fetch stock data and add TA features
stock_data = fetch_stock_data(ticker, start_date, end_date)
sequences = add_ta_features(stock_data)

# Normalize the data
scaler = MinMaxScaler()
sequences_scaled = scaler.fit_transform(sequences)

# Convert data to PyTorch tensors
sequences_tensor = torch.FloatTensor(sequences_scaled).unsqueeze(2)
targets_tensor = torch.FloatTensor(stock_data['Close'].values[sequence_length:])

# Define the Gaussian RBM model using PyTorch
class GaussianRBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(GaussianRBM, self).__init__()
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.1)
        self.visible_bias = nn.Parameter(torch.zeros(visible_units))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_units))

    def forward(self, v):
        h_mean = torch.sigmoid(torch.matmul(v, self.W) + self.hidden_bias)
        h_sample = torch.bernoulli(h_mean)
        v_mean = torch.matmul(h_sample, self.W.t()) + self.visible_bias
        return v_mean, h_mean

# Instantiate and train the Gaussian RBM
visible_units = sequences.shape[1]
hidden_units = 5
rbm = GaussianRBM(visible_units, hidden_units)

num_epochs = 100
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    v_mean, h_mean = rbm(sequences_tensor)
    loss = criterion(v_mean, sequences_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate predictions using the trained RBM
predictions, _ = rbm(sequences_tensor)

# Inverse transform the predictions to original scale
predictions_inv = scaler.inverse_transform(predictions.squeeze().detach().numpy())

# Plot the results
import matplotlib.pyplot as plt

plt.plot(stock_data.index[sequence_length:], stock_data['Close'][sequence_length:], label='Actual Prices', linewidth=2)
plt.plot(stock_data.index[sequence_length:], predictions_inv[:, -1], label='Predicted Prices', linestyle='dashed', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


=================================
=================================

import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import talib
import arch

# Fetch historical stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to add additional features to sequences using TA-Lib
def add_ta_features(data):
    # Calculate technical analysis indicators
    data['SMA'] = talib.SMA(data['Close'], timeperiod=10)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], _, _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Drop rows with NaN values due to indicator calculation
    data = data.dropna()
    
    return data[['Close', 'Volume', 'SMA', 'RSI', 'MACD']].values

# Function to create temporal sequences for RBM training
def create_sequences(data, sequence_length):
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length, 0]  # Target is the next day's closing price
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

# Set parameters
ticker = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'
sequence_length = 10

# Fetch stock data and add TA features
stock_data = fetch_stock_data(ticker, start_date, end_date)
sequences = add_ta_features(stock_data)

# Normalize the data
scaler = MinMaxScaler()
sequences_scaled = scaler.fit_transform(sequences)

# Convert data to PyTorch tensors
sequences_tensor = torch.FloatTensor(sequences_scaled).unsqueeze(2)
targets_tensor = torch.FloatTensor(stock_data['Close'].values[sequence_length:])

# Train GARCH model to predict volatility
garch_model = arch.arch_model(stock_data['Close'], vol='Garch', p=1, q=1)
garch_results = garch_model.fit(disp='off')

# Predict volatility using trained GARCH model
volatility_predictions = garch_results.conditional_volatility[-len(targets_tensor):].values

# Combine RBM and GARCH predictions
combined_predictions = targets_tensor + torch.FloatTensor(volatility_predictions)

# Inverse transform the predictions to original scale
combined_predictions_inv = scaler.inverse_transform(combined_predictions.unsqueeze(1).numpy())

# Plot the results
import matplotlib.pyplot as plt

plt.plot(stock_data.index[sequence_length:], stock_data['Close'][sequence_length:], label='Actual Prices', linewidth=2)
plt.plot(stock_data.index[sequence_length:], combined_predictions_inv, label='Combined Predictions', linestyle='dashed', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

