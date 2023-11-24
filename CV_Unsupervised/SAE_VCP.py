import yfinance as yf
import talib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch historical stock prices using yfinance
def fetch_stock_prices(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close'].values

# Function to calculate technical indicators using TA-Lib
def calculate_technical_indicators(prices):
    # Example: Bollinger Bands
    upper_band, _, lower_band = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # Return Bollinger Bands as features
    return upper_band, lower_band

# Fetch historical stock prices for Apple Inc. (AAPL)
ticker = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'
prices = fetch_stock_prices(ticker, start_date, end_date)

# Calculate technical indicators
upper_band, lower_band = calculate_technical_indicators(prices)

# Combine features (closing prices and Bollinger Bands)
features = np.vstack([prices, upper_band, lower_band])

# Normalize the data
normalized_data = (features - np.mean(features)) / np.std(features)

# Convert the data to PyTorch tensors
data_tensor = torch.FloatTensor(normalized_data).unsqueeze(0)  # Add a batch dimension

# Stacked Autoencoder model
class StackedAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(StackedAutoencoder, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            self.encoder.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.decoder.append(nn.Linear(hidden_sizes[i + 1], hidden_sizes[i]))

    def forward(self, x):
        encoded = x
        for layer in self.encoder:
            encoded = torch.relu(layer(encoded))

        decoded = encoded
        for layer in self.decoder:
            decoded = torch.relu(layer(decoded))

        return encoded, decoded


# Model, optimizer, and loss
sae = StackedAutoencoder(input_size=3, hidden_sizes=[32, 16])
optimizer = optim.Adam(sae.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training (unchanged)
for epoch in range(100):
    optimizer.zero_grad()
    encoded, decoded = sae(data_tensor)
    loss = criterion(decoded, data_tensor)
    loss.backward()
    optimizer.step()

# Extract features using the trained encoder
with torch.no_grad():
    encoded_data, _ = sae.encoder(data_tensor)

# Visualize the original and reconstructed data
plt.plot(normalized_data[0], label='Closing Prices')
plt.plot(normalized_data[1], label='Upper Bollinger Band')
plt.plot(normalized_data[2], label='Lower Bollinger Band')
plt.plot(decoded.numpy().squeeze(), label='Reconstructed Data')
plt.title('Original vs Reconstructed Data')
plt.legend()
plt.show()

# Use the encoded data to identify periods of volatility contraction
threshold = 0.1  # Adjust based on experimentation
volatility_contraction_periods = np.where(encoded_data.squeeze().numpy() < threshold)[0]

# Visualize the identified volatility contraction periods
plt.plot(normalized_data[0], label='Closing Prices')
plt.scatter(volatility_contraction_periods, normalized_data[0, volatility_contraction_periods], color='red', label='Volatility Contraction')
plt.title('Volatility Contraction Pattern')
plt.legend()
plt.show()

=================================
=================================

# Function to calculate candlestick patterns using TA-Lib
def calculate_candlestick_patterns(prices):
    # Example: Three Inside Up pattern
    pattern_result = CDL3INSIDE(prices['Open'], prices['High'], prices['Low'], prices['Close'])
    
    # Return candlestick pattern as features
    return pattern_result
    
    
# Calculate candlestick patterns
candlestick_patterns = calculate_candlestick_patterns(prices)

# Normalize the data
normalized_data = (candlestick_patterns - np.mean(candlestick_patterns)) / np.std(candlestick_patterns)
