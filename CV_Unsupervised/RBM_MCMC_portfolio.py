import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
import emcee

# Fetch historical stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to preprocess data for RBM
def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to train RBM on historical stock data
def train_rbm(data, n_components=10, learning_rate=0.01, n_iter=20):
    rbm = BernoulliRBM(n_components=n_components, learning_rate=learning_rate, n_iter=n_iter)
    rbm.fit(data)
    return rbm

# Function to sample from RBM to generate synthetic portfolio weights
def sample_portfolio_weights(rbm, num_samples):
    weights = np.zeros((num_samples, rbm.components_.shape[0]))
    for i in range(num_samples):
        hidden_states = np.random.rand(rbm.components_.shape[0])
        visible_states = rbm.inverse_transform(hidden_states)
        weights[i] = visible_states
    return weights

# Define a simple objective function for portfolio returns and volatility
def objective(weights, returns, covariance_matrix, risk_aversion):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return -(portfolio_return - risk_aversion * portfolio_volatility)

# Fetch stock data and calculate daily returns
ticker_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
start_date = '2021-01-01'
end_date = '2022-01-01'

returns_data = pd.DataFrame()
for ticker in ticker_list:
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    returns_data[ticker] = stock_data['Adj Close'].pct_change().dropna()

returns_data = returns_data.dropna()

# Preprocess data for RBM
scaled_data, scaler = preprocess_data(returns_data.values)

# Train RBM
rbm = train_rbm(scaled_data)

# Sample synthetic portfolio weights using RBM
num_samples = 1000
synthetic_weights = sample_portfolio_weights(rbm, num_samples)

# Calculate expected returns and covariance matrix
expected_returns = returns_data.mean().values
covariance_matrix = returns_data.cov().values

# Define risk aversion parameter for the objective function
risk_aversion = 2.0

# Define initial weights for the MCMC sampler
initial_weights = np.ones(len(ticker_list)) / len(ticker_list)

# Perform MCMC sampling to optimize portfolio weights
sampler = emcee.EnsembleSampler(num_samples, len(ticker_list), objective,
                                args=[expected_returns, covariance_matrix, risk_aversion])

# Burn-in phase
n_burn = 100
_, _, _ = sampler.run_mcmc(initial_weights.reshape((1, -1)), n_burn, progress=True)

# Actual sampling
n_samples = 1000
sampler.reset()
sampler.run_mcmc(initial_weights.reshape((1, -1)), n_samples, progress=True)

# Extract samples from the MCMC chain
samples = sampler.get_chain(flat=True)

# Plot the MCMC samples
plt.figure(figsize=(12, 6))
for i in range(len(ticker_list)):
    plt.hist(samples[:, i], bins=30, alpha=0.5, label=ticker_list[i])

plt.xlabel('Portfolio Weight')
plt.ylabel('Frequency')
plt.legend()
plt.title('MCMC Portfolio Optimization Weights')
plt.show()
