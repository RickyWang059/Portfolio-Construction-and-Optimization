#Written by Chi-Jui, Wang

import numpy as np
import pandas as pd
import scipy.optimize as sco

# 1. Read the CSV file and format the data
file_path = "Portfolio optimized.csv"
data = pd.read_csv(file_path)

# Set the date column as index and format the dates
data.set_index(pd.to_datetime(data['Unnamed: 0'], format='%d-%b-%Y'), inplace=True)
data.drop(columns=['Unnamed: 0'], inplace=True)

# Calculate the monthly returns for each stock
monthly_returns = data.pct_change().dropna()

# Define the function to calculate portfolio's annualized return, standard deviation, and Sharpe Ratio
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate_annual):
    portfolio_return = np.dot(weights, mean_returns) * 12  # Annualized return
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)  # Annualized standard deviation
    sharpe_ratio = (portfolio_return - risk_free_rate_annual) / portfolio_std_dev  # Annualized Sharpe Ratio
    return portfolio_return, portfolio_std_dev, sharpe_ratio

# Set the risk-free rate (annualized)
risk_free_rate_annual = 0.0247  

# Calculate the covariance matrix and mean of monthly returns
mean_returns = monthly_returns.mean()
cov_matrix = monthly_returns.cov()

# ---- 2. Calculate Sharpe Ratio when weights are equally distributed (10%) ----
print("\n---- Equal weight (10% each) case ----")

# Set equal weights for each stock (10%)
initial_weights = np.array([0.1] * len(monthly_returns.columns))

# Calculate portfolio performance (equal weights)
initial_return, initial_std_dev, initial_sharpe = portfolio_performance(initial_weights, mean_returns, cov_matrix, risk_free_rate_annual)

# Print results
print("Annualized return for portfolio with equal weights: ", initial_return)
print("Annualized standard deviation for portfolio with equal weights: ", initial_std_dev)
print("Sharpe Ratio (equal weights): ", initial_sharpe)

# ---- 3. Case when shorting is not allowed ----
print("\n---- No shorting allowed ----")

# Set weight bounds (0 to 1), no shorting allowed
bounds_no_shorting = tuple((0, 1) for asset in range(len(monthly_returns.columns)))

# Define the negative Sharpe Ratio as the objective function for minimization
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate_annual):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate_annual)[2]

# Define the constraint that the sum of weights must equal 1
def constraint_sum_of_weights(weights):
    return np.sum(weights) - 1

# Set the constraints
constraints = ({'type': 'eq', 'fun': constraint_sum_of_weights})

# Use minimize to optimize
optimized_result_no_shorting = sco.minimize(neg_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate_annual),
                                            method='SLSQP', bounds=bounds_no_shorting, constraints=constraints)

# Optimized weights (no shorting allowed)
optimized_weights_no_shorting = optimized_result_no_shorting.x

# Portfolio performance with optimized weights (no shorting allowed)
optimized_return_no_shorting, optimized_std_dev_no_shorting, optimized_sharpe_no_shorting = portfolio_performance(
    optimized_weights_no_shorting, mean_returns, cov_matrix, risk_free_rate_annual)

# Print results
print("Optimized weights (no shorting): ", optimized_weights_no_shorting)
print("Annualized return (no shorting): ", optimized_return_no_shorting)
print("Annualized standard deviation (no shorting): ", optimized_std_dev_no_shorting)
print("Optimized Sharpe Ratio (no shorting): ", optimized_sharpe_no_shorting)

# ---- 4. Case when shorting is allowed ----
print("\n---- Shorting allowed ----")

# Set unrestricted weight bounds, shorting allowed
bounds_shorting = tuple((None, None) for asset in range(len(monthly_returns.columns)))

# Use minimize to optimize
optimized_result_shorting = sco.minimize(neg_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate_annual),
                                         method='SLSQP', bounds=bounds_shorting, constraints=constraints)

# Optimized weights (shorting allowed)
optimized_weights_shorting = optimized_result_shorting.x

# Portfolio performance with optimized weights (shorting allowed)
optimized_return_shorting, optimized_std_dev_shorting, optimized_sharpe_shorting = portfolio_performance(
    optimized_weights_shorting, mean_returns, cov_matrix, risk_free_rate_annual)

# Print results
print("Optimized weights (shorting allowed): ", optimized_weights_shorting)
print("Annualized return (shorting allowed): ", optimized_return_shorting)
print("Annualized standard deviation (shorting allowed): ", optimized_std_dev_shorting)
print("Optimized Sharpe Ratio (shorting allowed): ", optimized_sharpe_shorting)
