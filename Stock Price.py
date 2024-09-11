import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco

# 定義要下載的股票代碼
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "AVGO", "TSLA"]

# 抓取數據，設置時間範圍和每月的區隔
data = yf.download(tickers, start="2019-09-01", end="2024-08-31", interval="1mo", auto_adjust=False)

# 只提取收盤價，並按照日期降序排列
close_prices = data['Close'].sort_index(ascending=True)

# 計算每支股票的月回報率
monthly_returns = close_prices.pct_change().dropna()

# 計算標準差 (Standard Deviation) 和方差 (Variance)
std_dev = monthly_returns.std()
variance = monthly_returns.var()

# 打印基本數據
print("每支股票的月回報率:\n", monthly_returns.mean())
print("每支股票的標準差:\n", std_dev)
print("每支股票的方差:\n", variance)

# 定義投資組合的年化回報率、標準差和 Sharpe Ratio 計算函數
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_returns) * 12  # 年化回報率
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)  # 年化標準差
    sharpe_ratio = (portfolio_return - risk_free_rate*12 ) / portfolio_std_dev  # 年化 Sharpe Ratio
    return portfolio_return, portfolio_std_dev, sharpe_ratio

# 設置無風險利率
risk_free_rate = 0.0247 

# 計算協方差矩陣和月回報率
mean_returns = monthly_returns.mean()
cov_matrix = monthly_returns.cov()

# ---- 在權重均為10%的情況下計算 Sharpe Ratio ----
print("\n---- 權重均為10%的情況 ----")

# 設置每支股票權重均為10%
initial_weights = np.array([0.1] * len(tickers))

# 計算投資組合表現（權重均為10%）
initial_return, initial_std_dev, initial_sharpe = portfolio_performance(initial_weights, mean_returns, cov_matrix, risk_free_rate)

# 打印結果
print("權重均為10%的投資組合的年化回報率: ", initial_return)
print("權重均為10%的投資組合的年化標準差: ", initial_std_dev)
print("Sharpe Ratio (權重均為10%): ", initial_sharpe)

# ---- 不允許做空的情況 ----
print("\n---- 不允許做空的情況 ----")

# 設置權重範圍 (0 到 1)，不允許做空
bounds_no_shorting = tuple((0, 1) for asset in range(len(tickers)))

# 定義負的 Sharpe Ratio 作為目標函數進行最小化
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# 定義權重加起來為1的約束
def constraint_sum_of_weights(weights):
    return np.sum(weights) - 1

# 設置約束條件
constraints = ({'type': 'eq', 'fun': constraint_sum_of_weights})

# 使用 minimize 進行優化
optimized_result_no_shorting = sco.minimize(neg_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                                            method='SLSQP', bounds=bounds_no_shorting, constraints=constraints)

# 最優化權重（不允許做空）
optimized_weights_no_shorting = optimized_result_no_shorting.x

# 最優投資組合的表現（不允許做空）
optimized_return_no_shorting, optimized_std_dev_no_shorting, optimized_sharpe_no_shorting = portfolio_performance(
    optimized_weights_no_shorting, mean_returns, cov_matrix, risk_free_rate)

# 打印結果
print("最優權重 (不允許做空): ", optimized_weights_no_shorting)
print("投資組合的年化回報率 (不允許做空): ", optimized_return_no_shorting)
print("投資組合的年化標準差 (不允許做空): ", optimized_std_dev_no_shorting)
print("最優 Sharpe Ratio (不允許做空): ", optimized_sharpe_no_shorting)

# ---- 允許做空的情況 ----
print("\n---- 允許做空的情況 ----")

# 設置無限制的權重範圍，允許做空
bounds_shorting = tuple((None, None) for asset in range(len(tickers)))

# 使用 minimize 進行優化
optimized_result_shorting = sco.minimize(neg_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                                         method='SLSQP', bounds=bounds_shorting, constraints=constraints)

# 最優化權重（允許做空）
optimized_weights_shorting = optimized_result_shorting.x

# 最優投資組合的表現（允許做空）
optimized_return_shorting, optimized_std_dev_shorting, optimized_sharpe_shorting = portfolio_performance(
    optimized_weights_shorting, mean_returns, cov_matrix, risk_free_rate)

# 打印結果
print("最優權重 (允許做空): ", optimized_weights_shorting)
print("投資組合的年化回報率 (允許做空): ", optimized_return_shorting)
print("投資組合的年化標準差 (允許做空): ", optimized_std_dev_shorting)
print("最優 Sharpe Ratio (允許做空): ", optimized_sharpe_shorting)
