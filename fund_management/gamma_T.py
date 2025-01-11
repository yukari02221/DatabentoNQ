import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

def calculate_put_gamma(S, K, T, r, sigma):
    """Calculate gamma for European put option."""
    if T <= 0:
        # 満期を過ぎていればガンマは 0 として扱う（あるいは計算しない）
        return 0.0
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def calculate_put_price(S, K, T, r, sigma):
    """Calculate price for European put option."""
    # 満期を過ぎていればペイオフのみ
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return put_price

# -------------------------------
# 1) パラメータ設定
# -------------------------------
K = 100          # Strike price
r = 0.05         # Annual interest rate
sigma = 0.05      # Volatility (10%)
days = 30        # Number of days until expiry (約 1 ヶ月 = 30 営業日とみなす)
T_initial = days / 252.0  # ~0.119 年 (252営業日換算)

np.random.seed(42)  # For reproducibility
daily_returns = np.random.normal(0, sigma/np.sqrt(252), days)

# -------------------------------
# 2) 価格のシミュレーション (prices)
# -------------------------------
initial_price = K
prices = [initial_price]
for ret in daily_returns:
    prices.append(prices[-1] * (1 + ret))
prices = np.array(prices)

# -------------------------------
# 3) 残存期間リスト (time_to_expiry)
#    0日目: T_initial, 30日目: 0
# -------------------------------
time_to_expiry = [T_initial - i*(1/252) for i in range(days+1)]

# -------------------------------
# 4) 初期オプション価格 & 投資額
# -------------------------------
initial_option_price = calculate_put_price(prices[0], K, time_to_expiry[0], r, sigma)
initial_investment = initial_option_price

# -------------------------------
# 5) 日々のガンマ・オプション価格・リターン算出
# -------------------------------
gammas = []
option_prices = []
for p, t in zip(prices, time_to_expiry):
    gammas.append(calculate_put_gamma(p, K, t, r, sigma))
    option_prices.append(calculate_put_price(p, K, t, r, sigma))

returns = [
    (opt_price - initial_investment) / initial_investment * 100
    for opt_price in option_prices
]

# -------------------------------
# 6) 可視化
# -------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[1, 1, 1])
fig.suptitle('Put Option Analysis\n(Strike = 100, σ = 10%, ~1-month expiry)', fontsize=12)

# Gamma
ax1.plot(range(days+1), gammas, 'b-', label='Gamma')
ax1.set_ylabel('Gamma')
ax1.set_title('Daily Gamma')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Asset Price
ax2.plot(range(days+1), prices, 'g-', label='Asset Price')
ax2.axhline(y=K, color='r', linestyle='--', alpha=0.5, label='Strike Price')
ax2.set_ylabel('Price')
ax2.set_title('Asset Price Movement')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Option Returns
ax3.plot(range(days+1), returns, 'purple', label='Option Return')
ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
ax3.set_xlabel('Days')
ax3.set_ylabel('Return (%)')
ax3.set_title('Option Returns (%)')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# 7) 統計量出力
# -------------------------------
print("\nSummary Statistics:")
print(f"Average Gamma: {np.mean(gammas):.6f}")
print(f"Max Gamma: {np.max(gammas):.6f}")
print(f"Initial Asset Price: {prices[0]:.2f}")
print(f"Final Asset Price: {prices[-1]:.2f}")
print(f"Initial Option Price: {option_prices[0]:.2f}")
print(f"Final Option Price: {option_prices[-1]:.2f}")
print(f"Total Return: {returns[-1]:.2f}%")
print(f"Maximum Return: {max(returns):.2f}%")
print(f"Minimum Return: {min(returns):.2f}%")
