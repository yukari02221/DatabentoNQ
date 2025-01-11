import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_gamma(S, K, T, r, sigma):
    """
    Calculate gamma for European options using Black-Scholes model
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    N_prime_d1 = norm.pdf(d1)
    gamma = N_prime_d1 / (S * sigma * np.sqrt(T))
    return gamma

def plot_gamma_comparison():
    """異なるボラティリティと期間でのガンマを比較プロット"""
    
    # 基本パラメータ
    K = 100  # 権利行使価格
    r = 0.05  # 金利
    
    # 原資産価格の範囲
    S = np.linspace(60, 140, 200)
    
    # パラメータ設定
    sigmas = [0.1, 0.2, 0.3]  # ボラティリティ
    times = [1/12, 3/12, 6/12, 1.0]  # 期間（年）
    time_labels = ['1M', '3M', '6M', '1Y']  # 期間のラベル
    
    # 色とラインスタイルの設定
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # ボラティリティごとの色
    line_styles = ['-', '--', ':', '-.']  # 期間ごとの線種
    
    plt.figure(figsize=(15, 10))
    
    # 各ボラティリティと期間の組み合わせでプロット
    for i, (sigma, color) in enumerate(zip(sigmas, colors)):
        for T, style, time_label in zip(times, line_styles, time_labels):
            gamma = calculate_gamma(S, K, T, r, sigma)
            plt.plot(S, gamma, 
                    label=f'σ = {sigma}, {time_label}',
                    color=color,
                    linestyle=style,
                    linewidth=2)
    
    # グラフの装飾
    plt.axvline(x=K, color='gray', linestyle='--', alpha=0.5, label='Strike Price')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Underlying Price ($)', fontsize=12)
    plt.ylabel('Gamma', fontsize=12)
    plt.title('Option Gamma: Comparison of Different Volatilities and Time to Expiry', 
             fontsize=14, pad=15)
    
    # 凡例の設定
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.15, 1))
    
    # 軸の設定
    plt.ylim(0, max(calculate_gamma(S, K, 1/12, r, 0.1)) * 1.1)
    plt.grid(True, alpha=0.3)
    
    # グラフ全体のレイアウト調整
    plt.tight_layout()
    
    return plt.gcf()

if __name__ == "__main__":
    # グラフ生成と保存
    fig = plot_gamma_comparison()
    fig.savefig('gamma_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    
    # グラフ表示
    plt.show()