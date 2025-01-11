import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import sqlite3
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple
from monte_carlo import MonteCarloProFirm

class StatisticalMetrics(NamedTuple):
    """統計的評価の結果を保持する構造体"""
    mean_return: float
    confidence_lower: float
    confidence_upper: float
    p_value: float
    p_value_factor: float
    t_statistic: float

@dataclass
class TradingMetrics:
    """トレードメトリクスを保持するデータクラス"""
    optimal_f: float
    max_loss: float
    avg_loss: float
    loss_std: float
    expected_max_loss: float
    recommended_lots: float
    conservative_lots: float
    pure_optimal_lots: float
    expected_value: float

class PositionSizeOptimizer:
    # クラス定数
    CONFIDENCE_LEVEL = 0.95
    BASE_SAFETY_FACTOR = 0.95
    MAX_RECOMMENDED_LOTS = 100
    DEFAULT_OPTIMAL_F = 0.02
    SIGMA_THRESHOLD = 3  # 3シグマルール用の定数

    def __init__(self, db_path: str = "trades.db", initial_capital: float = 18000):
        self.db_path = db_path
        self.initial_capital = initial_capital

    def fetch_trade_data(self) -> pd.DataFrame:
        """データベースから全トレードデータを取得"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM trades ORDER BY entry_time"
            df = pd.read_sql_query(query, conn)
            # ISO8601形式を指定して日時を解析
            df['entry_time'] = pd.to_datetime(df['entry_time'], format='ISO8601')
            df['exit_time'] = pd.to_datetime(df['exit_time'], format='ISO8601')
            
            self._print_trade_data_summary(df)
            return df

    def _print_trade_data_summary(self, df: pd.DataFrame) -> None:
        """取引データの基本統計量を出力"""
        print("\n=== 取引データの基本統計 ===")
        print(f"総取引数: {len(df)}")
        print("\nPnL統計:")
        print(df['pnl'].describe())
        print("\nサイズ統計:")
        print(df['size'].describe())

    def evaluate_statistical_significance(self, returns: pd.Series) -> StatisticalMetrics:
        """リターンデータの統計的評価を行う"""
        n_samples = len(returns)
        mean_return = returns.mean()
        std_return = returns.std()
        
        # 信頼区間の計算
        degrees_of_freedom = n_samples - 1
        t_value = stats.t.ppf((1 + self.CONFIDENCE_LEVEL) / 2, degrees_of_freedom)
        margin_of_error = t_value * (std_return / np.sqrt(n_samples))
        
        confidence_lower = mean_return - margin_of_error
        confidence_upper = mean_return + margin_of_error
        
        # t検定
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # p値に基づく信頼性係数
        p_value_factor = (
            1.0 if p_value < 0.01
            else 0.8 if p_value < 0.05
            else 0.6
        )
        
        self._print_statistical_evaluation(
            n_samples, mean_return, std_return, 
            confidence_lower, confidence_upper, 
            t_stat, p_value, p_value_factor
        )
        
        return StatisticalMetrics(
            mean_return=mean_return,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            p_value=p_value,
            p_value_factor=p_value_factor,
            t_statistic=t_stat
        )

    def _print_statistical_evaluation(self, n_samples: int, mean_return: float, 
                                    std_return: float, confidence_lower: float, 
                                    confidence_upper: float, t_stat: float, 
                                    p_value: float, p_value_factor: float) -> None:
        """統計評価結果の出力"""
        print("\n=== 統計的信頼性評価 ===")
        print(f"サンプル数: {n_samples}")
        print(f"平均リターン: ${mean_return:.2f}")
        print(f"標準偏差: ${std_return:.2f}")
        print(f"95%信頼区間: ${confidence_lower:.2f} ～ ${confidence_upper:.2f}")
        print(f"t統計量: {t_stat:.4f}")
        print(f"p値: {p_value:.4f}")
        print(f"p値に基づく信頼性係数: {p_value_factor:.2f}")
        
    def calculate_pure_optimal_lots(self, optimal_f: float, max_loss: float) -> float:
        """純粋なOptimal fに基づくロット数の計算"""
        pure_lots = (optimal_f * self.initial_capital) / max_loss
        return min(pure_lots, self.MAX_RECOMMENDED_LOTS)

    def calculate_optimal_f(self, returns: np.ndarray) -> float:
        """Optimal fの計算"""
        def negative_growth_rate(f: float) -> float:
            f_value = float(f[0]) if isinstance(f, np.ndarray) else float(f)
            
            max_loss = abs(min(returns))
            if f_value * max_loss >= 1:
                return np.inf
                        
            growth_terms = 1 + f_value * returns
            growth_rates = np.log(growth_terms)
            valid_rates = growth_rates[np.isfinite(growth_rates)]
            
            if len(valid_rates) == 0:
                return np.inf
                        
            mean_growth = np.mean(valid_rates)
            self._print_growth_rate_calculation(f_value, max_loss, returns, growth_terms, valid_rates, mean_growth)
            return -mean_growth

        max_loss = abs(min(returns))
        kelly_fraction = 0.95/max_loss
        upper_bound = max(0.02, min(0.5, kelly_fraction))
        lower_bound = min(0.01, upper_bound/2)

        self._print_optimal_f_calculation_start(returns, kelly_fraction, lower_bound, upper_bound)
        
        result = minimize(
            negative_growth_rate, 
            x0=[min(0.1, upper_bound/2)],
            bounds=[(lower_bound, upper_bound)],
            method='L-BFGS-B'
        )

        self._print_optimization_result(result)
        
        return result.x[0] if result.success else self.DEFAULT_OPTIMAL_F

    def _print_growth_rate_calculation(self, f_value: float, max_loss: float, 
                                     returns: np.ndarray, growth_terms: np.ndarray, 
                                     valid_rates: np.ndarray, mean_growth: float) -> None:
        """成長率計算のデバッグ情報出力"""
        print(f"\n=== 成長率計算のデバッグ ===")
        print(f"f値: {f_value:.4f}")
        print(f"1ロットあたりの最大損失時の資金減少率: {(f_value * max_loss):.4f}")
        print(f"リターンの範囲: {returns.min():.4f} から {returns.max():.4f}")
        print(f"growth_terms範囲: {growth_terms.min():.4f} から {growth_terms.max():.4f}")
        print(f"有効な成長率の数: {len(valid_rates)} / {len(returns)}")
        if len(valid_rates) > 0:
            print(f"成長率の範囲: {valid_rates.min():.4f} から {valid_rates.max():.4f}")
        print(f"平均成長率: {mean_growth:.4f}")

    def _print_optimal_f_calculation_start(self, returns: np.ndarray, kelly_fraction: float,
                                         lower_bound: float, upper_bound: float) -> None:
        """Optimal f計算開始時の情報出力"""
        print(f"\n=== Optimal f 計算開始 ===")
        print(f"1ロットあたりのリターンデータ数: {len(returns)}")
        print(f"リターン統計: min={returns.min():.4f}, max={returns.max():.4f}, mean={returns.mean():.4f}")
        print(f"Kelly Fraction: {kelly_fraction:.4f}")
        print(f"最適化範囲: {lower_bound:.4f} から {upper_bound:.4f}")

    def _print_optimization_result(self, result: minimize) -> None:
        """最適化結果の出力"""
        print("\n=== 最適化結果 ===")
        print(f"成功: {result.success}")
        print(f"最適化メッセージ: {result.message}")
        print(f"反復回数: {result.nit}")

    def calculate_lot_sizes(self, stats_metrics: StatisticalMetrics, 
                          optimal_f: float, max_loss: float) -> Tuple[float, float]:
        """統計的信頼性を考慮したロット数の計算"""
        conservative_optimal_f = optimal_f * (stats_metrics.confidence_lower / stats_metrics.mean_return)
        adjusted_safety_factor = self.BASE_SAFETY_FACTOR * stats_metrics.p_value_factor

        base_recommended_lots = (conservative_optimal_f * self.initial_capital) / max_loss
        base_conservative_lots = (conservative_optimal_f * self.initial_capital * 
                                adjusted_safety_factor) / max_loss

        if stats_metrics.p_value > 0.05:
            print("警告: 取引戦略の有効性が統計的に十分に証明されていません")

        self._print_lot_calculation(
            conservative_optimal_f, adjusted_safety_factor,
            base_recommended_lots, base_conservative_lots
        )

        return (
            min(base_recommended_lots, self.MAX_RECOMMENDED_LOTS),
            min(base_conservative_lots, self.MAX_RECOMMENDED_LOTS)
        )

    def _print_lot_calculation(self, conservative_optimal_f: float, 
                             adjusted_safety_factor: float,
                             base_recommended_lots: float, 
                             base_conservative_lots: float) -> None:
        """ロット数計算結果の出力"""
        print("\n=== 保守的ロット数計算 ===")
        print(f"調整後Optimal f: {conservative_optimal_f:.4f}")
        print(f"安全係数: {adjusted_safety_factor:.2f}")
        print(f"基本推奨ロット数（制限前）: {base_recommended_lots:.2f}")
        print(f"基本保守的ロット数（制限前）: {base_conservative_lots:.2f}")

    def calculate_metrics(self, df: pd.DataFrame) -> Optional[TradingMetrics]:
        """トレードメトリクスの計算（1ロットあたりに正規化）"""
        try:
            print("\n=== メトリクス計算開始 ===")
            print(f"入力データ件数: {len(df)}")
            
            # 1ロットあたりのリターンを計算
            returns = df['pnl'] / df['size']
            
            # 統計的評価
            stats_metrics = self.evaluate_statistical_significance(returns)
            
            # 異常値や無効な値をフィルタリング
            returns = returns[np.isfinite(returns)]
            print(f"\n有効なリターンデータ数: {len(returns)}")
            
            # 異常値の除外（3シグマルール）
            threshold = returns.mean() + self.SIGMA_THRESHOLD * returns.std()
            filtered_returns = returns[np.abs(returns) < threshold]
            
            self._print_returns_statistics(returns, filtered_returns)
            
            # Optimal fの計算
            optimal_f = self.calculate_optimal_f(filtered_returns.values)
            print(f"\n計算されたOptimal f: {optimal_f:.4f}")
            
            # 損失の統計
            losses = filtered_returns[filtered_returns < 0]
            print(f"\n損失トレード数: {len(losses)}")
            
            if len(losses) == 0:
                print("損失データがありません。デフォルト値を使用します。")
                return self._create_default_metrics()

            # 損失統計の計算
            loss_stats = self._calculate_loss_statistics(losses)
            
            # 統計的調整ありのロット数の計算
            recommended_lots, conservative_lots = self.calculate_lot_sizes(
                stats_metrics, optimal_f, loss_stats['max_loss']
            )

            # 純粋なOptimal fベースのロット数を計算
            pure_optimal_lots = self.calculate_pure_optimal_lots(
                optimal_f, loss_stats['max_loss']
            )

            return TradingMetrics(
                optimal_f=optimal_f,
                max_loss=loss_stats['max_loss'],
                avg_loss=loss_stats['avg_loss'],
                loss_std=loss_stats['loss_std'],
                expected_max_loss=loss_stats['expected_max_loss'],
                recommended_lots=recommended_lots,
                conservative_lots=conservative_lots,
                pure_optimal_lots=pure_optimal_lots,
                expected_value=stats_metrics.confidence_lower
            )
                
        except Exception as e:
            print(f"メトリクス計算中にエラーが発生: {str(e)}")
            raise

    def _print_returns_statistics(self, returns: pd.Series, 
                                filtered_returns: pd.Series) -> None:
        """リターン統計の出力"""
        print(f"3シグマルール適用後のデータ数: {len(filtered_returns)}")
        print("フィルタリング後の1ロットあたりのリターン統計:")
        print(filtered_returns.describe())

    def _calculate_loss_statistics(self, losses: pd.Series) -> dict:
            """損失統計の計算"""
            max_loss = abs(losses.min())
            avg_loss = abs(losses.mean())
            loss_std = losses.std()
            expected_max_loss = avg_loss + (2 * loss_std)

            print(f"1ロットあたりの最大損失: ${max_loss:.2f}")
            print(f"1ロットあたりの平均損失: ${avg_loss:.2f}")
            print(f"1ロットあたりの損失標準偏差: ${loss_std:.2f}")
            print(f"1ロットあたりの予想最大損失: ${expected_max_loss:.2f}")

            return {
                'max_loss': max_loss,
                'avg_loss': avg_loss,
                'loss_std': loss_std,
                'expected_max_loss': expected_max_loss
            }

    def _create_default_metrics(self) -> TradingMetrics:
        """デフォルトメトリクスの作成"""
        return TradingMetrics(
            optimal_f=self.DEFAULT_OPTIMAL_F,
            max_loss=0,
            avg_loss=0,
            loss_std=0,
            expected_max_loss=0,
            recommended_lots=0,
            conservative_lots=0,
            pure_optimal_lots=0,
            expected_value=0
        )

    def generate_report(self) -> Optional[TradingMetrics]:
        """分析レポートの生成"""
        df = self.fetch_trade_data()
        return self.calculate_metrics(df)

class InitialCapitalCalculator:
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        
    def calculate_recent_pnl(self, start_date: str = "2025-01-08") -> float:
        """2025/1/8以降のトレード収支を計算"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT SUM(pnl) as total_pnl
            FROM trades
            WHERE entry_time >= ?
            """
            cursor = conn.execute(query, (start_date,))
            result = cursor.fetchone()
            return float(result[0]) if result[0] is not None else 0.0

def calculate_initial_capital(monte_carlo_result: Tuple[int, float], start_date: str = "2025-01-08") -> float:
    """
    モンテカルロシミュレーション結果と最近のトレード収支から初期資金を計算
    
    Parameters:
    monte_carlo_result: Tuple[int, float] - (最適ロット数, その確率)
    start_date: str - 収支計算開始日 (YYYY-MM-DD形式)
    
    Returns:
    float: 計算された初期資金
    """
    best_lot, probability = monte_carlo_result
    
    # 指定された計算式に基づく初期資金の計算
    # 2000 × (50 × モンテカルロ確率 + 200)
    base_capital = 2000 / (50 * probability + 200) * 2000
    
    # 2025/1/8以降のトレード収支を取得
    calculator = InitialCapitalCalculator()
    recent_pnl = calculator.calculate_recent_pnl(start_date)
    
    # 最終的な初期資金を計算
    total_capital = base_capital + recent_pnl
    
    return total_capital


def main():
    # MonteCarloProFirmのインスタンス化と実行
    simulator = MonteCarloProFirm(
        db_path="trades.db",
        initial_capital=50_000.0,
        profit_target=3_000.0,
        max_drawdown=2_000.0,
        num_days=20,
        num_simulations=10_000
    )
    
    # 最適なロット数と確率を取得
    best_lot, best_prob = simulator.find_best_lot_in_range(lot_min=5, lot_max=10)
    print(f"\n最適ロット数: {best_lot}")
    print(f"成功確率: {best_prob:.4f}")
    
    # 初期資金を計算
    initial_capital = calculate_initial_capital((best_lot, best_prob))
    print(f"\n計算された初期資金: ${initial_capital:,.2f}")
    
    # PositionSizeOptimizerの実行
    optimizer = PositionSizeOptimizer(initial_capital=initial_capital)
    metrics = optimizer.generate_report()
    
    if metrics is not None:
            print("\n=== 最終分析結果 ===")
            print(f"口座資金: ${initial_capital:.2f}")
            print(f"1枚あたりの95%区間下限期待値: ${metrics.expected_value:.2f}")
            print(f"Optimal f: {metrics.optimal_f:.4f}")
            print(f"最大損失: ${metrics.max_loss:.2f}")
            print(f"平均損失: ${metrics.avg_loss:.2f}")
            print(f"損失の標準偏差: ${metrics.loss_std:.2f}")
            print(f"予想最大損失: ${metrics.expected_max_loss:.2f}")
            print(f"純粋なOptimal fベースのロット数: {metrics.pure_optimal_lots:.1f}")
            print(f"統計調整ありの推奨ロット数: {metrics.recommended_lots:.1f}")
            print(f"統計調整ありの保守的ロット数: {metrics.conservative_lots:.1f}")
    else:
        print("分析に必要な十分なデータがありません")
            


if __name__ == "__main__":
    main()
