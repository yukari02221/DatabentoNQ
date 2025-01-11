import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MonteCarloProFirm:
    def __init__(
        self,
        db_path: str = "trades.db",
        initial_capital: float = 50_000.0,
        profit_target: float = 3_000.0,
        max_drawdown: float = 2_000.0,
        num_days: int = 20,
        num_simulations: int = 10_000
    ):
        """
        :param db_path: トレード履歴が格納されたSQLite DBファイルのパス
        :param initial_capital: シミュレーションでの口座初期資金 (例: 50,000ドル)
        :param profit_target: 合格条件となる利益幅 (例: +3,000ドル)
        :param max_drawdown: トレーリングされる最大許容DD (例: 2,000ドル)
        :param num_days: 合格判定を行う期間 (営業日数) (例: 20日)
        :param num_simulations: モンテカルロ試行回数
        """
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.profit_target = profit_target
        self.max_drawdown = max_drawdown
        self.num_days = num_days
        self.num_simulations = num_simulations

        # 保存用データ
        self.df_trades = None  # fetch_trades()で読み込んだDataFrameをキャッシュ
        self.equity_curves = []  # プロット用に各ロットでのサンプル曲線を保存

    def fetch_trades(self) -> pd.DataFrame:
        """
        データベースから全トレードを取得し、1lotあたりのPnL 'pnl_per_lot' を計算して返す。
        （ロットごとのスケーリングは後で実施）
        """
        if self.df_trades is not None:
            return self.df_trades  # 既に読み込んでいれば再利用

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_time", conn)

        # size=0 のレコードが無い前提でpnl_per_lotを計算
        df["pnl_per_lot"] = df["pnl"] / df["size"]

        self.df_trades = df
        return df

    def run_simulation_for_lot(self, lot: int, plot_simulations: bool = False, max_plot_lines: int = 50) -> float:
        """
        指定されたロット数 `lot` で Monte Carlo法を実施し、
        「num_days(=20)営業日以内に +profit_target(=3,000)ドル達成 かつ
         一度も(ピーク - max_drawdown)を下回らない」確率を返す。

        1日の最大利益は 1499 ドルまでにクリップする。

        :param lot: シミュレーションに用いるロット数 (例: 5~10)
        :param plot_simulations: Trueの場合、シミュレーション中の損益曲線をプロット用に記録
        :param max_plot_lines: 描画するシミュレーションの最大本数
        :return: 成功確率(0.0～1.0)
        """
        df = self.fetch_trades()
        # ロット数を掛けたPnL配列を準備
        # daily_pnl 取得後に 1日最大1499 にクリップする
        pnl_array = (df["pnl_per_lot"] * lot).values

        success_count = 0

        # （今回のロットについてのみ）描画用のエクイティカーブを一時格納
        # 後で self.equity_curves にまとめて格納
        temp_equity_curves = []

        for sim_idx in range(self.num_simulations):
            capital = self.initial_capital
            peak_capital = capital
            is_success = False

            daily_capitals = [capital]

            for day in range(self.num_days):
                # ランダムサンプリング
                raw_pnl = np.random.choice(pnl_array)
                # 1日あたりの最大利益は 1499 ドルまで
                daily_pnl = min(raw_pnl, 1499.0)

                capital += daily_pnl
                if capital > peak_capital:
                    peak_capital = capital

                dd_limit = peak_capital - self.max_drawdown

                # ドローダウン割れ
                if capital < dd_limit:
                    daily_capitals.append(capital)
                    break

                # +3,000 到達で成功
                if (capital - self.initial_capital) >= self.profit_target:
                    is_success = True
                    daily_capitals.append(capital)
                    break

                daily_capitals.append(capital)

            if is_success:
                success_count += 1

            # 描画用に格納
            if plot_simulations and sim_idx < max_plot_lines:
                temp_equity_curves.append(daily_capitals)

        success_prob = success_count / self.num_simulations

        # plot_simulations=True なら、後でまとめてプロットするために保存
        if plot_simulations:
            # ロット別にわかるよう、タプル (lot, [curve1, curve2, ...]) として保存する
            self.equity_curves.append((lot, temp_equity_curves))

        return success_prob

    def find_best_lot_in_range(self, lot_min: int = 5, lot_max: int = 10) -> tuple:
        """
        lot_min～lot_maxの範囲で、それぞれ MonteCarlo シミュレーションし、
        最も成功確率が高いロット数と、その成功確率を返す。

        :return: (best_lot, best_prob)
        """
        best_lot = None
        best_prob = -1.0

        for lot in range(lot_min, lot_max + 1):
            prob = self.run_simulation_for_lot(lot=lot, plot_simulations=False, max_plot_lines=0)
            if prob > best_prob:
                best_prob = prob
                best_lot = lot

        return best_lot, best_prob

    def plot_all_equity_curves(self):
        """
        run_simulation_for_lot() 実行時に「plot_simulations=True」で蓄積した
        全ロット分のサンプル損益曲線を一括でプロットする。
        """
        plt.figure(figsize=(10, 6))
        color_map = plt.cm.get_cmap('tab10')  # 10色程度のカラーマップ

        # self.equity_curves = [(lot, [curve1, curve2, ...]), (lot, [...]), ...]
        for idx, (lot, curves) in enumerate(self.equity_curves):
            color = color_map(idx % 10)
            for curve in curves:
                plt.plot(curve, alpha=0.3, color=color)
            # 凡例用に1本だけラベル付きで描画
            if len(curves) > 0:
                plt.plot(curves[0], alpha=0.8, color=color, label=f"Lot {lot} (sample)")

        plt.title("Monte Carlo Equity Curves (Various Lots)")
        plt.xlabel("Day")
        plt.ylabel("Capital")
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    simulator = MonteCarloProFirm(
        db_path="trades.db",
        initial_capital=50_000.0,
        profit_target=3_000.0,
        max_drawdown=2_000.0,
        num_days=20,
        num_simulations=10_000  # 必要に応じて調整
    )

    # 1) まずは 5～10ロットについて、どのロットが最も成功率が高いか調べる
    best_lot, best_prob = simulator.find_best_lot_in_range(lot_min=5, lot_max=10)
    print(f"【ロット範囲 5～10 の中で最も成功確率が高いロット】")
    print(f"  Lot = {best_lot},  成功確率 = {best_prob*100:.2f}%")

    # 2) ベストだったロットを含む複数ロットで、実際にプロットを見たい場合
    #    ここでは例として 5, best_lot, 10 の3種類を描画してみる
    lots_to_plot = [best_lot]
    for lot in lots_to_plot:
        # それぞれ plot_simulations=True で実行すると、エクイティカーブを self.equity_curves に保存する
        # max_plot_lines=50 などで表示本数を調整
        simulator.run_simulation_for_lot(lot=lot, plot_simulations=True, max_plot_lines=10_000)

    # 3) まとめて描画
    simulator.plot_all_equity_curves()


if __name__ == "__main__":
    main()
