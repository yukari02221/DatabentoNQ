from logging_utils import CustomLogger
import asyncio
import pandas as pd
from databento_dbn import SymbolMappingMsg, OHLCVMsg, SystemMsg
from databento_client import DatabentoClient, DatabentoCommunicationError
from audio_manager import AudioManager
import os
from typing import Optional, Dict
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
import threading
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import sys
import numpy as np

# -------------------------------------------------
# (A) 非同期でデータを取得し、価格データとリターンを保持するだけのクラス
# -------------------------------------------------
class CMELiveDataFetcher:
    def __init__(self, api_key: Optional[str] = None, debug_mode: bool = False):
        self.logger = CustomLogger("CMELiveDataFetcher")
        
        self.api_key = api_key or os.environ.get('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided.")
        
        self.databento_client = DatabentoClient(api_key)
        self.audio_manager = AudioManager(debug_mode)      
        self.instrument_id_map = {}
        # price_data[symbol] の中に times / returns などを格納
        self.price_data: Dict[str, Dict] = defaultdict(lambda: {
            'times': [],
            'prices': [],
            'base_price': None,
            'returns': [],
            'market_open_time': None,
            # 60区間分の価格データを保持するキュー
            'price_queue': [],
            'queue_initialized': False  # キューの初期化フラグ
        })

        self.ny_tz = pytz.timezone('America/New_York')
        self.logger.log('INFO', 'init', f"Timezone set to: {self.ny_tz}")    
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.logger.log('INFO', 'init', "Running in debug mode")     
        # スレッドセーフにするためのロック
        self.lock = threading.Lock()
        # キューの最大サイズ
        self.QUEUE_MAX_SIZE = 60
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.databento_client.stop()
        await self.reset_price_data()

    async def reset_price_data(self):
        with self.lock:
            for symbol in self.price_data:
                self.price_data[symbol].update({
                    'times': [],
                    'prices': [],
                    'base_price': None,
                    'returns': [],
                    'market_open_time': None,
                    'price_queue': [],
                    'queue_initialized': False
                })
        self.logger.log('INFO', 'reset_price_data', "Price data has been reset for new session")

    def get_next_market_open(self) -> datetime:
        """
        次の平日9:30に相当する日付に対して、
        30分前(9:00)を返すようにする。
        """
        now = datetime.now(self.ny_tz)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        # すでに9:30を過ぎていれば翌営業日へ
        if now >= market_open:
            market_open += timedelta(days=1)
        # 土日をスキップして次の月曜日へ
        while market_open.weekday() in [5, 6]:  # 5:土曜, 6:日曜
            days_until_monday = (7 - market_open.weekday()) % 7
            market_open += timedelta(days=days_until_monday)
        # 9:00(9:30の30分前)を開始時刻とする
        return market_open - timedelta(minutes=30)

    def get_market_close(self, target_date: datetime) -> datetime:
        """
        9:00に対する当日の市場クローズ(16:00)を返す。
        """
        return target_date.replace(hour=16, minute=0, second=0, microsecond=0)

    def process_bar(self, record) -> None:
        asyncio.create_task(self._process_bar_async(record))
        

    async def _process_bar_async(self, record) -> None:
            """
            受信したレコードを非同期で処理する。
            ここで base_price の設定タイミングを 9:30 以降に限定する修正を行う。
            """
            try:
                if isinstance(record, SymbolMappingMsg):
                    symbol = record.stype_out_symbol
                    if record.instrument_id not in self.instrument_id_map:
                        self.instrument_id_map[record.instrument_id] = symbol
                        self.logger.log('INFO', 'process_bar',
                            f"Symbol Mapping: ID={record.instrument_id}, "
                            f"Out={symbol}, Time={record.pretty_ts_event}"
                        )

                elif isinstance(record, SystemMsg):
                    self.logger.log('INFO','process_bar', f"System Message: {record.msg}")

                elif isinstance(record, OHLCVMsg):
                    with self.lock:
                        mapped_symbol = self.instrument_id_map.get(record.instrument_id, "Unknown")
                        bar_data = {
                            'timestamp': pd.Timestamp(record.ts_event),
                            'symbol': mapped_symbol,
                            'open':   record.open / 1e9,
                            'high':   record.high / 1e9,
                            'low':    record.low / 1e9,
                            'close':  record.close / 1e9,
                            'volume': record.volume
                        }
                        current_time = (pd.Timestamp(record.ts_event)
                                        .tz_localize("UTC")
                                        .tz_convert(self.ny_tz))

                        is_after_930 = (current_time.hour > 9) or (current_time.hour == 9 and current_time.minute >= 30)
                        
                        if is_after_930:
                            # 価格キューの更新
                            if not self.price_data[mapped_symbol]['queue_initialized']:
                                # 最初の60区間分のデータを収集
                                self.price_data[mapped_symbol]['price_queue'].append(bar_data['close'])
                                if len(self.price_data[mapped_symbol]['price_queue']) >= self.QUEUE_MAX_SIZE:
                                    self.price_data[mapped_symbol]['queue_initialized'] = True
                                    self.logger.log('INFO', 'process_bar',
                                        f"Price queue initialized for {mapped_symbol} with {self.QUEUE_MAX_SIZE} periods")
                            else:
                                # キューが初期化済みの場合、FIFO方式で更新
                                self.price_data[mapped_symbol]['price_queue'].pop(0)  # 最も古いデータを削除
                                self.price_data[mapped_symbol]['price_queue'].append(bar_data['close'])  # 新しいデータを追加

                            # 9:30以降の場合のみデータを記録
                            if self.price_data[mapped_symbol]['base_price'] is None:
                                self.price_data[mapped_symbol]['base_price'] = bar_data['close']
                                
                            self.price_data[mapped_symbol]['times'].append(current_time)
                            self.price_data[mapped_symbol]['prices'].append(bar_data['close'])
                            
                            base = self.price_data[mapped_symbol]['base_price']
                            ret = (bar_data['close'] / base - 1) * 100
                            self.price_data[mapped_symbol]['returns'].append(ret)

                            self.logger.log('INFO','process_bar',
                                f"OHLCV Data: {bar_data}, Return={ret:.2f}%, Queue size={len(self.price_data[mapped_symbol]['price_queue'])}")
                        else:
                            self.logger.log('INFO','process_bar',
                                f"9:30以前のデータはスキップ: {bar_data}")

                else:
                    self.logger.log('INFO','process_bar', f"New Record Type: {type(record)}")

            except Exception as e:
                self.logger.log('ERROR','process_bar', f"Error: {str(e)}")

    def get_price_queue_copy(self, symbol: str):
        """
        スレッド安全に price_queue のコピーを取得するためのメソッド
        """
        with self.lock:
            return list(self.price_data[symbol]['price_queue'])

    async def run_session(self, market_close: datetime):
        """
        1度のセッションで、9:00〜16:00までデータを購読して待機する処理を行う。
        """
        try:
            # セッション開始前にデータ初期化
            await self.reset_price_data()
            await self.databento_client.setup_subscription()
            self.databento_client.start(self.process_bar)
            self.logger.log('INFO','run_session', "Streaming started")

            now = datetime.now(self.ny_tz)
            wait_seconds = (market_close - now).total_seconds()
            if wait_seconds > 0:
                await self.databento_client.wait_for_close(timeout=wait_seconds)

        except Exception as e:
            self.logger.log('ERROR','run_session', f"Error: {str(e)}")
            raise
        finally:
            self.databento_client.stop()
            self.logger.log('INFO','run_session', "Streaming stopped")
            await self.reset_price_data()

    async def run_async(self):
        """
        メインループ。
        - debug_mode=True の場合、24時間走らせるイメージ
        - debug_mode=False の場合、本来の営業日サイクルで 9:00〜16:00 を繰り返す
        """
        asyncio.create_task(self.audio_manager.check_and_play_audio())
        while True:
            try:
                if self.debug_mode:
                    # デバッグ中は「今から24時間後まで走らせる」イメージ
                    await self.run_session(datetime.now(self.ny_tz) + timedelta(hours=24))
                else:
                    next_open = self.get_next_market_open()  # 例: 明日9:00
                    mclose = self.get_market_close(next_open) # 例: 当日の16:00
                    now = datetime.now(self.ny_tz)
                    wait_secs = (next_open - now).total_seconds()
                    if wait_secs > 0:
                        await asyncio.sleep(wait_secs)
                    await self.run_session(mclose)

            except Exception as e:
                self.logger.log('ERROR','run_async', f"Main loop error: {str(e)}")
                # 5分待って再トライ
                await asyncio.sleep(300)

    def get_data_copy(self, symbol: str):
        """
        スレッド安全に times / returns のコピーを取得するためのメソッド。
        PyQt側でグラフ更新するときに呼び出す。
        """
        with self.lock:
            t_list = list(self.price_data[symbol]['times'])
            r_list = list(self.price_data[symbol]['returns'])
        return t_list, r_list


# -------------------------------------------------
# (B) PyQt + PyQtGraphを用いた GUI （メインスレッド）
# -------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, fetcher: CMELiveDataFetcher, symbol: str):
        super().__init__()
        self.fetcher = fetcher
        self.symbol = symbol
        
        #四分位数の境界値
        self.quartile_boundaries = {
            'Q1': 0.195,
            'Q2': 0.249,
            'Q3': 0.298,
            'Q4': 0.424            
        }
        try:
            csv_path = "subsequent_volatility_distribution.csv"
            self.subsequent_vol_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            self.logger.log('ERROR','MainWindow', f"{csv_path} not found. Volatility analysis will be disabled.")
            self.subsequent_vol_df = None
        
        # ウィンドウタイトル
        self.setWindowTitle(f"Market Monitor - {self.symbol}")
        self.resize(1200, 800)

        # 中央ウィジェット
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QtWidgets.QVBoxLayout(self.central_widget)

        # pyqtgraph の PlotWidget を用意
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # plot itemの初期化
        self.return_curve = self.plot_widget.plot([],[],
                                                  pen=pg.mkPen('y', width=2),
                                                  name='Return')
        self.volatility_curve = self.plot_widget.plot([],[],
                                                      pen=pg.mkPen('b', width=2),
                                                      name='Volatility')
        # 正のσライン
        self.sigma1_line = self.plot_widget.plot([], [],
                                                 pen=pg.mkPen('g', style=Qt.DashLine),
                                                 name='1σ')
        self.sigma2_line = self.plot_widget.plot([], [],
                                                 pen=pg.mkPen('r', style=Qt.DashLine),
                                                 name='2σ')
        # 負のσライン
        self.sigma1_line_neg = self.plot_widget.plot([],[],
                                                   pen=pg.mkPen('g', style=Qt.DashLine),
                                                   name='-1σ')
        self.sigma2_line_neg = self.plot_widget.plot([],[],
                                                   pen=pg.mkPen('r', style=Qt.DashLine),
                                                   name='-2σ')

        # Plotの見た目調整
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Change Rate (%)')
        self.plot_widget.setLabel('bottom', 'Time (min from open)')
        self.plot_widget.addLegend()
        
        self.volatility_buffer = []
        self.vol_times_buffer = []
        self.start_time = None
        self.current_quartile = None
        
        # 極地の追跡用の変数
        self.sigma_shift_upper = 0  # 上側のσライン用
        self.sigma_shift_lower = 0  # 下側のσライン用
        self.max_return = float('-inf')
        self.min_return = float('inf')

        # 初期σライン値を保持する変数を追加
        self.initial_sigma_values = {
            'times': [],
            'sigma1_upper': [],
            'sigma2_upper': [],
            'sigma1_lower': [],
            'sigma2_lower': []
        }
        
        # σライン初期化済みフラグ
        self.sigma_lines_initialized = False        
        
        # 一定間隔でタイマー起動→ update_plot()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)  # 1秒ごとに更新
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def initialize_sigma_lines(self):
        """初期のσライン値を計算して保存"""
        if not self.current_quartile or self.sigma_lines_initialized:
            return
            
        print(f"Initializing sigma lines for quartile {self.current_quartile}")
        
        # 現在のx_valsに基づいて時間点を生成
        times, returns = self.fetcher.get_data_copy(self.symbol)
        if len(times) == 0:
            return
            
        x_vals = [(t - self.start_time).total_seconds() / 60 for t in times]
        max_time = max(x_vals)
        
        # より細かい時間点を生成（滑らかな曲線のため）
        time_points = np.linspace(0, max(360, max_time), 100)
        
        # 各時間点でのσ値を計算
        sigma1_upper = []
        sigma2_upper = []
        sigma1_lower = []
        sigma2_lower = []
        
        for t in time_points:
            # 最も近い期間を見つける
            closest_period = min([5, 30, 60, 240, 360], 
                               key=lambda x: abs(x - t))
            
            df_q = self.subsequent_vol_df[
                (self.subsequent_vol_df['quartile'] == self.current_quartile) &
                (self.subsequent_vol_df['period'] == f"{closest_period}min")
            ]
            
            if not df_q.empty:
                mean_ = df_q['subsequent_volatility'].mean()
                std_ = df_q['subsequent_volatility'].std()
                
                sigma1_upper.append(mean_ + std_)
                sigma2_upper.append(mean_ + 2 * std_)
                sigma1_lower.append(-(mean_ + std_))
                sigma2_lower.append(-(mean_ + 2 * std_))
        
        # 初期値を保存
        self.initial_sigma_values['times'] = time_points
        self.initial_sigma_values['sigma1_upper'] = sigma1_upper
        self.initial_sigma_values['sigma2_upper'] = sigma2_upper
        self.initial_sigma_values['sigma1_lower'] = sigma1_lower
        self.initial_sigma_values['sigma2_lower'] = sigma2_lower
        
        self.sigma_lines_initialized = True
        self.update_sigma_lines()  # 初期描画
        
    def calculate_initial_volatility(self, returns_window):
        """
        returns_window: たとえば最初の5つのリターン（%）を格納したリスト
        例: [0.00, 0.07, -0.26, -0.35, -0.54]

        この中の最大値 - 最小値 をボラティリティとする。
        （差は必ず0以上なのでabsはつけなくてもOKですが、つけても同じ結果になります）
        """
        if not returns_window:
            return 0.0
        vol_range = max(returns_window) - min(returns_window)
        return abs(vol_range)
    
    def determine_quartile(self, value):
        """value (初期ボラ) に応じて Q1/Q2/Q3/Q4 を返す"""
        # 例えば:
        if value < self.quartile_boundaries['Q1']:
            return 'Q1'
        elif value < self.quartile_boundaries['Q2']:
            return 'Q2'
        elif value < self.quartile_boundaries['Q3']:
            return 'Q3'
        else:
            return 'Q4'        

    def update_plot(self):
        # データ取得
        times, returns = self.fetcher.get_data_copy(self.symbol)
        if len(times) == 0:
            return

        # タイムスタンプ処理
        if self.start_time is None and times:
            self.start_time = times[0]

        x_vals = [(t - self.start_time).total_seconds() / 60 for t in times]
        
        # 1. リターンの折れ線データ更新
        self.return_curve.setData(x_vals, returns)

        # 2. 四分位数の初期判定（まだ実施していない場合）
        if self.current_quartile is None and len(returns) > 5:
            init_vol = self.calculate_initial_volatility(returns[:5])
            self.current_quartile = self.determine_quartile(init_vol)
            # σラインの初期化と描画
            self.initialize_sigma_lines()

        # 高値/安値の更新チェックとシフト量の計算
        if returns and self.current_quartile:
            current_max = max(returns)
            current_min = min(returns)
            shift_updated = False
            
            if self.max_return == float('-inf'):
                self.max_return = current_max
            if self.min_return == float('inf'):
                self.min_return = current_min

            # σラインが描画されているか確認
            sigma_data = self.sigma1_line.getData()
            if sigma_data[0] is None or len(sigma_data[0]) == 0:
                print("Initial sigma line drawing")
                self.update_sigma_lines()
                return

            # -------------------------
            # 高値更新 => 下側ライン(-1σ, -2σ)を上に動かす
            # -------------------------
            if current_max > self.max_return:
                print(f"[High Update] current_max={current_max} > old={self.max_return}")
                # たとえば "差分 = current_max - self.max_return" を
                # sigma_shift_lower にプラスして、下側ラインがその分だけ上昇
                diff = (current_max - self.max_return)
                old_shift = self.sigma_shift_lower
                self.sigma_shift_lower += diff
                print(f"   sigma_shift_lower: {old_shift} -> {self.sigma_shift_lower}")
                shift_updated = True

            # -------------------------
            # 安値更新 => 上側ライン(+1σ, +2σ)を下に動かす
            # -------------------------
            if current_min < self.min_return:
                print(f"[Low Update] current_min={current_min} < old={self.min_return}")
                # 差分 = current_min - old_min は負の値
                # これを sigma_shift_upper に加算 => 上側ラインがその分だけ下に移動
                diff = (current_min - self.min_return)
                old_shift = self.sigma_shift_upper
                self.sigma_shift_upper += diff
                print(f"   sigma_shift_upper: {old_shift} -> {self.sigma_shift_upper}")
                shift_updated = True

            # max/min の更新
            if current_max > self.max_return:
                self.max_return = current_max
            if current_min < self.min_return:
                self.min_return = current_min

            # もしシフト値が変わったらライン再描画
            if shift_updated:
                print(f"[Update Sigma] upper={self.sigma_shift_upper}, lower={self.sigma_shift_lower}")
                self.update_sigma_lines()

        # 5. プロット範囲の調整
        self.adjust_plot_ranges(x_vals, returns)

    def update_sigma_lines(self):
        """保存された初期値にシフト量を適用して再描画"""
        if not self.sigma_lines_initialized:
            self.initialize_sigma_lines()
            return
            
        print(f"Updating sigma lines with shifts - Upper: {self.sigma_shift_upper}, Lower: {self.sigma_shift_lower}")
        
        # シフト量を適用して再描画
        times = self.initial_sigma_values['times']
        
        # 上側のライン
        sigma1_upper = [v + self.sigma_shift_upper for v in self.initial_sigma_values['sigma1_upper']]
        sigma2_upper = [v + self.sigma_shift_upper for v in self.initial_sigma_values['sigma2_upper']]
        
        # 下側のライン
        sigma1_lower = [v + self.sigma_shift_lower for v in self.initial_sigma_values['sigma1_lower']]
        sigma2_lower = [v + self.sigma_shift_lower for v in self.initial_sigma_values['sigma2_lower']]
        
        # σラインの描画前後の値を確認
        print("\nBefore setData:")
        old_sigma1_upper = self.sigma1_line.getData()[1]
        old_sigma2_upper = self.sigma2_line.getData()[1]
        old_sigma1_lower = self.sigma1_line_neg.getData()[1]
        old_sigma2_lower = self.sigma2_line_neg.getData()[1]
        
        if old_sigma1_upper is not None:
            print(f"Upper σ1: {old_sigma1_upper[0]:.4f} -> {sigma1_upper[0]:.4f}")
            print(f"Upper σ2: {old_sigma2_upper[0]:.4f} -> {sigma2_upper[0]:.4f}")
            print(f"Lower σ1: {old_sigma1_lower[0]:.4f} -> {sigma1_lower[0]:.4f}")
            print(f"Lower σ2: {old_sigma2_lower[0]:.4f} -> {sigma2_lower[0]:.4f}")
        
        # σラインの描画
        self.sigma1_line.setData(times, sigma1_upper)
        self.sigma2_line.setData(times, sigma2_upper)
        self.sigma1_line_neg.setData(times, sigma1_lower)
        self.sigma2_line_neg.setData(times, sigma2_lower)
        
        # 描画後の値を再度確認
        print("\nAfter setData:")
        current_sigma1_upper = self.sigma1_line.getData()[1]
        current_sigma2_upper = self.sigma2_line.getData()[1]
        current_sigma1_lower = self.sigma1_line_neg.getData()[1]
        current_sigma2_lower = self.sigma2_line_neg.getData()[1]
        
        if current_sigma1_upper is not None:
            print(f"Upper σ1: {current_sigma1_upper[0]:.4f}")
            print(f"Upper σ2: {current_sigma2_upper[0]:.4f}")
            print(f"Lower σ1: {current_sigma1_lower[0]:.4f}")
            print(f"Lower σ2: {current_sigma2_lower[0]:.4f}")

    def adjust_plot_ranges(self, x_vals, returns):
        """プロットの表示範囲を調整"""
        # Y軸の範囲調整
        y_candidates = list(returns)
        _, y_s1 = self.sigma1_line.getData()
        _, y_s2 = self.sigma2_line.getData()
        _, y_s1_neg = self.sigma1_line_neg.getData()
        _, y_s2_neg = self.sigma2_line_neg.getData()
        
        for y_data in [y_s1, y_s2, y_s1_neg, y_s2_neg]:
            if y_data is not None:
                y_candidates.extend(y_data)
        
        if y_candidates:
            ymin = min(y_candidates)
            ymax = max(y_candidates)
            yrange = ymax - ymin
            self.plot_widget.setYRange(ymin - 0.1*yrange, ymax + 0.1*yrange)
        
        # X軸の範囲調整
        if x_vals:
            self.plot_widget.setXRange(0, max(x_vals) + 1)
                    

def run_fetcher_loop(fetcher: CMELiveDataFetcher):
    """
    別スレッドで asyncio イベントループを作り、run_async を回す
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def runner():
        async with fetcher:
            await fetcher.run_async()

    try:
        loop.run_until_complete(runner())
    finally:
        loop.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='NQH5', help='Symbol to subscribe')
    args = parser.parse_args()

    fetcher = CMELiveDataFetcher(debug_mode=False)

    # 2) バックグラウンドスレッドで fetcher を起動
    fetcher_thread = threading.Thread(
        target=run_fetcher_loop,
        args=(fetcher,),
        daemon=True
    )
    fetcher_thread.start()

    # 3) PyQtアプリ起動（メインスレッド）
    app = QtWidgets.QApplication(sys.argv)
    
    window = MainWindow(fetcher, args.symbol)
    window.show()

    # 4) Databentoで購読するシンボルをセットアップ
    #    すでに run_async 内で "NQ.c.0" をデフォルト購読してますが
    #    もしシンボルを引数で変えたいときは fetcher 内で setup_subscription をしてもよい
    
    # 5) GUIループ開始
    sys.exit(app.exec_())


if __name__=="__main__":
    main()

else:
    __all__ = ['MainWindow']