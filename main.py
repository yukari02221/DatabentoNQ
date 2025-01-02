import databento as db
from logging_utils import CustomLogger
import asyncio
import pandas as pd
from databento_dbn import SymbolMappingMsg, OHLCVMsg, SystemMsg
import os
from typing import Optional, Dict
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
import threading
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys

# -------------------------------------------------
# (A) 非同期でデータを取得し、価格データとリターンを保持するだけのクラス
# -------------------------------------------------
class CMELiveDataFetcher:
    def __init__(self, api_key: Optional[str] = None, debug_mode: bool = False):
        self.logger = CustomLogger("CMELiveDataFetcher")
        
        self.api_key = api_key or os.environ.get('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided.")
        
        self.client = None
        self.logger.log('INFO', 'init', "CMELiveDataFetcher initialized")
        
        self.instrument_id_map = {}
        
        # price_data[symbol] の中に times / returns などを格納
        # Matplotlibを使わないので fig/ax は削除
        self.price_data: Dict[str, Dict] = defaultdict(lambda: {
            'times': [],
            'prices': [],
            'base_price': None,
            'returns': [],
            'market_open_time': None,
        })

        self.ny_tz = pytz.timezone('America/New_York')
        self.logger.log('INFO', 'init', f"Timezone set to: {self.ny_tz}")
        
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.logger.log('INFO', 'init', "Running in debug mode")
        
        # スレッドセーフにするためのロック
        self.lock = threading.Lock()
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.stop()
        await self.reset_price_data()

    async def reset_price_data(self):
        with self.lock:
            for symbol in self.price_data:
                self.price_data[symbol].update({
                    'times': [],
                    'prices': [],
                    'base_price': None,
                    'returns': [],
                    'market_open_time': None
                })
        self.logger.log('INFO', 'reset_price_data', "Price data has been reset for new session")

    def get_next_market_open(self) -> datetime:
        now = datetime.now(self.ny_tz)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now >= market_open:
            market_open += timedelta(days=1)
        while market_open.weekday() in [5, 6]:
            days_until_monday = (7 - market_open.weekday()) % 7
            market_open += timedelta(days=days_until_monday)
        return market_open - timedelta(minutes=30)

    def get_market_close(self, target_date: datetime) -> datetime:
        return target_date.replace(hour=16, minute=0, second=0, microsecond=0)

    async def setup_subscription(self, symbols: str = "NQ.c.0"):
        try:
            if self.client:
                self.client.stop()
            self.client = db.Live(key=self.api_key)
            self.client.subscribe(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                stype_in="continuous",
                symbols=symbols,
                start="now",
            )
            self.logger.log("INFO", "setup_subscription", f"Subscription done for {symbols}")
        except Exception as e:
            self.logger.log("ERROR", "setup_subscription", f"Failed: {str(e)}")
            raise

    def process_bar(self, record) -> None:
        asyncio.create_task(self._process_bar_async(record))

    async def _process_bar_async(self, record) -> None:
        try:
            if isinstance(record, SymbolMappingMsg):
                self.instrument_id_map[record.instrument_id] = record.stype_out_symbol
                self.logger.log('INFO','process_bar',
                    f"Symbol Mapping: ID={record.instrument_id}, "
                    f"Out={record.stype_out_symbol}, Time={record.pretty_ts_event}"
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
                    
                    # base_price セット
                    if self.price_data[mapped_symbol]['base_price'] is None:
                        self.price_data[mapped_symbol]['base_price'] = bar_data['close']
                    
                    self.price_data[mapped_symbol]['times'].append(current_time)
                    self.price_data[mapped_symbol]['prices'].append(bar_data['close'])
                    
                    base = self.price_data[mapped_symbol]['base_price']
                    if base is not None:
                        ret = (bar_data['close'] / base - 1) * 100
                        self.price_data[mapped_symbol]['returns'].append(ret)
                    else:
                        self.price_data[mapped_symbol]['returns'].append(0.0)

                self.logger.log('INFO','process_bar',
                    f"OHLCV Data: {bar_data}, Return={self.price_data[mapped_symbol]['returns'][-1]:.2f}%")

            else:
                self.logger.log('INFO','process_bar', f"New Record Type: {type(record)}")

        except Exception as e:
            self.logger.log('ERROR','process_bar', f"Error: {str(e)}")

    async def run_session(self, market_close: datetime):
        try:
            await self.reset_price_data()
            await self.setup_subscription()
            self.client.add_callback(record_callback=self.process_bar)
            self.client.start()
            self.logger.log('INFO','run_session', "Streaming started")

            now = datetime.now(self.ny_tz)
            wait_seconds = (market_close - now).total_seconds()
            if wait_seconds > 0:
                await self.client.wait_for_close(timeout=wait_seconds)

        except Exception as e:
            self.logger.log('ERROR','run_session', f"Error: {str(e)}")
            raise
        finally:
            if self.client:
                self.client.stop()
            self.logger.log('INFO','run_session', "Streaming stopped")
            await self.reset_price_data()

    async def run_async(self):
        while True:
            try:
                if self.debug_mode:
                    # デバッグ中は「明日まで走らせる」イメージ
                    await self.run_session(datetime.now(self.ny_tz) + timedelta(hours=24))
                else:
                    next_open = self.get_next_market_open()
                    mclose = self.get_market_close(next_open)
                    now = datetime.now(self.ny_tz)
                    wait_secs = (next_open - now).total_seconds()
                    if wait_secs > 0:
                        await asyncio.sleep(wait_secs)
                    await self.run_session(mclose)

            except Exception as e:
                self.logger.log('ERROR','run_async', f"Main loop error: {str(e)}")
                await asyncio.sleep(300)  # 5分待ってリトライ

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
        self.symbol = "NQH5"
        
        # ウィンドウタイトル
        self.setWindowTitle(f"Realtime Returns - {self.symbol}")
        self.resize(800, 600)

        # 中央ウィジェット
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        # レイアウト
        layout = QtWidgets.QVBoxLayout(self.central_widget)

        # pyqtgraph の PlotWidget を用意
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # 折れ線用の PlotDataItem
        self.curve = self.plot_widget.plot([], [], pen='y')

        # X軸を「整数 or float」のままで扱う (実際は経過秒などでOK)
        # ここでは「秒単位の経過時間」を X 軸にしてみる例にします
        # start_time に最初の約定データが来た時刻を記録し、そこからの差分（秒）を x とする
        self.start_time = None

        # Plotの見た目調整
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('bottom', 'Time (sec from first data)')
        self.plot_widget.setLabel('left', 'Return from Open (%)')

        # 一定間隔でタイマー起動→ update_plot()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)  # 1秒ごとに更新
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        # データ取得
        times, returns = self.fetcher.get_data_copy(self.symbol)
        if len(times) == 0:
            return

        # 最初のデータを基準にして秒数を計算
        # times は pydatetime 付き pandas.Timestamp なので、.timestamp() で epoch秒に変換
        if self.start_time is None:
            self.start_time = times[0].timestamp()

        x_vals = [t.timestamp() - self.start_time for t in times]
        y_vals = returns
        
        # 折れ線データ更新
        self.curve.setData(x_vals, y_vals)


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
    parser.add_argument('--symbol', default='NQ.c.0', help='Symbol to subscribe')
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

if __name__ == "__main__":
    main()