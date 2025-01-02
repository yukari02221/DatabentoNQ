import databento as db
from logging_utils import CustomLogger
import asyncio
import pandas as pd
from databento_dbn import SymbolMappingMsg, OHLCVMsg, SystemMsg
import os
from typing import Optional, Dict
from datetime import datetime, timedelta
import pytz
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict
import numpy as np

class CMELiveDataFetcher:
    def __init__(self, api_key: Optional[str] = None):
        self.logger = CustomLogger("CMELiveDataFetcher")
        
        # 環境変数からAPIキーを取得
        self.api_key = api_key or os.environ.get('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either through environment variable 'DATABENTO_API_KEY' or as a parameter")
        
        self.client = None
        self.logger.log('INFO', 'init', "CMELiveDataFetcher initialized")

        # instrument_id → symbol名 (stype_out_symbol) を保持する辞書
        self.instrument_id_map = {}

        # 価格データを保存するための辞書
        self.price_data: Dict[str, Dict] = defaultdict(lambda: {
            'times': [],
            'prices': [],
            'base_price': None,
            'returns': [],
            'fig': None,
            'ax': None,
            'market_open_time': None
        })
        
        # プロット更新間隔（秒）
        self.plot_update_interval = 5.0
        self.last_plot_update = time.time()

        
        # NY市場の時間設定
        self.ny_tz = pytz.timezone('America/New_York')
        self.logger.log('INFO', 'init', f"Timezone set to: {self.ny_tz}")

    def reset_price_data(self):
        """価格データのリセット"""
        for symbol in self.price_data:
            self.price_data[symbol].update({
                'times': [],
                'prices': [],
                'base_price': None,
                'returns': [],
                'market_open_time': None
            })
            # プロット関連のオブジェクトは保持
            plt.close(self.price_data[symbol]['fig'])
            self.price_data[symbol]['fig'] = None
            self.price_data[symbol]['ax'] = None
        
        self.logger.log('INFO', 'reset_price_data', "Price data has been reset for new session")
        
    def get_next_market_open(self) -> datetime:
        """次の取引開始時間（NY時間9:30の30分前）を取得"""
        now = datetime.now(self.ny_tz)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # 現在時刻が今日の取引開始時間以降の場合、次の取引日を取得
        if now >= market_open:
            market_open += timedelta(days=1)
        
        # 土日を月曜日に調整
        while market_open.weekday() in [5, 6]:  # 5=土曜日, 6=日曜日
            days_until_monday = (7 - market_open.weekday()) % 7
            market_open += timedelta(days=days_until_monday)
            
        # 市場開始30分前の時間を返す
        return market_open - timedelta(minutes=30)

    def get_market_close(self, target_date: datetime) -> datetime:
        """指定された日の取引終了時間（NY時間16:00）を取得

        Args:
            target_date (datetime): 取引終了時間を取得したい日付

        Returns:
            datetime: NY時間16:00に設定された取引終了時間
        """
        return target_date.replace(hour=16, minute=0, second=0, microsecond=0)

    def setup_plot(self, symbol: str):
        """プロットの初期設定"""
        if self.price_data[symbol]['fig'] is None:
            self.price_data[symbol]['fig'], self.price_data[symbol]['ax'] = plt.subplots(figsize=(12, 6))
            self.price_data[symbol]['ax'].set_title(f'{symbol} Intraday Returns')
            self.price_data[symbol]['ax'].set_xlabel('Time (UTC)')
            self.price_data[symbol]['ax'].set_ylabel('Return from Open (%)')
            self.price_data[symbol]['ax'].grid(True)
            plt.ion()  # インタラクティブモードを有効化
            
    def update_plot(self, symbol: str):
        """プロットの更新"""
        current_time = time.time()
        if current_time - self.last_plot_update < self.plot_update_interval:
            return

        if len(self.price_data[symbol]['returns']) > 0:
            ax = self.price_data[symbol]['ax']
            ax.clear()
            ax.plot(self.price_data[symbol]['times'], self.price_data[symbol]['returns'], 'b-')
            ax.set_title(f'{symbol} Intraday Returns ({datetime.now().strftime("%Y%m%d")})')
            ax.set_xlabel('Time (UTC)')
            ax.set_ylabel('Return from Open (%)')
            ax.grid(True)
            plt.draw()
            plt.pause(1)
            
        self.last_plot_update = current_time

    def setup_subscription(self, symbols: str = "NQ.c.1"):
        try:
            self.client = db.Live(key=self.api_key)
            self.client.subscribe(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                stype_in="continuous",
                symbols=symbols,
                start="now",
            )
            self.logger.log(
                "INFO",
                "setup_subscription",
                f"Subscription setup completed for {symbols}"
            )
        except Exception as e:
            self.logger.log("ERROR", "setup_subscription", f"Failed to setup subscription: {str(e)}")
            raise

    def process_bar(self, record) -> None:
        try:
            # 1) シンボルマッピングメッセージの場合: instrument_id_mapを更新
            if isinstance(record, SymbolMappingMsg):
                self.instrument_id_map[record.instrument_id] = record.stype_out_symbol
                self.logger.log(
                    'INFO', 
                    'process_bar',
                    f"Symbol Mapping: ID={record.instrument_id}, "
                    f"Input Symbol Type={record.stype_in_symbol}, "
                    f"Output Symbol Type={record.stype_out_symbol}, "
                    f"Time={record.pretty_ts_event}, "
                    f"Start={record.pretty_start_ts}, "
                    f"End={record.pretty_end_ts}"
                )

            # 2) システムメッセージの場合
            elif isinstance(record, SystemMsg):
                self.logger.log(
                    'INFO', 
                    'process_bar',
                    f"System Message: {record.msg}"
                )

            # 3) OHLCVメッセージの場合: instrument_idからシンボル名を取り出してOHLCVを処理
            elif isinstance(record, OHLCVMsg):
                # instrument_id からシンボル名を取得（存在しなければ 'Unknown' などに）
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
                current_time = (
                    pd.Timestamp(record.ts_event)
                    .tz_localize("UTC")    # tz-naive → UTC にする
                    .tz_convert(self.ny_tz)  # UTC → America/New_York に変換
                )

                # NY時間9:30のタイムスタンプを設定（初回のみ）
                if self.price_data[mapped_symbol]['market_open_time'] is None:
                    market_open_time = pd.Timestamp(datetime.now(self.ny_tz).replace(
                        hour=9, minute=30, second=0, microsecond=0
                    ))
                    self.price_data[mapped_symbol]['market_open_time'] = market_open_time
                
                # データを保存
                self.price_data[mapped_symbol]['times'].append(current_time)
                self.price_data[mapped_symbol]['prices'].append(bar_data['close'])
                
                # NY時間9:30以降の最初のデータを基準価格として設定
                if (self.price_data[mapped_symbol]['base_price'] is None and 
                    current_time >= self.price_data[mapped_symbol]['market_open_time']):
                    self.price_data[mapped_symbol]['base_price'] = bar_data['close']
                    self.setup_plot(mapped_symbol)
                
                # リターンを計算（%）- 基準価格が設定されている場合のみ
                if self.price_data[mapped_symbol]['base_price'] is not None:
                    current_return = ((bar_data['close'] / self.price_data[mapped_symbol]['base_price']) - 1) * 100
                    self.price_data[mapped_symbol]['returns'].append(current_return)
                    
                    # プロットを更新
                    self.update_plot(mapped_symbol)
                    
                    self.logger.log('INFO', 'process_bar', 
                                  f"OHLCV Data: {bar_data}, Return: {current_return:.2f}%")
                else:
                    self.price_data[mapped_symbol]['returns'].append(0.0)  # 寄り付き前はリターン0%
                    self.logger.log('INFO', 'process_bar', 
                                  f"OHLCV Data: {bar_data}, Waiting for market open...")

            else:
                self.logger.log(
                    'INFO',
                    'process_bar',
                    f"New Record Type: {type(record)}\n"
                    f"Available Attributes: {dir(record)}"
                )

        except Exception as e:
            self.logger.log(
                'ERROR', 
                'process_bar', 
                f"Error processing record: {str(e)}, "
                f"Record type: {type(record)}"
            )

    async def run_session(self, market_close: datetime):
        """1取引セッションの実行"""
        try:
            # セッション開始時にデータをリセット
            self.reset_price_data()
            self.setup_subscription()
            self.client.add_callback(record_callback=self.process_bar)
            self.client.start()
            self.logger.log('INFO', 'run_session', "Streaming started")

            # 取引終了時刻までの待機時間を計算
            now = datetime.now(self.ny_tz)
            wait_seconds = (market_close - now).total_seconds()
            
            if wait_seconds > 0:
                await self.client.wait_for_close(timeout=wait_seconds)

        except Exception as e:
            self.logger.log(
                'ERROR', 
                'run_session', 
                f"Error during streaming: {str(e)}"
            )
            raise
        finally:
            if self.client:
                self.client.stop()
                self.logger.log('INFO', 'run_session', "Streaming stopped")
            # セッション終了時にもデータをリセット
            self.reset_price_data()

    async def run_async(self):
        """メインループ処理"""
        while True:
            try:
                # 次の取引開始時間を取得
                next_market_open = self.get_next_market_open()
                market_close = self.get_market_close(next_market_open)
                
                self.logger.log(
                    'INFO', 
                    'run_async', 
                    f"Waiting for next market session. Open: {next_market_open}, Close: {market_close}"
                )

                # 取引開始時間まで待機
                now = datetime.now(self.ny_tz)
                wait_seconds = (next_market_open - now).total_seconds()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

                # セッション実行
                await self.run_session(market_close)

            except Exception as e:
                self.logger.log(
                    'ERROR', 
                    'run_async', 
                    f"Error in main loop: {str(e)}"
                )
                # エラー発生時は5分待機してから再試行
                await asyncio.sleep(300)

def main():
    logger = CustomLogger("Main")
    logger.log('INFO', 'main', "Starting CME live data fetcher")

    try:
        fetcher = CMELiveDataFetcher()
        asyncio.run(fetcher.run_async())
    except Exception as e:
        logger.log('ERROR', 'main', f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()