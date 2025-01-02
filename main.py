import databento as db
from logging_utils import CustomLogger
import asyncio
import pandas as pd
from databento_dbn import SymbolMappingMsg, OHLCVMsg, SystemMsg
import os
from typing import Optional

class CMELiveDataFetcher:
    def __init__(self, api_key: Optional[str] = None):
        self.logger = CustomLogger("CMELiveDataFetcher")
        
        # 環境変数からAPIキーを取得
        self.api_key = api_key or os.environ.get('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either through environment variable 'DATABENTO_API_KEY' or as a parameter")
        
        self.client = db.Live(key=self.api_key)
        self.logger.log('INFO', 'init', "Live client initialized")

        # instrument_id → symbol名 (stype_out_symbol) を保持する辞書
        self.instrument_id_map = {}

    def setup_subscription(self, symbols: str = "NQ.c.1"):
        try:
            self.client.subscribe(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                stype_in="continuous",
                symbols=symbols,
                start="2024-12-31T22:00:00"
            )
            self.logger.log(
                'INFO', 
                'setup_subscription', 
                f"Subscription setup completed for {symbols}"
            )
        except Exception as e:
            self.logger.log(
                'ERROR', 
                'setup_subscription', 
                f"Failed to setup subscription: {str(e)}"
            )
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

                self.logger.log('INFO', 'process_bar', f"OHLCV Data: {bar_data}")

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

    async def run_async(self):
        try:
            self.client.add_callback(record_callback=self.process_bar)
            self.client.start()
            self.logger.log('INFO', 'run_async', "Streaming started")

            # 1時間実行
            await self.client.wait_for_close(timeout=3600)

        except Exception as e:
            self.logger.log(
                'ERROR', 
                'run_async', 
                f"Error during streaming: {str(e)}"
            )
            raise
        finally:
            self.client.stop()
            self.logger.log('INFO', 'run_async', "Streaming stopped")

def main():
    logger = CustomLogger("Main")
    logger.log('INFO', 'main', "Starting CME live data fetcher")

    try:
        fetcher = CMELiveDataFetcher()
        fetcher.setup_subscription()
        asyncio.run(fetcher.run_async())
    except Exception as e:
        logger.log('ERROR', 'main', f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()