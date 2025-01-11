import sqlite3
import pandas as pd
from datetime import datetime
import pytz
from PyQt5 import QtWidgets
import sys
from collections import defaultdict
import threading
from main import MainWindow

class HistoricalDataFetcher:
    def __init__(self, db_path: str, debug_mode: bool = False):
        self.db_path = db_path
        self.debug_mode = debug_mode
        self.price_data = defaultdict(lambda: {
            'times': [],
            'prices': [],
            'base_price': None,
            'returns': [],
            'market_open_time': None,
        })
        self.lock = threading.Lock()
        self.ny_tz = pytz.timezone('America/New_York')
        
    def load_historical_data(self, date_str: str):
        """特定の日付のデータを読み込む"""
        try:
            conn = sqlite3.connect(self.db_path)
            print(f"Loading data for date: {date_str}")
            
            # 指定日のデータを読み込む
            query = """
            SELECT datetime, open, high, low, close, volume
            FROM mnq_1m_ohlcv
            WHERE datetime >= ? AND datetime < ?
            ORDER BY datetime            
            """
            
            start_time = int(date_str + "0000")
            end_time = int(date_str + "2359")
            
            df = pd.read_sql_query(query, conn, params=(start_time, end_time))
            
            if len(df) == 0:
                print(f"No data found for date {date_str}")
                return
            
            print(f"Loaded {len(df)} rows of data")
            
            # タイムスタンプ変換
            df['timestamp'] = pd.to_datetime(df['datetime'].astype(str), format='%Y%m%d%H%M')
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(self.ny_tz)
            
            # 9:30以降のデータのみを使用
            market_open = df['timestamp'].dt.normalize() + pd.Timedelta(hours=9, minutes=30)
            df = df[df['timestamp'] >= market_open]
            
            if len(df) > 0:
                with self.lock:
                    symbol = 'MNQ'
                    self.price_data[symbol]['times'] = list(df['timestamp'])
                    self.price_data[symbol]['prices'] = list(df['close'])
                    self.price_data[symbol]['base_price'] = df.iloc[0]['close']
                    # リターンを計算
                    base_price = self.price_data[symbol]['base_price']
                    self.price_data[symbol]['returns'] = [
                        ((price / base_price) - 1) * 100
                        for price in self.price_data[symbol]['prices']
                    ]
                    self.price_data[symbol]['market_open_time'] = market_open.iloc[0]
                print(f"Processed {len(df)} rows of market data")
        
        except Exception as e:
            print(f"Error loading data: {str(e)}")
        finally:
            conn.close()

    def get_data_copy(self, symbol: str):
        """MainWindowクラスと同じインターフェースを提供"""
        with self.lock:
            if symbol not in self.price_data:
                return [], []
            t_list = list(self.price_data[symbol]['times'])
            r_list = list(self.price_data[symbol]['returns'])
        return t_list, r_list        

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', required=True, help='Path to market_data.db')
    parser.add_argument('--date', required=True, help='Date to analyze (YYYYMMDD)')
    args = parser.parse_args()
    
    # データ読み込み
    fetcher = HistoricalDataFetcher(args.db_path)
    fetcher.load_historical_data(args.date)
    
    # PyQtアプリ起動
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(fetcher, 'MNQ')
    window.show()
    
    sys.exit(app.exec_())
    
if __name__=="__main__":
    main()