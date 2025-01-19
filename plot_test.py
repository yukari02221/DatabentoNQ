import sqlite3
import pandas as pd
from datetime import datetime
import pytz
from PyQt5 import QtWidgets, QtCore
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
            'price_queue': [],
            'queue_initialized': False
        })
        self.lock = threading.Lock()
        self.ny_tz = pytz.timezone('America/New_York')
        
        self.simulation_data = None
        self.current_index = 0
        self.timer = None
        self.QUEUE_MAX_SIZE = 60

    def load_historical_data(self, date_str: str):
        try:
            conn = sqlite3.connect(self.db_path)
            
            # シミュレーション開始時刻の1時間前から取得
            init_start_time = int(date_str + "1300")    # 13:00
            simulation_start = int(date_str + "1400")   # 14:00 (シミュレーション開始)
            end_time = int(date_str + "2100")          # 21:00
            
            query = """
            SELECT datetime, open, high, low, close, volume
            FROM mnq_1m_ohlcv
            WHERE datetime >= ? AND datetime < ?
            ORDER BY datetime            
            """
            
            df = pd.read_sql_query(query, conn, params=(init_start_time, end_time))
            
            if len(df) == 0:
                print(f"No data found for date {date_str}")
                return
            
            # タイムスタンプの変換
            df['timestamp'] = pd.to_datetime(df['datetime'].astype(str), format='%Y%m%d%H%M')
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(self.ny_tz)
            
            # シミュレーション開始時刻を特定
            simulation_start_time = pd.to_datetime(str(simulation_start), format='%Y%m%d%H%M')
            simulation_start_time = simulation_start_time.tz_localize('UTC').tz_convert(self.ny_tz)
            
            with self.lock:
                symbol = 'MNQ'
                
                # 13:00-13:59のデータをキューに追加
                prev_data = df[df['timestamp'] < simulation_start_time]
                if not prev_data.empty:
                    for _, row in prev_data.iterrows():
                        self.price_data[symbol]['price_queue'].append({
                            'low': row['low'],
                            'high': row['high'],
                            'close': row['close']
                        })
                    self.price_data[symbol]['queue_initialized'] = True
                    print(f"Initialized queue with {len(prev_data)} historical periods (13:00-13:59)")
                
                # シミュレーション用のデータを設定 (14:00以降)
                self.simulation_data = df[df['timestamp'] >= simulation_start_time]
                print(f"Loaded {len(self.simulation_data)} rows for simulation (14:00 onwards)")
                
                if len(self.simulation_data) > 0:
                    self.price_data[symbol]['base_price'] = self.simulation_data.iloc[0]['close']
                    self.price_data[symbol]['market_open_time'] = simulation_start_time
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()

    def simulate_next_minute(self):
        if self.simulation_data is None or self.current_index >= len(self.simulation_data):
            if self.timer:
                self.timer.stop()
            return
        
        with self.lock:
            symbol = 'MNQ'
            current_row = self.simulation_data.iloc[self.current_index]
            
            # 価格データを辞書形式で保存
            price_data = {
                'low': current_row['low'],
                'high': current_row['high'],
                'close': current_row['close']
            }
            
            # キューの更新（既に初期化されているはず）
            self.price_data[symbol]['price_queue'].pop(0)  # 最も古いデータを削除
            self.price_data[symbol]['price_queue'].append(price_data)  # 新しいデータを追加
            
            self.price_data[symbol]['times'].append(current_row['timestamp'])
            self.price_data[symbol]['prices'].append(price_data['close'])
            
            base_price = self.price_data[symbol]['base_price']
            ret = ((price_data['close'] / base_price) - 1) * 100
            self.price_data[symbol]['returns'].append(ret)
            
            self.current_index += 1
            
            print(f"Added data point {self.current_index}/{len(self.simulation_data)}: "
                  f"Time={current_row['timestamp']}, Return={ret:.2f}%")

    def get_data_copy(self, symbol: str):
        with self.lock:
            if symbol not in self.price_data:
                return [], []
            t_list = list(self.price_data[symbol]['times'])
            r_list = list(self.price_data[symbol]['returns'])
        return t_list, r_list

    def get_price_queue_copy(self, symbol: str):
        """スレッドセーフに price_queue のコピーを取得"""
        with self.lock:
            if symbol not in self.price_data:
                return []
            return list(self.price_data[symbol]['price_queue'])

class SimulationWindow(MainWindow):
    def __init__(self, fetcher: HistoricalDataFetcher, symbol: str):
        # 親クラスの初期化を先に行う
        super().__init__(fetcher, symbol)
        
        # シミュレーション用タイマーの設定
        self.simulation_timer = QtCore.QTimer()
        self.simulation_timer.timeout.connect(fetcher.simulate_next_minute)
        self.simulation_timer.start(1000)  # 1秒ごとに更新

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', required=True, help='Path to market_data.db')
    parser.add_argument('--date', required=True, help='Date to analyze (YYYYMMDD)')
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    
    fetcher = HistoricalDataFetcher(args.db_path)
    fetcher.load_historical_data(args.date)
    
    window = SimulationWindow(fetcher, 'MNQ')
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()