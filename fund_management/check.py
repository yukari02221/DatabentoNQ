import sqlite3
import pandas as pd
from datetime import datetime
import os

class TradeDatabase:
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self.backup_path = db_path + '.backup'
        self.initialize_database()

    def initialize_database(self):
        """データベースとテーブルの初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # トレードテーブルの作成
                conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    trade_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    pnl REAL NOT NULL,
                    commission REAL NOT NULL,
                    fees REAL NOT NULL,
                    direction TEXT NOT NULL
                )
                """)
                conn.commit()
                print("データベースの初期化が完了しました")

        except Exception as e:
            print(f"データベース初期化中にエラーが発生しました: {str(e)}")
            raise

    def show_table_schema(self, table_name: str):
        """指定したテーブルのスキーマ情報を表示する"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns_info = cursor.fetchall()
                if columns_info:
                    print(f"テーブル '{table_name}' のスキーマ情報:")
                    print("cid | name | type | notnull | dflt_value | pk")
                    for col in columns_info:
                        print(col)
                else:
                    print(f"テーブル '{table_name}' は存在しません。")
        except Exception as e:
            print(f"スキーマ情報取得中にエラーが発生しました: {str(e)}")

    # ...（以下、既存のメソッドは省略）...

def main():
    try:
        # データベースの初期化
        db = TradeDatabase("trades.db")
        
        # テーブル構造の確認
        db.show_table_schema("trades")
        
        # ここで後続の処理を行う場合は続けて記述
        # 例: CSVファイルからデータを読み込み、挿入処理を行うなど
        # df = read_trade_data("trade.csv")
        # success = db.insert_trades_safely(df)
        # if not success:
        #     print("トレードデータの挿入に失敗しました")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
