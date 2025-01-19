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
        """データベースとテーブルの初期化（既存のテーブル構造を変更しない）"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT,
                    symbol TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    pnl REAL NOT NULL,
                    commission REAL NOT NULL,
                    fees REAL NOT NULL,
                    direction TEXT NOT NULL,
                    trade_date DATE
                )
                """)
                conn.commit()
                print("データベースの初期化が完了しました")

        except Exception as e:
            print(f"データベース初期化中にエラーが発生しました: {str(e)}")
            raise

    def insert_trades_safely(self, df: pd.DataFrame) -> bool:
        """
        安全なトレードデータの挿入
        Returns:
            bool: 挿入成功の場合True
        """
        try:
            # データ検証
            is_valid, message = self.validate_trade_data(df)
            if not is_valid:
                print(f"データ検証エラー: {message}")
                return False

            # 既存データとの重複チェック
            with sqlite3.connect(self.db_path) as conn:
                existing_ids_query = "SELECT id FROM trades WHERE id IS NOT NULL"
                result = pd.read_sql_query(existing_ids_query, conn)
                existing_ids = result['id'].astype(str).tolist() if not result.empty else []

                # DataFrame上で新しいトレードを抽出
                new_trades = df[~df['id'].astype(str).isin(existing_ids)]
                if len(new_trades) == 0:
                    print("新規トレードはありません")
                    return True

                # バックアップ作成
                self.create_backup()

                try:
                    for _, row in new_trades.iterrows():
                        # 日時を文字列に変換
                        entry_time_str = row['entry_time'].isoformat() if pd.notnull(row['entry_time']) else None
                        exit_time_str = row['exit_time'].isoformat() if pd.notnull(row['exit_time']) else None

                        conn.execute("""
                        INSERT INTO trades (
                            id, symbol, size, entry_time, exit_time,
                            entry_price, exit_price, pnl, commission, fees, direction
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(row['id']), 
                            row['symbol'], 
                            row['size'],
                            entry_time_str, 
                            exit_time_str,
                            row['entry_price'], 
                            row['exit_price'],
                            row['pnl'], 
                            row['commission'], 
                            row['fees'],
                            row['direction']
                        ))

                    conn.commit()
                    print(f"{len(new_trades)}件のトレードを追加しました")
                    return True

                except Exception as e:
                    conn.rollback()
                    print(f"データ挿入中にエラーが発生しました: {str(e)}")
                    self.restore_from_backup()
                    return False

        except Exception as e:
            print(f"予期せぬエラーが発生しました: {str(e)}")
            return False

    def validate_trade_data(self, df: pd.DataFrame) -> tuple[bool, str]:
        try:
            required_columns = [
                'id', 'symbol', 'size', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'pnl', 'commission', 'fees', 'direction'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"必須カラムが不足しています: {missing_columns}"

            if not df['size'].dtype.kind in 'iu':
                return False, "size カラムは整数である必要があります"
            
            if not pd.api.types.is_datetime64_any_dtype(df['entry_time']):
                return False, "entry_time は日時形式である必要があります"
            
            if not pd.api.types.is_datetime64_any_dtype(df['exit_time']):
                return False, "exit_time は日時形式である必要があります"

            numeric_columns = ['entry_price', 'exit_price', 'pnl', 'commission', 'fees']
            for col in numeric_columns:
                if not pd.to_numeric(df[col], errors='coerce').notna().all():
                    return False, f"{col} には数値のみ使用可能です"

            if (df['entry_time'] > df['exit_time']).any():
                return False, "entry_time は exit_time より前である必要があります"

            if not df['direction'].isin(['Long', 'Short']).all():
                return False, "direction は 'Long' または 'Short' である必要があります"

            return True, "検証成功"
            
        except Exception as e:
            return False, f"データ検証中にエラーが発生しました: {str(e)}"

    def create_backup(self):
        try:
            if os.path.exists(self.db_path):
                import shutil
                shutil.copy2(self.db_path, self.backup_path)
                print(f"バックアップを作成しました: {self.backup_path}")
        except Exception as e:
            print(f"バックアップ作成中にエラーが発生しました: {str(e)}")

    def restore_from_backup(self):
        try:
            if os.path.exists(self.backup_path):
                import shutil
                shutil.copy2(self.backup_path, self.db_path)
                print(f"バックアップから復元しました: {self.backup_path}")
        except Exception as e:
            print(f"復元中にエラーが発生しました: {str(e)}")

def parse_datetime(date_string: str) -> datetime:
    """日時文字列をdatetimeオブジェクトに変換する"""
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",  # マイクロ秒あり
        "%Y-%m-%d %H:%M:%S"      # マイクロ秒なし
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"サポートされている日時フォーマットではありません: {date_string}")

def read_trade_data(file_path: str) -> pd.DataFrame:
    """
    トレードデータを直接作成して返す
    """
    trades = [
        {
            'id': '540719608',
            'symbol': '/MNQ',
            'size': 1,
            'entry_time': parse_datetime('2025-01-17 23:37:30'),
            'exit_time': parse_datetime('2025-01-17 23:46:07'),
            'entry_price': 21553.00,
            'exit_price': 21553.00,
            'pnl': 0.00,
            'commission': 0,
            'fees': -0.74,
            'direction': 'Short'
        },
        {
            'id': '541019059',
            'symbol': '/MNQ',
            'size': 1,
            'entry_time': parse_datetime('2025-01-18 00:06:36'),
            'exit_time': parse_datetime('2025-01-18 00:07:35'),
            'entry_price': 21565.75,
            'exit_price': 21578.25,
            'pnl': -25.00,
            'commission': 0,
            'fees': -0.74,
            'direction': 'Short'
        },
        {
            'id': '541265496',
            'symbol': '/MNQ',
            'size': 1,
            'entry_time': parse_datetime('2025-01-18 00:22:30'),
            'exit_time': parse_datetime('2025-01-18 00:29:04'),
            'entry_price': 21514.50,
            'exit_price': 21525.75,
            'pnl': -22.50,
            'commission': 0,
            'fees': -0.74,
            'direction': 'Short'
        }
    ]
    
    return pd.DataFrame(trades)

def main():
    try:
        df = read_trade_data("dummy_path")  # パスは使用されませんが、関数の引数として必要
        print("データフレームの内容:")
        print(df)
        
        db = TradeDatabase("trades.db")
        success = db.insert_trades_safely(df)
        if not success:
            print("トレードデータの挿入に失敗しました")
        else:
            print("トレードデータの挿入に成功しました")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()