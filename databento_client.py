import databento as db
from typing import Optional, Callable
from logging_utils import CustomLogger
from datetime import datetime, timedelta
import pytz

class DatabentoCommunicationError(Exception):
    """Databento通信に関するError"""
    pass

class DatabentoClient:
    def __init__(self, api_key: str):
        self.logger = CustomLogger("DatabentoClient")
        self.api_key = api_key
        self.client: Optional[db.Live] = None
        # "初回のみ"かどうかを判定するフラグ
        self.first_time_subscribed = False

    async def setup_subscription(self, symbols: str = "NQ.c.0", use_preopen_data: bool = False):
        """
        指定したシンボルのサブスクリプションをセットアップする。
        use_preopen_data=True なら「9時の1時間前からのヒストリカルを含める」動作とする。
        """
        try:
            # 既に購読中のclientがあれば一旦stop
            if self.client:
                self.client.stop()

            # サブスクリプション開始時刻を決定
            if use_preopen_data and (not self.first_time_subscribed):
                # 例：当日8:00(ET)をUTCにして指定
                start_time_str = get_utc_string_for_pre_open()  
            else:
                # 通常は "now" にする
                start_time_str = "now"

            self.logger.log("INFO", "setup_subscription",
                            f"Using start_time = {start_time_str} for subscription")

            self.client = db.Live(key=self.api_key)
            self.client.subscribe(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                stype_in="continuous",
                symbols=symbols,
                start=start_time_str,  # ここに文字列を入れる
            )
            self.logger.log("INFO", "setup_subscription", f"Subscription done for {symbols}")

            # 初回サブスクが済んだらフラグを立てる
            if not self.first_time_subscribed:
                self.first_time_subscribed = True

        except Exception as e:
            self.logger.log("ERROR", "setup_subscription", f"Failed: {str(e)}")
            raise DatabentoCommunicationError(f"Failed to setup subscription: {str(e)}")

    def start(self, record_callback: Callable):
        """データ受信を開始する"""
        if not self.client:
            raise DatabentoCommunicationError("Client not initialized")
        self.client.add_callback(record_callback=record_callback)
        self.client.start()
        
    def stop(self):
        """データ受信を停止する"""
        if self.client:
            self.client.stop()

    async def wait_for_close(self, timeout: float):
        """指定した時間だけデータ受信を継続する"""
        if not self.client:
            raise DatabentoCommunicationError("Client not initialized")
        await self.client.wait_for_close(timeout=timeout)


def get_utc_string_for_pre_open():
    """
    例: 今日の 9:00(ET) - 1時間 = 8:00(ET) を UTC に変換した文字列を返す
    """
    ny_tz = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny_tz)

    # 当日9:00
    today_9am_ny = now_ny.replace(hour=9, minute=0, second=0, microsecond=0)
    # 1時間前 => 8:00
    one_hour_before = today_9am_ny - timedelta(hours=1)

    # UTC変換
    one_hour_before_utc = one_hour_before.astimezone(pytz.UTC)
    # Databento 用フォーマット 'YYYY-MM-DDTHH:MM:SSZ'
    start_str = one_hour_before_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    return start_str
