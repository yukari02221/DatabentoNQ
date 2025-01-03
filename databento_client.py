import databento as db
from typing import Optional, Callable
from logging_utils import CustomLogger

class DatabentoCommunicationError(Exception):
    """Databento通信に関するError"""
    pass

class DatabentoClient:
    def __init__(self, api_key: str):
        self.logger = CustomLogger("DatabentoClient")
        self.api_key = api_key
        self.client: Optional[db.Live] = None
        
    async def setup_subscription(self, symbols: str = "NQ.c.0"):
        """指定したシンボルのサブスクリプションをセットアップする"""
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
            