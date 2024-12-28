import logging
from logging.handlers import RotatingFileHandler
from typing import Dict
from multiprocessing.queues import Queue as MPQueue

LOG_FILE = 'DatabentoNQ.log'
MAX_FILE_SIZE = 1024 * 1024
BACKUP_COUNT = 5

class CustomLogger:
    def __init__(self, name: str, log_file: str = LOG_FILE, max_file_size: int = MAX_FILE_SIZE, backup_count: int = BACKUP_COUNT):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)        
            formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S')
            
            # ファイルハンドラの設定
            file_handler = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=backup_count)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            #コンソールハンドラーの設定
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        self.logger.propagate = False
        
    def log(self, level: str, tag: str, message: str):
        log_message = f"{tag}: {message}"
        getattr(self.logger, level.lower())(log_message)

class ProcessLogger:
    """プロセス用のロガーラッパー"""
    def __init__(self, name: str, log_queue: 'MPQueue[Dict[str, str]]'):
        self.name = name
        self.log_queue = log_queue
    def log(self, level: str, tag: str, message: str) -> None:
        """ログメッセージをメインプロセスに送信"""
        self.log_queue.put({
            'name': self.name,
            'level': level,
            'tag': tag,
            'message': message            
        })