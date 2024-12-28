from logging_utils import CustomLogger
from multiprocessing import Queue
import queue

def log_handler(log_queue: Queue):
    """メインプロセスでのログ処理"""
    main_logger = CustomLogger("Main")
    while True:
        try:
            log_data = log_queue.get()
            if log_data is None:
                break
            main_logger.log(
                log_data['level'],
                log_data['tag'],
                log_data['message']
            )
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in log handler: {str(e)}")