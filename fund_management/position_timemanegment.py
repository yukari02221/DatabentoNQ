import threading
import time
import winsound

def play_sound_file(file_path):
    """指定された音声ファイルを再生する"""
    try:
        winsound.PlaySound(file_path, winsound.SND_FILENAME)
    except Exception as e:
        print(f"音声ファイルの再生中にエラーが発生しました: {e}")

def play_4min_sound():
    """4分経過後に音声を再生し、その後5分おきに再生する"""
    # 最初の4分待機
    time.sleep(240)  # 4分 = 240秒
    
    while True:
        play_sound_file('4minutes.wav')
        # 5分待機
        time.sleep(300)  # 5分 = 300秒

def play_5min_sound():
    """5分おきに音声を再生する"""
    while True:
        time.sleep(300)  # 5分待機
        play_sound_file('5minutes.wav')

def main():
    try:
        # 4分タイマーのスレッドを作成
        thread_4min = threading.Thread(target=play_4min_sound)
        thread_4min.daemon = True  # メインスレッド終了時に一緒に終了
        
        # 5分タイマーのスレッドを作成
        thread_5min = threading.Thread(target=play_5min_sound)
        thread_5min.daemon = True  # メインスレッド終了時に一緒に終了
        
        # スレッドを開始
        thread_4min.start()
        thread_5min.start()
        
        print("タイマーを開始しました。終了するには Ctrl+C を押してください。")
        
        # メインスレッドを実行し続ける
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nプログラムを終了します。")

if __name__ == "__main__":
    main()