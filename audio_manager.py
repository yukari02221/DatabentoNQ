import winsound
from logging_utils import CustomLogger
import pytz
from datetime import datetime
import asyncio

class AudioManager:
    """オーディオ再生を管理するクラス"""
    
    def __init__(self, debug_mode: bool):
        self.logger = CustomLogger("AudioManager")
        self.audio_played_today = False
        self.debug_mode = debug_mode
        self.ny_tz = pytz.timezone('America/New_York')
        self.morning_announcement_path = r'resorce\NY市場前朝礼.wav'
        
    async def check_and_play_audio(self):
        """定期的にオーディオ再生のタイミングをチェックする"""
        while True:
            now = datetime.now(self.ny_tz)
            
            if self.debug_mode:
                debug_now = now.replace(hour=9, minute=15)
                if not self.audio_played_today:
                    self.logger.log('DEBUG', 'audio', f"Debug time: {debug_now}")
                    try:
                        winsound.PlaySound(self.morning_announcement_path, winsound.SND_FILENAME)
                        self.audio_played_today = True
                        self.logger.log('INFO', 'audio', "Played morning announcement (DEBUG)")
                    except Exception as e:
                        self.logger.log('ERROR', 'audio', f"Failed to play audio: {str(e)}")
                        
            elif now.hour == 9 and now.minute == 15 and not self.audio_played_today:
                try:
                    winsound.PlaySound(self.morning_announcement_path, winsound.SND_FILENAME)
                    self.audio_played_today = True
                    self.logger.log('INFO', 'audio', "Played morning announcement")
                except Exception as e:
                    self.logger.log('ERROR', 'audio', f"Failed to play audio: {str(e)}")
            
            if now.hour == 0 and now.minute == 0:
                self.audio_played_today = False
            
            await asyncio.sleep(30)