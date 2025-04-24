import asyncio, edge_tts, uuid
import pygame, time, os
import concurrent.futures

voice_file_path = "D:/project/python/jarvis/audio/"

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

async def tts(text: str, voice: str = "zh-TW-YunJheNeural") -> str:
    """
    Convert text to speech using Edge TTS and save it to a file.

    Args:
        text (str): The text to convert to speech.
        file_name (str): The name of the output audio file.
        voice (str): The voice to use for the speech synthesis. Default is "zh-TW-YunJheNeural".
    """
    file_name = str(uuid.uuid4()) + ".mp3"
    file_full_name = voice_file_path + file_name
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(file_full_name)
    return file_name

async def generate_and_play_audio(text: str, voice: str = "zh-TW-YunJheNeural"):
    file_name = await tts(text, voice)
    play_audio(file_name)

def submit_audio_task(text: str, voice: str = "zh-TW-YunJheNeural"):
    executor.submit(generate_and_play_audio(text,voice))

def play_audio(file_name: str):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(voice_file_path + file_name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)  # 等待音频播放结束
        print("播放完成.")
        pygame.mixer.quit()
        os.unlink(voice_file_path + file_name)  # 删除音频文件
    except pygame.error as e:
        print(f"Error playing audio: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


