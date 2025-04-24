from funasr   import AutoModel
import datetime, pyaudio,time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from modelscope.pipelines import pipeline
import webrtcvad
import pygame
import wave
import threading,asyncio
from ..agent.chat_agent import answer


model_dir = r"D:\model\SenseVoiceSmall"

model_sence_voice = AutoModel(model = model_dir, trust_remote_code = True)

# 参数设置
AUDIO_RATE = 16000        # 音频采样率
AUDIO_CHANNELS = 1        # 单声道
CHUNK = 1024              # 音频块大小
VAD_MODE = 3              # VAD 模式 (0-3, 数字越大越敏感)
OUTPUT_DIR = r"D:\project\python\jarvis\audio\tmp"   # 输出目录
NO_SPEECH_THRESHOLD = 1   # 无效语音阈值，单位：秒

flag_sv_used = 1
flag_sv_enroll = 0
thred_sv = 0.35

last_active_time = time.time()
recording_active = True
segments_to_save = []
saved_intervals = []
last_vad_end_time = 0  # 上次保存的 VAD 有效段结束时间
audio_file_count = 0
audio_file_count_tmp = 0

owner_voice_file_path = r"D:\project\python\jarvis\audio\声纹1-lin.wav"

sv_pipeline = pipeline(
    task='speaker-verification',
    model= 'damo/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)
# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# 保存音频和视频
def save_audio_video():
    pygame.mixer.init()

    global segments_to_save, video_queue, last_vad_end_time, saved_intervals

    # 全局变量，用于保存音频文件名计数
    global audio_file_count
    global flag_sv_enroll
    global set_SV_enroll

    if flag_sv_enroll:
        audio_output_path = owner_voice_file_path
    else:
        audio_file_count += 1
        audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"
    # audio_output_path = f"{OUTPUT_DIR}/audio_0.wav"

    print(f"临时音频保存路径: {audio_output_path}")
    
    if not segments_to_save:
        return
    
    # 停止当前播放的音频
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        print("检测到新的有效音，已停止当前音频播放")
        
    # 获取有效段的时间范围
    start_time = segments_to_save[0][1]
    end_time = segments_to_save[-1][1]
    
    # 检查是否与之前的片段重叠
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("当前片段与之前片段重叠，跳过保存")
        segments_to_save.clear()
        return
    
    # 保存音频
    audio_frames = [seg[0] for seg in segments_to_save]
    if flag_sv_enroll:
        audio_length = 0.5 * len(segments_to_save)
        if audio_length < 1:
            print("声纹注册语音需大于1秒，请重新注册")
            return 1

    wf = wave.open(audio_output_path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    print(f"音频保存至 {audio_output_path}")

    if flag_sv_enroll:
        text = "声纹注册完成！现在只有你可以命令我啦！"
        print(text)
        flag_sv_enroll = 0
        # system_introduction(text)
    else:
    # 使用线程执行推理
        if judge_speaker(audio_output_path) is False:
            print("非管理员声音, 无需理会")
            # 清空缓冲区
            segments_to_save.clear()
            return
        text = parse_audio(audio_output_path)
        asyncio.run(answer(text))
        
        # inference_thread = threading.Thread(target=answer, args=(text))
        # inference_thread.start()
        
        # 记录保存的区间
        saved_intervals.append((start_time, end_time))
        
    # 清空缓冲区
    segments_to_save.clear()

def record_audio():
    print("按下回车开始录音")
    input()
    print("按下回车结束录音")
    
    recording = []
    try:
        def callback(indata,frames,time,status):
            recording.append(indata.copy())
        with sd.InputStream(samplerate=44100, channels=1, callback=callback):
            input() # 等待用户按下回车 结束录音
    except Exception as e:
        print(f"录音失败: {e}")
        return
    
    # 将录音保存为文件
    audio_data = np.concatenate(recording, axis=0)
    write(filename="recording.wav", rate=44100, data=(audio_data * 32767).astype(np.int16))
    print("录音已保存为 recording.wav")
    
def parse_audio(input_file: str = "recording.wav"):
    res= model_sence_voice.generate(
            input=input_file,
            cache={},
            language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
    )
    print(f"音频解析内容 res: {res}")
    return res[0]['text'].split(">")[-1]

def judge_speaker(target_speaker_file_path):
    sv_score = sv_pipeline([target_speaker_file_path, owner_voice_file_path], thr= 0.35)
    if sv_score["text"] != "yes":
        print("无有效权限访问")
        return False
    else:
        print("有权限访问")
        return True

# 检测 VAD 活动
def check_vad_activity(audio_data):
    # 将音频数据分块检测
    num, rate = 0, 0.4
    step = int(AUDIO_RATE * 0.02)  # 20ms 块大小
    flag_rate = round(rate * len(audio_data) // step)

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    if num > flag_rate:
        return True
    return False

# 音频录制线程
def audio_recorder():
    global audio_queue, recording_active, last_active_time, segments_to_save, last_vad_end_time

    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    audio_buffer = []
    print("音频录制已开始")
    
    while recording_active:
        data = stream.read(CHUNK)
        audio_buffer.append(data)
        
        # 每 0.5 秒检测一次 VAD
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            # 拼接音频数据并检测 VAD
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)
            
            if vad_result:
                print("检测到语音活动")
                last_active_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
            else:
                print("静音中...")
            
            audio_buffer = []  # 清空缓冲区
        
        # 检查无效语音时间
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
            # 检查是否需要保存
            if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:
                save_audio_video()
                last_active_time = time.time()
            else:
                pass
                # print("无新增语音段，跳过保存")
    
    stream.stop_stream()
    stream.close()
    p.terminate()