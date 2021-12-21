import pyaudio
from collections import UserDict

audio_config = {
    "frames_per_buffer": 1024,
    "format": pyaudio.paFloat32,
    "channels": 1,
    "rate": 16000
}

stop_config = {
    # 静音多少秒停止
    "SILENCE_THRESHOLD_SEC": 2,
    # 多小声算静音，我这里底噪太大了，所以设置的比较高
    "SILENCE_THRESHOLD": 10
}
