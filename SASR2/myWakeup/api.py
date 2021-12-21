import pyaudio,wave
from .utils import wavfile_to_mfccs,normalize_mean,spectrogram,load_audio
import numpy as np
import librosa
from .model import DNN,GatedConv
import torch
# 定义数据流块
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
# 录音时间
WAVE_OUTPUT_FILENAME = './TmpWavFiles'
RECORD_SECONDS = 3
# input_size = 7700


class WakeUpModule:
    def __init__(self):
        self.model = GatedConv()
        # self.model.load_state_dict(torch.load('./model/best_model.pt'))
    def wakeUp(self):
        pa = pyaudio.PyAudio()
        t=0
        stream = pa.open(rate=RATE, format=FORMAT, channels=CHANNELS, input=True, frames_per_buffer=CHUNK)
        while True:
            record_buf = []
            for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
                pcm = stream.read(CHUNK)
                record_buf.append(pcm)
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(record_buf))
            wf.close()

            wav = load_audio('./tmp.wav')
            spect = spectrogram(wav)
            spect = torch.unsqueeze(spect, 0)
            pred = self.model.forward(torch.tensor(spect))
            print(pred)
            pred = torch.argmax(pred)
            if pred == 0 or t==1:
                print("主人主人我在呢！")
                break
            t+=1
        stream.stop_stream()
        stream.close()
        pa.terminate()
        return True

