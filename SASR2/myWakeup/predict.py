# import pyaudio,wave
from utils import wavfile_to_mfccs,normalize_mean,spectrogram,load_audio
import numpy as np
import librosa
from model import DNN,GatedConv
import torch
# 定义数据流块
CHUNK = 1024
# FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
# 录音时间
# pa = pyaudio.PyAudio()
# stream = pa.open(rate=RATE,format=FORMAT,channels = CHANNELS,input =True, frames_per_buffer=CHUNK)
WAVE_OUTPUT_FILENAME = './tmp.wav'
RECORD_SECONDS = 3
input_size = 7700  #

# model = DNN(input_size = input_size,hidden_size=10000)
model = GatedConv()
model.load_state_dict(torch.load('./model/best_model.pt'))
while True:
    # record_buf = []
    # for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
    #     pcm = stream.read(CHUNK)
    #     record_buf.append(pcm)
    #
    # wf = wave.open(WAVE_OUTPUT_FILENAME,'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(pa.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(record_buf))
    # wf.close()
    # mfccs,_,_ = wavfile_to_mfccs('./tmp.wav')
    wav = load_audio('./tmp.wav')
    spect = spectrogram(wav)
    # y = np.concatenate(mfccs)
    # if len(y) > input_size:
    #     y = y[:input_size]
    # else:
    #     y = y + [0] * (input_size - len(y))
    # X_train, scaler_mean = normalize_mean(X_train)
    spect = torch.unsqueeze(spect,0)
    pred = model.forward(torch.tensor(spect))
    print(pred)
    pred = torch.argmax(pred)
    if pred == 1:
        print("主人主人我在呢！")
stream.stop_stream()
stream.close()
pa.terminate()
# wf = wave.open('../02.wav', 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(pa.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(record_buf))
# wf.close()
# wf = wave.open('../02.wav','rb')
# data = wf.readframes(CHUNK)
# print(data)
# print(len(data))
