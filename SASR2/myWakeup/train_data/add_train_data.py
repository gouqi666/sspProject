import pyaudio,wave
#from utils.wakeup import wavfile_to_mfccs,normalize_mean
import numpy as np
import glob
import argparse
import os
from tqdm import tqdm
# 定义数据流块
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
# 录音时间
pa = pyaudio.PyAudio()
stream = pa.open(rate=RATE,format=FORMAT,channels = CHANNELS,input =True, frames_per_buffer=CHUNK)

RECORD_SECONDS = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--positive', action='store_true')
    args = parser.parse_args()
    ## add negative samples in default, you can add positive smaples by run "python add_train_data.py --postive".
    if False:
        WAVE_OUTPUT_FILENAME = './train_data/positive'
        pos_wav_files = glob.glob(WAVE_OUTPUT_FILENAME + "/*.WAV")
        count = len(pos_wav_files)
        num = 50  ## once add 20 samples in default, you can change it.
        is_start = input("can we start?")
        while is_start[0] != 'y':  # y means ok
            is_start = input("can we start?")
        for i in range(0,num):
            record_buf = []
            print("%d record begin:" % i)
            for _ in tqdm(range(int(RATE / CHUNK * RECORD_SECONDS))):
                pcm = stream.read(CHUNK)
                record_buf.append(pcm)
            print("%d record end:" % i)
            wf = wave.open(os.path.join(WAVE_OUTPUT_FILENAME, 'a' + str(count) + '.wav'), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(record_buf))
            wf.close()
            count += 1
            is_start = input("can we start next record?(y or n)")
            while is_start[0] != 'y':  # y means ok
                is_start = input("can we start next record?")
    else:
        WAVE_OUTPUT_FILENAME = './train_data/negative'
        neg_wav_files = glob.glob(WAVE_OUTPUT_FILENAME + "/*.WAV")
        count = len(neg_wav_files)

        num = 50  ## once add 20 samples in default, you can change it.

        for i in range(num):
            record_buf = []
            print("%d record begin:" % i)
            for _ in tqdm(range(int(RATE / CHUNK * RECORD_SECONDS))):
                pcm = stream.read(CHUNK)
                record_buf.append(pcm)
            print("%d record end:" % i)
            wf = wave.open(os.path.join(WAVE_OUTPUT_FILENAME, 'a' + str(count) + '.wav'), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(record_buf))
            wf.close()
            count += 1

