import numpy
import librosa
import os
import glob
import numpy as np
from utils import wavfile_to_mfccs
def read_wav_files(data_path):
    pos_data_path = os.path.join(data_path,'positive')
    neg_data_path = os.path.join(data_path,'negative')
    pos_wav_files = sorted(glob.glob(pos_data_path + "/*.WAV"))
    neg_wav_files = sorted(glob.glob(neg_data_path + "/*.WAV"))
    x, y = [], []
    for file in pos_wav_files:
        mfccs,_,_ = wavfile_to_mfccs(file)
        x.append(list(np.concatenate(mfccs)))
        y.append(1)
    for file in neg_wav_files:
        mfccs,_,_ = wavfile_to_mfccs(file)
        x.append(list(np.concatenate(mfccs)))
        y.append(0)
    return x,y