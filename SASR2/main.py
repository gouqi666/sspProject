import torch
from torch import nn
import numpy as np
import wave,pyaudio
import os
from LM.dataset import LMDataSet
from LM.model import Transformer
from LM.trainer import LMTrainer
from utils import lm_token_path, am_token_path, TextFeaturizer
from myWakeup.api import  WakeUpModule
from AM.api import ASR
from interface.recorder import AudioRecorderWithAutoStop
from AM.utils.user_config import UserConfig
from interface.config import audio_config, stop_config

import flask, json
import requests
import pandas as pd
import numpy as np
import base64
import datetime
import hashlib
import urllib
from tts.tts_xunfei import read_text
from log import write_log
#
import warnings
warnings.filterwarnings("ignore")

CHUNK = 1024
CHANNELS = 2
RATE = 16000
FORMAT = pyaudio.paInt16

if __name__ == "__main__":
    wakeup = WakeUpModule()
    am_config = UserConfig(r'AM/conformerCTC(M)/am_data.yml', r'AM/conformerCTC(M)/conformerM.yml')
    amModule = ASR(am_config)
    lmModule = LMTrainer()
    WAVE_OUTPUT_DIR = './DIALOGUE'
    url='http://114.212.86.105:5050/dialog/reply'
    if not os.path.exists(WAVE_OUTPUT_DIR):
        os.mkdir(WAVE_OUTPUT_DIR)
        count = 0
    else:
        dirs = os.listdir(WAVE_OUTPUT_DIR)
        count = len(dirs)
    first_time=True
    while True:
        print("start")
        #stream.start()
        is_wakeup = wakeup.wakeUp()
        if is_wakeup:
            print('wake up!')
            recorder = AudioRecorderWithAutoStop()
            _audio = recorder.record(True)
            WAVE_OUTPUT_FILENAME = os.path.join(WAVE_OUTPUT_DIR,str(count)+'.wav')
            recorder.save(WAVE_OUTPUT_FILENAME)
            am_result = amModule.stt(WAVE_OUTPUT_FILENAME)
            lm_result = lmModule.predict(am_result)
            u="".join(lm_result[0])
            print('lm output:',u)
            count += 1
            write_log(u,first_time)
            first_time=False
            header={"Content-Type":"application/json"}
            body={
            "bot_id": "d15265f7-deda-430f-8feb-caa6e05ee645",
            "customer_id":"project",
            "utterance":u
            }
            r=requests.post(url,headers=header,data=json.dumps(body))
            data=r.json()
            replys=data['data']
            for re in replys:
                print(re['value'])
                read_text(re['value'])
                write_log(re['value'],first_time)
                
            ## lm_result 是解析出的text
            ## 接下来接policy + tts


# print(LMTrainer().predict([item[0] for item in pinyin("郭俊杰")]))
