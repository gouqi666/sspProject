import librosa
import numpy as np
import pyaudio
import simpleaudio
from scipy.io import wavfile
from .config import audio_config, stop_config


def STE(curFrame):
    # 短时能量
    amp = np.sum(np.abs(curFrame))
    return amp


class SilenceDetector:
    def __init__(self):
        self.status = -1  # 0为未开始，1为正在录音，2为录音结束
        self.status_name = ['等待发音', '正在录音', '录音结束']
        self.chunks_threshold = int(np.ceil(audio_config['rate'] / audio_config['frames_per_buffer']
                                            * stop_config['SILENCE_THRESHOLD_SEC']))
        self.silence_count = 0
        self.switchStatus()

    @staticmethod
    def checkSilence(audio):
        return STE(audio) < stop_config["SILENCE_THRESHOLD"]

    def switchStatus(self):
        self.status += 1
        print(self.status_name[self.status])

    def stop(self, audio):
        if self.checkSilence(audio):
            if self.status == 1:
                self.silence_count += 1
        else:
            if self.status == 0:
                self.switchStatus()
            else:
                self.silence_count = 0
        if self.silence_count >= self.chunks_threshold:
            self.switchStatus()
        return self.silence_count >= self.chunks_threshold


class AudioRecorderWithAutoStop:
    def __init__(self):
        self.s = SilenceDetector()
        self.audio = []
        self.last_record = None

    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.float32)
        self.audio.append(data)
        if self.s.stop(data):
            return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def record(self, return_silence_tail=False):
        p = pyaudio.PyAudio()
        stream = p.open(**audio_config,
                        output=False,
                        input=True,
                        stream_callback=self.callback)
        stream.start_stream()
        while stream.is_active():
            pass
        stream.stop_stream()
        stream.close()
        p.terminate()
        res = np.concatenate(self.audio)
        self.audio.clear()
        if return_silence_tail:
            self.last_record = res
        else:
            self.last_record = res[:-self.s.chunks_threshold * audio_config['frames_per_buffer']]
        return self.last_record

    def save(self, file_name):
        if self.last_record is not None:
            wavfile.write(file_name, audio_config['rate'], self.last_record)


if __name__ == '__main__':
    recorder = AudioRecorderWithAutoStop()
    _audio = recorder.record(True)
    recorder.save('1.wav')
    # obj = simpleaudio.play_buffer(_audio, sample_rate=16000, num_channels=1, bytes_per_sample=4)

