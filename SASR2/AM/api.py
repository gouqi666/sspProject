from AM.AMmodel.model import AM
from AM.utils.user_config import UserConfig


class ASR:
    def __init__(self, am_config):
        self.am = AM(am_config)
        self.am.load_model(False)

    def decode_am_result(self, result):
        return self.am.decode_result(result)

    def stt(self, wav_path):
        am_result = self.am.predict(wav_path)

        am_result = self.decode_am_result(am_result[0])

        return am_result

    def am_test(self, wav_path):
        am_result = self.am.predict(wav_path)
        am_result = self.decode_am_result(am_result[0])
        return am_result


am_config = UserConfig(r'AM/conformerCTC(M)/am_data.yml', r'AM/conformerCTC(M)/conformerM.yml')
ASRModule = ASR(am_config)


def speech2pinyin(wav_path):
    return ASRModule.stt(wav_path)
