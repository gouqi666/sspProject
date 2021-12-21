import os

import numpy as np

from ..AMmodel.conformer import ConformerCTC
from ..utils.speech_featurizers import SpeechFeaturizer
from ..utils.text_featurizers import TextFeaturizer


class AM():
    def __init__(self, config):
        self.config = config
        self.update_model_type()
        self.speech_config = self.config['speech_config']
        if self.model_type != 'MultiTask':
            self.text_config = self.config['decoder_config']
        else:
            self.text_config = self.config['decoder3_config']
        self.model_config = self.config['model_config']
        self.text_feature = TextFeaturizer(self.text_config, True)
        self.speech_feature = SpeechFeaturizer(self.speech_config)

        self.init_steps = None

    def update_model_type(self):
        self.config['decoder_config'].update({'model_type': 'CTC'})
        self.model_type = 'CTC'

    def conformer_model(self, training):

        self.model_config.update({'vocabulary_size': self.text_feature.num_classes})

        assert self.model_config['name'] == 'ConformerCTC'
        self.model_config.update({'speech_config': self.speech_config})
        self.model = ConformerCTC(**self.model_config)

    def load_model(self, training=True):
        self.conformer_model(training)
        self.model.add_featurizers(self.text_feature)
        f, c = self.speech_feature.compute_feature_dim()

        if not training:
            if self.model.mel_layer is not None:
                self.model._build(
                    [3, 16000 if self.speech_config['streaming'] is False else self.model.chunk_size * 2, 1])
                self.model.return_pb_function([None, None, 1])
            else:
                self.model._build([3, 80, f, c])
                self.model.return_pb_function([None, None, f, c])
            self.load_checkpoint(self.config)

    def decode_result(self, word):
        de = []
        for i in word:
            if i != self.text_feature.stop:
                de.append(self.text_feature.index_to_token[int(i)])
            else:
                break
        return de

    def predict(self, fp):
        if '.pcm' in fp:
            data = np.fromfile(fp, 'int16')
            data = np.array(data, 'float32')
            data /= 32768
        else:
            data = self.speech_feature.load_wav(fp)
        if self.model.mel_layer is None:
            mel = self.speech_feature.extract(data)
            mel = np.expand_dims(mel, 0)

            input_length = np.array([[mel.shape[1] // self.model.time_reduction_factor]], 'int32')
        else:
            mel = data.reshape([1, -1, 1])
            input_length = np.array(
                [[mel.shape[1] // self.model.time_reduction_factor // (self.speech_config['sample_rate'] *
                                                                       self.speech_config['stride_ms'] / 1000)]],
                'int32')
        result = self.model.recognize_pb(mel, input_length)[0]

        return result

    def load_checkpoint(self, config):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(config['learning_config']['running_config']["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.init_steps = int(files[-1].split('_')[-1].replace('.h5', ''))
