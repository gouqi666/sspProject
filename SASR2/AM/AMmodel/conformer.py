from ..AMmodel.conformer_blocks import ConformerEncoder
import tensorflow as tf

from ..AMmodel.layers.time_frequency import Melspectrogram, Spectrogram
from ..utils.text_featurizers import TextFeaturizer


class CTCModel(tf.keras.Model):
    def __init__(self,
                 encoder: tf.keras.Model,
                 num_classes: int,
                 speech_config,
                 name="ctc_model",

                 **kwargs):
        super(CTCModel, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        # Fully connected layer
        self.speech_config = speech_config
        self.mel_layer = None
        if speech_config['use_mel_layer']:
            if speech_config['mel_layer_type'] == 'Melspectrogram':
                self.mel_layer = Melspectrogram(sr=speech_config['sample_rate'],
                                                n_mels=speech_config['num_feature_bins'],
                                                n_hop=int(
                                                    speech_config['stride_ms'] * speech_config['sample_rate'] // 1000),
                                                n_dft=1024,
                                                trainable_fb=speech_config['trainable_kernel']
                                                )
            else:
                self.mel_layer = Spectrogram(
                    n_hop=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000),
                    n_dft=1024,
                    trainable_kernel=speech_config['trainable_kernel']
                )
            self.mel_layer.trainable = speech_config['trainable_kernel']
        self.wav_info = speech_config['add_wav_info']
        if self.wav_info:
            assert speech_config['use_mel_layer'] == True, 'shold set use_mel_layer is True'

        self.fc = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=num_classes, activation="linear",
                                  use_bias=True), name="fully_connected")
        self.recognize_pb = None

    def _build(self, sample_shape):
        features = tf.random.normal(shape=sample_shape)
        self(features, training=False)

    def summary(self, line_length=None, **kwargs):
        self.encoder.summary(line_length=line_length, **kwargs)
        super(CTCModel, self).summary(line_length, **kwargs)

    def add_featurizers(self,
                        text_featurizer: TextFeaturizer,
                        ):

        self.text_featurizer = text_featurizer

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        if self.mel_layer is not None:
            if self.wav_info:
                wav = inputs
                inputs = self.mel_layer(inputs)
            else:
                inputs = self.mel_layer(inputs)
        if self.wav_info:
            enc_outputs = self.encoder([inputs, wav], training=training)
        else:
            enc_outputs = self.encoder(inputs, training=training)
        outputs = self.fc(enc_outputs[-1], training=training)
        for i in range(10, 15):
            outputs += self.fc(enc_outputs[i], training=training)
        return outputs

    def return_pb_function(self, shape, beam=False):
        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape, dtype=tf.float32),
                tf.TensorSpec([None,1], dtype=tf.int32),

            ]
        )
        def recognize_tflite(features, length):
            logits = self.call(features, training=False)

            probs = tf.nn.softmax(logits)
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.squeeze(length, -1), greedy=True
            )[0][0]
            return [decoded]

        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape, dtype=tf.float32),
                tf.TensorSpec([None, 1], dtype=tf.int32),
            ]
        )
        def recognize_beam_tflite(features, length):
            logits = self.call(features, training=False)

            probs = tf.nn.softmax(logits)
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.squeeze(length, -1), greedy=False,
                beam_width=self.text_featurizer.decoder_config["beam_width"]
            )[0][0]
            return [decoded]

        self.recognize_pb = recognize_tflite if not beam else recognize_beam_tflite

    def get_config(self):
        if self.mel_layer is not None:
            config = self.mel_layer.get_config()
            config.update(self.encoder.get_config())
        else:
            config = self.encoder.get_config()
        config.update(self.fc.get_config())
        return config


class ConformerCTC(CTCModel):
    def __init__(self,
                 dmodel: int,
                 reduction_factor: int,
                 vocabulary_size: int,
                 num_blocks: int = 16,
                 head_size: int = 36,
                 num_heads: int = 4,
                 kernel_size: int = 32,
                 fc_factor: float = 0.5,
                 dropout: float = 0,
                 name='conformerCTC',
                 speech_config=dict,
                 **kwargs):
        super(ConformerCTC, self).__init__(
            encoder=ConformerEncoder(
                dmodel=dmodel,
                reduction_factor=reduction_factor,
                num_blocks=num_blocks,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                fc_factor=fc_factor,

                dropout=dropout,
                add_wav_info=speech_config['add_wav_info'],
                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000) * reduction_factor,
            ), num_classes=vocabulary_size, name=name, speech_config=speech_config)
        self.time_reduction_factor = reduction_factor
