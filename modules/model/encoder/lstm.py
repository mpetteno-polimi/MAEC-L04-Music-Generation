import logging as log
import tensorflow as tf
from tensorflow import keras as ks


# Hyperparams from MusicVAE config: cat-mel_2bar_big	cat-mel_2bar_big	2-bar melodies
class LstmEncoder(ks.Model):
    def __init__(self, config):  # todo: default config object
        super(LstmEncoder, self).__init__()
        # prepare keras-compatible tensor-like input data
        input_shape = (config.IN_SEQ_LEN, config.AVAILABLE_NOTES)
        self.inputsKs = tf.keras.Input(input_shape)
        self.LSTM = ks.layers.LSTM(128, )  # todo: parametrize and tune hparams
        self.densePitch = ks.layers.Dense(128, name='enc_pitch')
        self.denseStep = ks.layers.Dense(1, name='enc_step')
        self.denseDuration = ks.layers.Dense(1, name='enc_duration')
        self.denseMean = ks.layers.Dense(config.LATENT_DIMENSION, name="z_mean")
        self.denseLogVar = ks.layers.Dense(config.LATENT_DIMENSION, name="z_log_var")

    def call(self, inputs, training=None, mask=None):
        keras_in = self.inputsKs(inputs)
        x = self.LSTM(keras_in)

        outputs = {
            'pitch': self.densePitch(x),
            'step': self.denseStep(x),
            'duration': self.denseDuration(x),
        }

        z_mean = self.denseMean(outputs)
        z_log_var = self.denseLogVar(outputs)

        return [z_mean, z_log_var]

        # VISUALIZATION FZ
        # keras.utils.plot_model(self.model, "encoder_model_info.png", show_shapes=True)
        # self.model.summary()

        # understandable vae that offers a slightly more complex encoder at https://keras.io/examples/generative/vae/

    # ================================= PRIVATE METHODS =========================================
