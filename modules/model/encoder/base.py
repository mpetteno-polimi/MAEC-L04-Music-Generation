import logging as log
import tensorflow as tf
from tensorflow import keras as ks


class BaseEncoder(ks.Model):
    def batch_size(self):
        return self.batch_size

    def max_sequence_length(self):
        return self.max_sequence_length

    def input_depth(self):
        return self.input_depth

    def __init__(self, config):  # todo: default config object
        super(BaseEncoder, self).__init__()

        try:
            if config.BATCH_SIZE:
                self.batch_size = config.BATCH_SIZE
            if config.MAX_SEQUENCE_LENGTH:
                self.max_sequence_length = config.MAX_SEQUENCE_LENGTH
            if config.INPUT_DEPTH:
                self.input_depth = config.INPUT_DEPTH
        except:  # todo: specify errors?
            self.batch_size = 512
            self.max_sequence_length = 256  # 1 bar, sixteenth notes: 16*16
            self.input_depth = 4   # MIDI: pitch, velocity, start_time, end_time}
            log.debug(self, "config input exception, set: self.batch_size = 512 - self.max_sequence_length = 256 -self.input_depth = 4")

        # prepare keras-compatible tensor-like input data
        input_shape = (self.batch_size, self.max_sequence_length, self.input_depth)
        self.inputsKs = tf.keras.Input(input_shape)

        # todo: convert to time major (converting to time major is important before feeding to LSTM)
        # one lstm layer of size config.HIDDEN_SIZE
        self.lstm1 = ks.layers.LSTM(config.HIDDEN_SIZE, return_sequences=True)
        self.lstm2 = ks.layers.LSTM(config.HIDDEN_SIZE, )

        self.dense_mean = ks.layers.Dense(config.LATENT_DIMENSION, name="z_mean")
        self.dense_log_var = ks.layers.Dense(config.LATENT_DIMENSION, name="z_log_var")

    def call(self, inputs, training=None, mask=None):
        keras_in = self.inputsKs(inputs)
        x = self.lstm1(keras_in)
        x = self.lstm2(x)
        z_mean = self.denseMean(x)
        z_log_var = self.denseLogVar(x)
        return [z_mean, z_log_var]
