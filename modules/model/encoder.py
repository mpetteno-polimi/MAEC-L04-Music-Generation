# TODO - DOC

import keras
from keras import models, layers
from keras import backend as K
from keras.initializers import initializers

from definitions import ConfigSections
from modules import utilities


class BidirectionalLstmEncoder(object):

    def __init__(self):
        self._model_config = utilities.load_configuration_section(ConfigSections.MODEL)
        self._model = None

    def encode(self, input_):
        return self._model(input_)

    def build(self, seq_length, frame_length, z_size):

        def sampling(mu_log_variance):
            mu, log_variance = mu_log_variance
            epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
            random_sample = mu + K.exp(log_variance / 2) * epsilon
            return random_sample

        def build_lstm_layer():
            return layers.LSTM(
                units=self._model_config.get("enc_rnn_size"),
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                bias_initializer="zeros",
                unit_forget_bias=True,
                dropout=self._model_config.get("enc_dropout"),
                recurrent_dropout=0.0,
                return_sequences=False,
                return_state=False,
                go_backwards=False,
                stateful=False,
                time_major=False,
                unroll=False
            )

        def build_stacked_bidirectional_lstm_layer(num_layers, current_layer):
            if num_layers == 0:
                return current_layer
            bidirectional_lstm_layer = layers.Bidirectional(layer=build_lstm_layer(), merge_mode="concat")
            return build_stacked_bidirectional_lstm_layer(num_layers - 1, bidirectional_lstm_layer(current_layer))

        encoder_inputs = keras.Input(shape=(seq_length, frame_length), name="encoder_input")

        # Bidirectional LSTM layer
        stacked_bidirectional_lstm_layer = build_stacked_bidirectional_lstm_layer(self._model_config.get("enc_layers"),
                                                                                  encoder_inputs)

        # Latent space layers
        z_mean = layers.Dense(
            units=z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            name="z_mean")(stacked_bidirectional_lstm_layer)
        z_log_var = layers.Dense(
            units=z_size,
            activation="softplus",
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            bias_initializer="zeros",
            name="z_log_var")(stacked_bidirectional_lstm_layer)
        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

        self._model = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def summary(self):
        self._model.summary()
