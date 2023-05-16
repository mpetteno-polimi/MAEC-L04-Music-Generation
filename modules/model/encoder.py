# TODO - DOC

import keras
from keras import layers
from keras.initializers import initializers

from definitions import ConfigSections
from modules.utilities import config, rnn, math


class BidirectionalLstmEncoder(keras.Model):
    """Maps Pianoroll inputs to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, name="bidirectional_lstm_encoder", **kwargs):
        super(BidirectionalLstmEncoder, self).__init__(name=name, **kwargs)
        self._model_config = config.load_configuration_section(ConfigSections.MODEL)
        self._layers_sizes = self._model_config.get("enc_rnn_size")

        # Init bidirectional LSTM layers
        self.bidirectional_lstm_layers = rnn.build_lstm_layers(
            layers_sizes=self._model_config.get("enc_rnn_size"),
            bidirectional=True,
            return_sequences=False,
            return_state=False,
            name="encoder"
        )

        # Init latent space layers
        z_size = self._model_config.get("z_size")

        self.dense_mean = layers.Dense(
            units=z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            name="z_mean"
        )

        self.dense_log_var = layers.Dense(
            units=z_size,
            activation=None,
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            bias_initializer="zeros",
            name="z_log_var"
        )

        self.sampling = layers.Lambda(math.sampling, name="z")

    def call(self, inputs, training=None, **kwargs):
        encoder_output = self.bidirectional_lstm_layers(inputs)
        z_mean = self.dense_mean(encoder_output)
        z_log_var = self.dense_log_var(encoder_output)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
