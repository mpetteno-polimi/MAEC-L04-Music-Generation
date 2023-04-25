# TODO - DOC

import keras
from keras import models, layers
from keras import backend as K


class BidirectionalLstmEncoder(object):

    def __init__(self):
        self._model = None

    def encode(self, input_):
        return self._model(input_)

    def build(self, seq_length, frame_length, z_size):

        def sampling(mu_log_variance):
            mu, log_variance = mu_log_variance
            epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
            random_sample = mu + K.exp(log_variance / 2) * epsilon
            return random_sample

        encoder_inputs = keras.Input(shape=(seq_length, frame_length), name="encoder_input")

        # TODO - Translate to LSTM
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)

        # Latent space layers
        z_mean = layers.Dense(z_size, name="z_mean")(x)
        z_log_var = layers.Dense(z_size, name="z_log_var")(x)
        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

        self._model = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def summary(self):
        self._model.summary()

