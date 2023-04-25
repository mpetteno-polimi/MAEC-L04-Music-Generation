# TODO - DOC

from keras import models, layers, losses
from keras import backend as K


class HierarchicalDecoder(object):

    def __init__(self):
        self._model = None

    def build(self, z_size, cnn_embedding_length):
        decoder_input_dim = z_size + cnn_embedding_length
        latent_inputs = layers.Input(shape=(decoder_input_dim,), name="decoder_input")

        # TODO - Translate to LSTM decoder
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

        self._model = models.Model(latent_inputs, decoder_outputs, name="decoder")

    def decode(self, input_):
        return self._model(input_)

    def reconstruction_loss(self, input_, output_):
        return 28 * 28 * losses.binary_crossentropy(K.flatten(input_), K.flatten(output_))

    def summary(self):
        self._model.summary()
