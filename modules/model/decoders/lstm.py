import logging as log
from tensorflow import keras as ks


class LstmDecoder(ks.Model):

    def __init__(self, config):  #todo: default config object
        super(LstmDecoder, self).__init__()
        """ Constructor for the Lstm decoders. """
        self.latent_inputs = ks.Input(config.LATENT_DIMENSION)
        self.dense1 = ks.layers.Dense(7 * 7 * 64, activation="relu")
        self.conv2DPitch = ks.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")
        self.conv2DStep = ks.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")
        self.conv2DDuration = ks.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")

        # Visualization
        # self.decoder_model.summary()

    def call(self, inputs, training=None, mask=None):
        keras_in = self.latent_inputs(inputs)
        x = self.dense1(keras_in)

        outputs = {
            'pitch': self.conv2DPitch(x),
            'step': self.conv2DStep(x),
            'duration': self.conv2DDuration(x)
        }

        return outputs

