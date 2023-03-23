from tensorflow import keras as ks


class BaseDecoder(ks.Model):  # from GrooVAE Hierarchical Decoder

    def __init__(self, config):  # todo: default config object
        super(BaseDecoder, self).__init__()
        """ Constructor for the Lstm decoders. """
        self.latent_inputs = ks.Input(config.LATENT_DIMENSION)
        self.lstm = ks.layers.LSTM(config.DECODER_OUTPUT_SIZE)

    def call(self, inputs, training=None, mask=None):
        keras_in = self.latent_inputs(inputs)
        output = self.lstm(keras_in)
        return output
