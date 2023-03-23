""" TODO - Module DOCUMENTATION """
from tensorflow import keras as ks

from model.encoder.lstm import LstmEncoder


class VAE(ks.Model):
    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def __init__(self, encoder, decoder, sampler, config, *args, **kwargs):  # todo: add default enc, dec, sampler
        """Initializer for a Variational Autoencoder model.

        Args:
          encoder: An Encoder implementation class.
          decoder: A Decoder implementation class.
        """
        super().__init__(*args, **kwargs)
        self._encoder = encoder
        self._decoder = decoder
        self._sampler = sampler

    def call(self, inputs, training=None, mask=None):
        [z_mean, z_log_var] = self._encoder.call(self._encoder, inputs)
        x = self._sampler.call(self._sampler, [z_mean, z_log_var])
        return self._decoder.call(x)

    def config_for_training(self, config):
        # todo: customize implementation and parametrize
        self.compile(optimizer=ks.optimizers.Adam())

    def train(self, data, epochs, batch_size):
        # todo: customize implementation and parametrize
        self.fit(data,
                 epochs=epochs,
                 batch_size=batch_size)
