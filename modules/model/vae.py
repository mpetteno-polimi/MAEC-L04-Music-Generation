""" TODO - Module DOCUMENTATION """
from model.decoders.lstm import LstmDecoder
from model.encoder.lstm import LstmEncoder
from model.sampler.sampler import StdSampler

from tensorflow import keras as ks


class VAE(ks.Model):
    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def hparams(self):
        return self._hparams

    def __init__(self, encoder=LstmEncoder, decoder=LstmDecoder, sampler=StdSampler, *args, **kwargs):
        """Initializer for a Variational Autoencoder model.

        Args:
          encoder: An Encoder implementation class.
          decoder: A Decoder implementation class.
        """
        super().__init__(*args, **kwargs)
        self._encoder = encoder
        self._decoder = decoder
        self._sampler = sampler

        self._hparams = None

    def call(self, inputs, training=None, mask=None):
        x = self._encoder.call(inputs)

        return self._decoder.call(x)


    def train_step(self, input_sequence, output_sequence, sequence_length, control_sequence=None):
        return
