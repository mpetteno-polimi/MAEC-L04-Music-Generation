""" TODO - Module DOCUMENTATION """

class VAE(object):

    def __init__(self, encoder, decoder):
        """Initializer for a Variational Autoencoder model.

        Args:
          encoder: An Encoder implementation class.
          decoder: A Decoder implementation class.
        """
        self._hparams = None
        self._encoder = encoder
        self._decoder = decoder

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def hparams(self):
        return self._hparams

    def build(self, hparams, output_depth):
        """Builds encoder and decoder.

        Args:
          hparams: An HParams object containing model hyperparameters. see the Hyperparameters class
          output_depth: Size of final output dimension.
        """

        self._hparams = hparams
        self._encoder.build(hparams)
        self._decoder.build(hparams, output_depth)
        return

    def encode(self, sequence, sequence_length, control_sequence=None):
        return

    def train(self, input_sequence, output_sequence, sequence_length, control_sequence=None):
        return

    def eval(self, input_sequence, output_sequence, sequence_length, control_sequence=None):
        return

    def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
        return

