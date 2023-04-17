from abc import ABC, abstractmethod
from magenta.models.music_vae.configs import HParams


class Model(ABC):
    @property
    @abstractmethod
    def hparams(self) -> HParams:
        pass

    @abstractmethod
    def build(self, hparams, output_depth, is_training):
        """Builds encoder and decoder.

    Must be called within a graph (tf1).

    Args:
      hparams: An HParams object containing model hyperparameters. See
          `get_default_hparams` below for required values.
      output_depth: Size of final output dimension.
      is_training: Whether the model will be used for training.
    """
    pass


class VAE(Model):
    """Variational Autoencoder satisfying MusicVAE compatible"""
    def __init__(self, encoder, decoder):
        """Initializer for a MusicVAE model.

    Args:
      encoder: A BaseEncoder implementation class to use.
      decoder: A BaseDecoder implementation class to use.
    """
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

    @abstractmethod
    def build(self, hparams, output_depth, is_training):
        """Builds encoder and decoder.

    Must be called within a graph.

    Args:
      hparams: An HParams object containing model hyperparameters. See
          `get_default_hparams` below for required values.
      output_depth: Size of final output dimension.
      is_training: Whether the model will be used for training.
    """
    pass

    @abstractmethod
    def encode(self, sequence, sequence_length, control_sequence=None):
        """Encodes input sequences into a MultivariateNormalDiag distribution.

    Args:
      sequence: A Tensor with shape `[num_sequences, max_length, input_depth]`
          containing the sequences to encode.
      sequence_length: The length of each sequence in the `sequence` Tensor.
      control_sequence: (Optional) A Tensor with shape
          `[num_sequences, max_length, control_depth]` containing control
          sequences on which to condition. These will be concatenated depthwise
          to the input sequences.

    Returns:
      A tfp.distributions.MultivariateNormalDiag representing the posterior
      distribution for each sequence.
    """
        pass

    @abstractmethod
    def _compute_model_loss(
        self, input_sequence, output_sequence, sequence_length, control_sequence):
        """Builds a model with loss for train/eval."""

    @abstractmethod
    def train(self, input_sequence, output_sequence, sequence_length,
              control_sequence=None):
        """Train on the given sequences, returning an optimizer.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
          identical).
      control_sequence: (Optional) sequence on which to condition. This will be
          concatenated depthwise to the model inputs for both encoding and
          decoding.

    Returns:
      optimizer: A tf.train.Optimizer.
    """

    @abstractmethod
    def eval(self, input_sequence, output_sequence, sequence_length,
             control_sequence=None):
        """Evaluate on the given sequences, returning metric update ops.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
        identical).
      control_sequence: (Optional) sequence on which to condition the decoder.

    Returns:
      metric_update_ops: tf.metrics update ops.
    """

    @abstractmethod
    def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
        """Sample with an optional conditional embedding `z`."""
