import abc
from abc import ABC


class VariationalDecoder(ABC):

    @abc.abstractmethod
    def build(self, hparams):
        """Builder method for BaseDecoder.

        Args:
          hparams: An HParams object containing model hyperparameters.
          hparams: Hyperparameters object containing model hyperparameters used to tune the Decoder build.
        """
        pass

    @abc.abstractmethod
    def decode(self, sample, hparams):
        """Decodes Sequence from latent space sample to MIDI sequence.

        Args:
          sample: sample obtained from latent space
          hparams: An Hyperparameters object containing model hyperparameters.
        """

    @abc.abstractmethod
    def reconstruction_loss(self, x_input, x_target):
        """Reconstruction loss calculation.

        Args:
          x_input: Batch of decoder input sequences for teacher forcing, sized
              `[batch_size, max(x_length), output_depth]`.
          x_target: Batch of expected output sequences to compute loss against,
              sized `[batch_size, max(x_length), output_depth]`.

        Returns:
          r_loss: The reconstruction loss for each sequence in the batch.
        """
        pass

