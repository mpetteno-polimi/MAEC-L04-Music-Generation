import abc


class Encoder(abc.ABC):
    """Abstract encoder class.

      Implementations must define the following abstract methods:
       -`build`
       -`encode`
    """

    @abc.abstractmethod
    def build(self, hparams):
        """Builder method for BaseEncoder.

        Args:
          hparams: An HParams object containing model hyperparameters.
        """
        pass

    @abc.abstractmethod
    def encode(self, sample):
        """Encodes input sample from a latent space into a sequences of tuples representing midi notes.

        Args:
           sample: sample from a latent space.

        Returns:
           outputs: Raw outputs to parameterize the prior distribution in VAE.encode, sized `[batch_size, N]`.
        """
        pass


