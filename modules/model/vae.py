""" TODO - Module DOCUMENTATION """
from modules.model.base import VAE
import tensorflow_probability as tfp
import tensorflow._api.v2.compat.v1 as tf
import tf_slim
import logging

tf_dist = tfp.distributions


class MyVAE(VAE):
    """ Our implementation of MusicVAE model - fully compatible with Magenta Music VAE"""

    def __init__(self, encoder, decoder):
        """Initializer for a MusicVAE model.
  
        Args:
          encoder: A BaseEncoder implementation class to use.
          decoder: A BaseDecoder implementation class to use.
        """
        self.global_step = None
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

    def build(self, hparams, output_depth, is_training):
        """Builds encoder and decoder.
  
        Must be called within a graph.
  
        Args:
          hparams: An HParams object containing model hyperparameters. See
              `get_default_hparams` below for required values.
          output_depth: Size of final output dimension.
          is_training: Whether or not the model will be used for training.
        """
        logging.info('Building MusicVAE model with %s, %s, and hparams:\n%s',
                     self.encoder.__class__.__name__,
                     self.decoder.__class__.__name__, hparams.values())
        self.global_step = tf.train.get_or_create_global_step()
        self._hparams = hparams
        self._encoder.build(hparams, is_training)
        self._decoder.build(hparams, output_depth, is_training)

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
        hparams = self.hparams
        z_size = hparams.z_size

        sequence = tf.cast(sequence, tf.float32)
        if control_sequence is not None:
            control_sequence = tf.cast(control_sequence, tf.float32)
            sequence = tf.concat([sequence, control_sequence], axis=-1)
        encoder_output = self.encoder.encode(sequence, sequence_length)

        mu = tf.layers.dense(
            encoder_output,
            z_size,
            name='encoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001))
        sigma = tf.layers.dense(
            encoder_output,
            z_size,
            activation=tf.nn.softplus,
            name='encoder/sigma',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001))

        return tf_dist.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def _compute_model_loss(
        self, input_sequence, output_sequence, sequence_length, control_sequence):
        """Builds a model with loss for train/eval."""
        hparams = self.hparams
        batch_size = hparams.batch_size

        input_sequence = tf.cast(input_sequence, tf.float32)
        output_sequence = tf.cast(output_sequence, tf.float32)

        max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)

        input_sequence = input_sequence[:, :max_seq_len]

        if control_sequence is not None:
            control_depth = control_sequence.shape[-1]
            control_sequence = tf.cast(control_sequence, tf.float32)
            control_sequence = control_sequence[:, :max_seq_len]
            # Shouldn't be necessary, but the slice loses shape information when
            # control depth is zero.
            control_sequence.set_shape([batch_size, None, control_depth])

        # The target/expected outputs.
        x_target = output_sequence[:, :max_seq_len]
        # Inputs to be fed to decoder, including zero padding for the initial input.
        x_input = tf.pad(output_sequence[:, :max_seq_len - 1],
                         [(0, 0), (1, 0), (0, 0)])
        x_length = tf.minimum(sequence_length, max_seq_len)

        # Either encode to get `z`, or do unconditional, decoder-only.
        if hparams.z_size:  # vae mode:
            q_z = self.encode(input_sequence, x_length, control_sequence)
            z = q_z.sample()

            # Prior distribution.
            p_z = tf_dist.MultivariateNormalDiag(
                loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)

            # KL Divergence (nats)
            kl_div = tf_dist.kl_divergence(q_z, p_z)

            # Concatenate the Z vectors to the inputs at each time step.
        else:  # unconditional, decoder-only generation
            kl_div = tf.zeros([batch_size, 1], dtype=tf.float32)
            z = None

        r_loss, metric_map = self.decoder.reconstruction_loss(
            x_input, x_target, x_length, z, control_sequence)[0:2]

        free_nats = hparams.free_bits * tf.math.log(2.0)
        kl_cost = tf.maximum(kl_div - free_nats, 0)

        g_step = tf.cast(self.global_step, tf.float32)
        beta = ((1.0 - tf.pow(hparams.beta_rate, g_step)) * hparams.max_beta)
        self.loss = tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost)

        scalars_to_summarize = {
            'loss': self.loss,
            'losses/r_loss': r_loss,
            'losses/kl_loss': kl_cost,
            'losses/kl_bits': kl_div / tf.math.log(2.0),
            'losses/kl_beta': beta,
        }
        return metric_map, scalars_to_summarize

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

        _, scalars_to_summarize = self._compute_model_loss(
            input_sequence, output_sequence, sequence_length, control_sequence)

        hparams = self.hparams
        g_step = tf.cast(self.global_step, tf.float32)
        lr = ((hparams.learning_rate - hparams.min_learning_rate) *
              tf.pow(hparams.decay_rate, g_step) +
              hparams.min_learning_rate)

        optimizer = tf.train.AdamOptimizer(lr)

        tf.summary.scalar('learning_rate', lr)
        for n, t in scalars_to_summarize.items():
            tf.summary.scalar(n, tf.reduce_mean(t))

        return optimizer

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
        metric_map, scalars_to_summarize = self._compute_model_loss(
            input_sequence, output_sequence, sequence_length, control_sequence)

        for n, t in scalars_to_summarize.items():
            metric_map[n] = tf.metrics.mean(t)

        metrics_to_values, metrics_to_updates = (
            tf_slim.metrics.aggregate_metric_map(metric_map))

        for metric_name, metric_value in metrics_to_values.items():
            tf.summary.scalar(metric_name, metric_value)

        return list(metrics_to_updates.values())

    def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
        """Sample with an optional conditional embedding `z`."""
        if z is not None and int(z.shape[0]) != n:
            raise ValueError(
                '`z` must have a first dimension that equals `n` when given. '
                'Got: %d vs %d' % (z.shape[0], n))

        if self.hparams.z_size and z is None:
            logging.warning(
                'Sampling from conditional model without `z`. Using random `z`.')
            normal_shape = [n, self.hparams.z_size]
            normal_dist = tfp.distributions.Normal(
                loc=tf.zeros(normal_shape), scale=tf.ones(normal_shape))
            z = normal_dist.sample()

        return self.decoder.sample(n, max_length, z, c_input, **kwargs)
