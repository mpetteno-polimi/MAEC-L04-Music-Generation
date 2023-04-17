from magenta.models.music_vae import base_model
from magenta.models.music_vae import lstm_utils
import tensorflow._api.v2.compat.v1 as tf
import magenta.contrib.training as contrib_training
import numpy as np

class MyHierarchicalLstmDecoder(base_model.BaseDecoder):
  """Hierarchical LSTM decoder."""

  def __init__(self,
               core_decoder,
               level_lengths,
               disable_autoregression=False,
               hierarchical_encoder=None):
    """Initializer for HierarchicalLstmDecoder.

    Hierarchicaly decodes a sequence across time.

    Each sequence is padded per-segment. For example, a sequence with
    three segments [1, 2, 3], [4, 5], [6, 7, 8 ,9] and a `max_seq_len` of 12
    is represented as `sequence = [1, 2, 3, 0, 4, 5, 0, 0, 6, 7, 8, 9]` with
    `sequence_length = [3, 2, 4]`.

    `z` initializes the first level LSTM to produce embeddings used to
    initialize the states of LSTMs at subsequent levels. The lowest-level
    embeddings are then passed to the given `core_decoder` to generate the
    final outputs.

    This decoder has 3 modes for what is used as the inputs to the LSTMs
    (excluding those in the core decoder):
      Autoregressive: (default) The inputs to the level `l` decoder are the
        final states of the level `l+1` decoder.
      Non-autoregressive: (`disable_autoregression=True`) The inputs to the
        hierarchical decoders are 0's.
      Re-encoder: (`hierarchical_encoder` provided) The inputs to the level `l`
        decoder are re-encoded outputs of level `l+1`, using the given encoder's
        matching level.

    Args:
      core_decoder: The BaseDecoder implementation to use at the output level.
      level_lengths: A list of the number of outputs of each level of the
        hierarchy. The final level is the (padded) maximum length. The product
        of the lengths must equal `hparams.max_seq_len`.
      disable_autoregression: Whether to disable the autoregression within the
        hierarchy. May also be a collection of levels on which to disable.
      hierarchical_encoder: (Optional) A HierarchicalLstmEncoder instance to use
        for re-encoding the decoder outputs at each level for use as inputs to
        the next level up in the hierarchy, instead of the final decoder state.
        The encoder level output lengths (except for the final single-output
        level) should be the reverse of `level_output_lengths`.

    Raises:
      ValueError: If `hierarchical_encoder` is given but has incompatible level
        lengths.
    """
    # Check for explicit True/False since lists may be given.
    if disable_autoregression is True:  # pylint:disable=g-bool-id-comparison
      disable_autoregression = list(range(len(level_lengths)))
    elif disable_autoregression is False:  # pylint:disable=g-bool-id-comparison
      disable_autoregression = []
    if (hierarchical_encoder and
        (tuple(hierarchical_encoder.level_lengths[-1::-1]) !=
         tuple(level_lengths))):
      raise ValueError(
          'Incompatible hierarchical encoder level output lengths: ',
          hierarchical_encoder.level_lengths, level_lengths)

    self._core_decoder = core_decoder
    self._level_lengths = level_lengths
    self._disable_autoregression = disable_autoregression
    self._hierarchical_encoder = hierarchical_encoder

  def build(self, hparams, output_depth, is_training=True):
    self.hparams = hparams
    self._output_depth = output_depth
    self._total_length = hparams.max_seq_len
    if self._total_length != np.prod(self._level_lengths):
      raise ValueError(
          'The product of the HierarchicalLstmDecoder level lengths (%d) must '
          'equal the padded input sequence length (%d).' % (
              np.prod(self._level_lengths), self._total_length))
    tf.logging.info('\nHierarchical Decoder:\n'
                    '  input length: %d\n'
                    '  level output lengths: %s\n',
                    self._total_length,
                    self._level_lengths)

    self._hier_cells = [
        lstm_utils.rnn_cell(
            hparams.dec_rnn_size,
            dropout_keep_prob=hparams.dropout_keep_prob,
            residual=hparams.residual_decoder)
        # Subtract 1 for the core decoder level
        for _ in range(len(self._level_lengths) - 1)]

    with tf.variable_scope('core_decoder', reuse=tf.AUTO_REUSE):
      self._core_decoder.build(hparams, output_depth, is_training)

  @property
  def state_size(self):
    return self._core_decoder.state_size

  def _merge_decode_results(self, decode_results):
    """Merge across time."""
    assert decode_results
    time_axis = 1
    zipped_results = lstm_utils.LstmDecodeResults(*list(zip(*decode_results)))
    if zipped_results.rnn_output[0] is None:
      rnn_output = None
      rnn_input = None
    else:
      rnn_output = tf.concat(zipped_results.rnn_output, axis=time_axis)
      rnn_input = tf.concat(zipped_results.rnn_input, axis=time_axis)
    return lstm_utils.LstmDecodeResults(
        rnn_output=rnn_output,
        rnn_input=rnn_input,
        samples=tf.concat(zipped_results.samples, axis=time_axis),
        final_state=zipped_results.final_state[-1],
        final_sequence_lengths=tf.stack(
            zipped_results.final_sequence_lengths, axis=time_axis))

  def _hierarchical_decode(self, z, base_decode_fn):
    """Depth first decoding from `z`, passing final embeddings to base fn."""
    batch_size = z.shape[0]
    # Subtract 1 for the core decoder level.
    num_levels = len(self._level_lengths) - 1

    hparams = self.hparams
    batch_size = hparams.batch_size

    def recursive_decode(initial_input, path=None):
      """Recursive hierarchical decode function."""
      path = path or []
      level = len(path)

      if level == num_levels:
        with tf.variable_scope('core_decoder', reuse=tf.AUTO_REUSE):
          return base_decode_fn(initial_input, path)

      scope = tf.VariableScope(
          tf.AUTO_REUSE, 'decoder/hierarchical_level_%d' % level)
      num_steps = self._level_lengths[level]
      with tf.variable_scope(scope):
        state = lstm_utils.initial_cell_state_from_embedding(
            self._hier_cells[level], initial_input, name='initial_state')
      if level not in self._disable_autoregression:
        # The initial input should be the same size as the tensors returned by
        # next level.
        if self._hierarchical_encoder:
          input_size = self._hierarchical_encoder.level(0).output_depth
        elif level == num_levels - 1:
          input_size = sum(tf.nest.flatten(self._core_decoder.state_size))
        else:
          input_size = sum(
              tf.nest.flatten(self._hier_cells[level + 1].state_size))
        next_input = tf.zeros([batch_size, input_size])
      lower_level_embeddings = []
      for i in range(num_steps):
        if level in self._disable_autoregression:
          next_input = tf.zeros([batch_size, 1])
        else:
          next_input = tf.concat([next_input, initial_input], axis=1)
        with tf.variable_scope(scope):
          output, state = self._hier_cells[level](next_input, state, scope)
        next_input = recursive_decode(output, path + [i])
        lower_level_embeddings.append(next_input)
      if self._hierarchical_encoder:
        # Return the encoding of the outputs using the appropriate level of the
        # hierarchical encoder.
        enc_level = num_levels - level
        return self._hierarchical_encoder.level(enc_level).encode(
            sequence=tf.stack(lower_level_embeddings, axis=1),
            sequence_length=tf.fill([batch_size], num_steps))
      else:
        # Return the final state.
        return tf.concat(tf.nest.flatten(state), axis=-1)

    return recursive_decode(z)

  def _reshape_to_hierarchy(self, t):
    """Reshapes `t` so that its initial dimensions match the hierarchy."""
    # Exclude the final, core decoder length.
    level_lengths = self._level_lengths[:-1]
    t_shape = t.shape.as_list()
    t_rank = len(t_shape)
    batch_size = t_shape[0]
    hier_shape = [batch_size] + level_lengths
    if t_rank == 3:
      hier_shape += [-1] + t_shape[2:]
    elif t_rank != 2:
      # We only expect rank-2 for lengths and rank-3 for sequences.
      raise ValueError('Unexpected shape for tensor: %s' % t)
    hier_t = tf.reshape(t, hier_shape)
    # Move the batch dimension to after the hierarchical dimensions.
    num_levels = len(level_lengths)
    perm = list(range(len(hier_shape)))
    perm.insert(num_levels, perm.pop(0))
    return tf.transpose(hier_t, perm)

  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences of concatenated segmeents for
        teacher forcing, sized `[batch_size, max_seq_len, output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
        sized `[batch_size, max_seq_len, output_depth]`.
      x_length: Length of input/output sequences, sized
        `[batch_size, level_lengths[0]]` or `[batch_size]`. If the latter,
        each length must either equal `max_seq_len` or 0. In this case, the
        segment lengths are assumed to be constant and the total length will be
        evenly divided amongst the segments.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
        `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
        `[batch_size, max_seq_len, control_depth]`. Required if conditioning on
        control sequences.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
      decode_results: The LstmDecodeResults.

    Raises:
      ValueError: If `c_input` is provided in re-encoder mode.
    """
    if self._hierarchical_encoder and c_input is not None:
      raise ValueError(
          'Re-encoder mode unsupported when conditioning on controls.')

    batch_size = int(x_input.shape[0])

    x_length = lstm_utils.maybe_split_sequence_lengths(
        x_length, np.prod(self._level_lengths[:-1]), self._total_length)

    hier_input = self._reshape_to_hierarchy(x_input)
    hier_target = self._reshape_to_hierarchy(x_target)
    hier_length = self._reshape_to_hierarchy(x_length)
    hier_control = (
        self._reshape_to_hierarchy(c_input) if c_input is not None else None)

    loss_outputs = []

    def base_train_fn(embedding, hier_index):
      """Base function for training hierarchical decoder."""
      split_size = self._level_lengths[-1]
      split_input = hier_input[hier_index]
      split_target = hier_target[hier_index]
      split_length = hier_length[hier_index]
      split_control = (
          hier_control[hier_index] if hier_control is not None else None)

      res = self._core_decoder.reconstruction_loss(
          split_input, split_target, split_length, embedding, split_control)
      loss_outputs.append(res)
      decode_results = res[-1]

      if self._hierarchical_encoder:
        # Get the approximate "sample" from the model.
        # Start with the inputs the RNN saw (excluding the start token).
        samples = decode_results.rnn_input[:, 1:]
        # Pad to be the max length.
        samples = tf.pad(
            samples,
            [(0, 0), (0, split_size - tf.shape(samples)[1]), (0, 0)])
        samples.set_shape([batch_size, split_size, self._output_depth])
        # Set the final value based on the target, since the scheduled sampling
        # helper does not sample the final value.
        samples = lstm_utils.set_final(
            samples,
            split_length,
            lstm_utils.get_final(split_target, split_length, time_major=False),
            time_major=False)
        # Return the re-encoded sample.
        return self._hierarchical_encoder.level(0).encode(
            sequence=samples,
            sequence_length=split_length)
      elif self._disable_autoregression:
        return None
      else:
        return tf.concat(tf.nest.flatten(decode_results.final_state), axis=-1)

    z = tf.zeros([batch_size, 0]) if z is None else z
    self._hierarchical_decode(z, base_train_fn)

    # Accumulate the split sequence losses.
    r_losses, metric_maps, decode_results = list(zip(*loss_outputs))

    # Merge the metric maps by passing through renamed values and taking the
    # mean across the splits.
    merged_metric_map = {}
    for metric_name in metric_maps[0]:
      metric_values = []
      for i, m in enumerate(metric_maps):
        merged_metric_map['segment/%03d/%s' % (i, metric_name)] = m[metric_name]
        metric_values.append(m[metric_name][0])
      merged_metric_map[metric_name] = (
          tf.reduce_mean(metric_values), tf.no_op())

    return (tf.reduce_sum(r_losses, axis=0),
            merged_metric_map,
            self._merge_decode_results(decode_results))

  def sample(self, n, max_length=None, z=None, c_input=None,
             **core_sampler_kwargs):
    """Sample from decoder with an optional conditional latent vector `z`.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) maximum total length of samples. If given, must
        match `hparams.max_seq_len`.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      **core_sampler_kwargs: (Optional) Additional keyword arguments to pass to
        core sampler.
    Returns:
      samples: Sampled sequences with concenated, possibly padded segments.
         Sized `[n, max_length, output_depth]`.
      decoder_results: The merged LstmDecodeResults from sampling.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`,
        or if `c_input` is provided in re-encoder mode.
    """
    if z is not None and int(z.shape[0]) != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0], n))
    z = tf.zeros([n, 0]) if z is None else z

    if self._hierarchical_encoder and c_input is not None:
      raise ValueError(
          'Re-encoder mode unsupported when conditioning on controls.')

    if max_length is not None:
      with tf.control_dependencies([
          tf.assert_equal(
              max_length, self._total_length,
              message='`max_length` must equal `hparams.max_seq_len` if given.')
      ]):
        max_length = tf.identity(max_length)

    if c_input is not None:
      # Reshape control sequence to hierarchy.
      c_input = tf.squeeze(
          self._reshape_to_hierarchy(tf.expand_dims(c_input, 0)),
          axis=len(self._level_lengths) - 1)

    core_max_length = self._level_lengths[-1]
    all_samples = []
    all_decode_results = []

    def base_sample_fn(embedding, hier_index):
      """Base function for sampling hierarchical decoder."""
      samples, decode_results = self._core_decoder.sample(
          n,
          max_length=core_max_length,
          z=embedding,
          c_input=c_input[hier_index] if c_input is not None else None,
          start_inputs=all_samples[-1][:, -1] if all_samples else None,
          **core_sampler_kwargs)
      all_samples.append(samples)
      all_decode_results.append(decode_results)
      if self._hierarchical_encoder:
        return self._hierarchical_encoder.level(0).encode(
            samples,
            decode_results.final_sequence_lengths)
      else:
        return tf.concat(tf.nest.flatten(decode_results.final_state), axis=-1)

    # Populate `all_sample_ids`.
    self._hierarchical_decode(z, base_sample_fn)

    all_samples = tf.concat(
        [tf.pad(s, [(0, 0), (0, core_max_length - tf.shape(s)[1]), (0, 0)])
         for s in all_samples],
        axis=1)
    return all_samples, self._merge_decode_results(all_decode_results)


def get_default_hparams():
  """Returns copy of default HParams for LSTM models."""
  hparams_map = base_model.get_default_hparams().values()
  hparams_map.update({
      'conditional': True,
      'dec_rnn_size': [512],  # Decoder RNN: number of units per layer.
      'enc_rnn_size': [256],  # Encoder RNN: number of units per layer per dir.
      'dropout_keep_prob': 1.0,  # Probability all dropout keep.
      'sampling_schedule': 'constant',  # constant, exponential, inverse_sigmoid
      'sampling_rate': 0.0,  # Interpretation is based on `sampling_schedule`.
      'use_cudnn': False,  # DEPRECATED
      'residual_encoder': False,  # Use residual connections in encoder.
      'residual_decoder': False,  # Use residual connections in decoder.
      'control_preprocessing_rnn_size': [256],  # Decoder control preprocessing.
  })
  return contrib_training.HParams(**hparams_map)
