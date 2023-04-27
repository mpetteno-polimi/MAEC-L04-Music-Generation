# TODO - DOC

from keras import models, layers, losses
from keras import backend as K
from keras.initializers import initializers

from definitions import ConfigSections
from modules import utilities


class HierarchicalDecoder(layers.Layer):

    def __init__(self, name="decoder", **kwargs):
        super(HierarchicalDecoder, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)

        def initial_cell_state_from_embedding(embedding):
            initial_hidden_state = layers.Dense(
                units=lstm_dec_size,
                activation='tanh',
                use_bias=True,
                kernel_initializer=initializers.RandomNormal(stddev=0.001),
                name="z_to_initial_state"
            )(embedding)
            initial_cell_state = K.zeros(lstm_dec_size)
            return [initial_hidden_state, initial_cell_state]

        lstm_dec_size = self._model_config.get("dec_rnn_size")
        lstm_dec_layers = self._model_config.get("dec_layers")

        # Input layer
        decoder_input = layers.Input(shape=(input_shape,), name="decoder_input")

        # Conductor layer - Stacked LSTM
        conductor_output = utilities.model.build_stacked_lstm_layer(
            num_layers=lstm_dec_layers,
            input_layer=decoder_input,
            initial_state=initial_cell_state_from_embedding(decoder_input),
            lstm_size=lstm_dec_size
        )

        # Core decoder
        decoder_output = utilities.model.build_stacked_lstm_layer(
            num_layers=lstm_dec_layers,
            input_layer=decoder_input,
            initial_state=initial_cell_state_from_embedding(decoder_input),
            lstm_size=lstm_dec_size
        )

        self._model = models.Model(decoder_input, decoder_output, name="decoder")

    def reconstruction_loss(self, input_, output_):
        return 28 * 28 * losses.binary_crossentropy(K.flatten(input_), K.flatten(output_))














self._output_layer = layers.Dense(output_depth, name='output_projection')

self._hier_cells = [
    lstm_utils.rnn_cell(
        hparams.dec_rnn_size,
        dropout_keep_prob=hparams.dropout_keep_prob,
        residual=hparams.residual_decoder)
    # Subtract 1 for the core decoder level
    for _ in range(len(self._level_lengths) - 1)]


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

        state = initial_cell_state_from_embedding(self._hier_cells[level], initial_input, name='initial_state')

        # The initial input should be the same size as the tensors returned by next level.
        if level == num_levels - 1:
            input_size = sum(tf.nest.flatten(self._core_decoder.state_size))
        else:
            input_size = sum(tf.nest.flatten(self._hier_cells[level + 1].state_size))
        next_input = tf.zeros([batch_size, input_size])

        lower_level_embeddings = []
        for i in range(num_steps):
            next_input = tf.concat([next_input, initial_input], axis=1)
            output, state = self._hier_cells[level](next_input, state)
            next_input = base_decode_fn(output, self._level_lengths[1])
            lower_level_embeddings.append(next_input)

        return tf.concat(tf.nest.flatten(state), axis=-1)

    return recursive_decode(z)


def initial_cell_state_from_embedding(cell, z, name=None):
    """Computes an initial RNN `cell` state from an embedding, `z`."""
    flat_state_sizes = tf.nest.flatten(cell.state_size)
    return tf.nest.pack_sequence_as(
        cell.zero_state(batch_size=z.shape[0], dtype=tf.float32),
        tf.split(
            tf.layers.dense(
                z,
                sum(flat_state_sizes),
                activation=tf.tanh,
                kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                name=name),
            flat_state_sizes,
            axis=1))

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
    hier_input = self._reshape_to_hierarchy(x_input)
    hier_target = self._reshape_to_hierarchy(x_target)

    loss_outputs = []

    def base_train_fn(embedding, hier_index):
        """Base function for training hierarchical decoder."""
        split_input = hier_input[hier_index]
        split_target = hier_target[hier_index]

        res = self._core_decoder.reconstruction_loss(split_input, split_target, embedding)
        loss_outputs.append(res)
        decode_results = res[-1]
        return tf.concat(tf.nest.flatten(decode_results.final_state), axis=-1)

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