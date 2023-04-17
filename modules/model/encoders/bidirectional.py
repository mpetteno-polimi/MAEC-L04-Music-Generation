from magenta.models.music_vae import base_model
from magenta.models.music_vae import lstm_utils
import magenta.contrib.rnn as contrib_rnn
import tensorflow._api.v2.compat.v1 as tf


class MyBidirectionalLstmEncoder(base_model.BaseEncoder):
    """Bidirectional LSTM Encoder."""

    def __init__(self):
        self._name_or_scope = None
        self._is_training = None
        self._cells = None

    @property
    def output_depth(self):
        return self._cells[0][-1].output_size + self._cells[1][-1].output_size

    def build(self, hparams, is_training=True, name_or_scope='encoder'):
        self._is_training = is_training
        self._name_or_scope = name_or_scope
        if hparams.use_cudnn:
            tf.logging.warning('cuDNN LSTM no longer supported. Using regular LSTM.')

        tf.logging.info('\nEncoder Cells (bidirectional):\n'
                        '  units: %s\n',
                        hparams.enc_rnn_size)

        self._cells = lstm_utils.build_bidirectional_lstm(
            layer_sizes=hparams.enc_rnn_size,
            dropout_keep_prob=hparams.dropout_keep_prob,
            residual=hparams.residual_encoder,
            is_training=is_training)

    def encode(self, sequence, sequence_length):
        if self._cells is None:
            tf.logging.warning('bidirectional encoder needs its build method to be called before encoding')
        cells_fw, cells_bw = self._cells

        _, states_fw, states_bw = contrib_rnn.stack_bidirectional_dynamic_rnn(
            cells_fw,
            cells_bw,
            sequence,
            sequence_length=sequence_length,
            time_major=False,
            dtype=tf.float32,
            scope=self._name_or_scope)
        # Note we access the outputs (h) from the states since the backward
        # ouputs are reversed to the input order in the returned outputs.
        last_h_fw = states_fw[-1][-1].h
        last_h_bw = states_bw[-1][-1].h

        return tf.concat([last_h_fw, last_h_bw], 1)
