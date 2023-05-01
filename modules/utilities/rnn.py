# TODO - DOC

from keras import layers

from definitions import ConfigSections
from modules.utilities import config

model_config = config.load_configuration_section(ConfigSections.MODEL)


def build_lstm_layers(layers_sizes, name,
                      bidirectional=False, return_sequences=False, return_state=False, stateful=False):
    rnn_cells = [layers.LSTMCell(
        units=layer_size,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=model_config.get("dropout"),
        recurrent_dropout=0.0) for layer_idx, layer_size in enumerate(layers_sizes)]
    rnn_layer = layers.RNN(
        cell=rnn_cells,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=False,
        stateful=stateful,
        time_major=False,
        unroll=False,
        name=name
    )
    return layers.Bidirectional(layer=rnn_layer, merge_mode="concat") if bidirectional else rnn_layer
