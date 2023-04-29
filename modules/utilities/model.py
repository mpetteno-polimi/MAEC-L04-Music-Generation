# TODO - DOC

import numpy as np
from keras import layers
from keras import backend as K
from keras.initializers import initializers

from definitions import ConfigSections
from modules.utilities import config

model_config = config.load_configuration_section(ConfigSections.MODEL)


def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + K.exp(log_variance / 2) * epsilon
    return random_sample


def build_lstm_layer(lstm_size, return_sequences=False, return_state=False, stateful=False):
    return layers.LSTM(
        units=lstm_size,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        dropout=model_config.get("dropout"),
        recurrent_dropout=0.0,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=False,
        stateful=stateful,
        time_major=False,
        unroll=False
    )


def build_bidirectional_lstm_layer(lstm_size, return_sequences=False, return_state=False, stateful=True):
    return layers.Bidirectional(
        layer=build_lstm_layer(lstm_size, return_sequences=return_sequences),
        merge_mode="concat"
    )


def build_stacked_rnn_layers(layers_sizes, type_="lstm", return_sequences=True, return_state=False, stateful=True):
    assert (len(layers_sizes) > 0)

    if type_ == "lstm":
        rnn_build_fn = build_lstm_layer
    elif type_ == "bidirectional_lstm":
        rnn_build_fn = build_bidirectional_lstm_layer
    else:
        raise ValueError("RNN Layer of type {} not supporter".format(type_))

    rnn_layers = []
    for i, layer_size in enumerate(layers_sizes):
        is_last_layer = (i == len(layers_sizes) - 1)
        curr_layer = rnn_build_fn(lstm_size=layer_size,
                                  return_sequences=not is_last_layer or return_sequences,
                                  return_state=return_state,
                                  stateful=stateful)
        rnn_layers.append(curr_layer)

    return rnn_layers


def call_stacked_rnn_layers(inputs, rnn_layers, initial_cell_states=None, training=True):
    output = rnn_layers[0](inputs, initial_state=initial_cell_states[0])
    for i, bidirectional_lstm_layer in enumerate(rnn_layers, start=1):
        output = rnn_layers[i](output, initial_cell_state=initial_cell_states[i], training=training)

    return output


def initial_cell_states_from_embedding(layers_sizes, embedding):
    def split(cell_states):
        return np.split(ary=cell_states, indices_or_sections=flatten_state_sizes, axis=1)

    def pack(cell_states):
        return list(zip(cell_states, cell_states[1:]))

    cell_state_sizes = [(layer_size / 2, layer_size / 2) for layer_size in layers_sizes]
    flatten_state_sizes = K.flatten(cell_state_sizes)
    initial_cell_states = layers.Dense(
        units=sum(flatten_state_sizes),
        activation='tanh',
        use_bias=True,
        kernel_initializer=initializers.RandomNormal(stddev=0.001),
        name="z_to_initial_state"
    )(embedding)
    split_initial_cell_states = layers.Lambda(split, name='initial_state_split')(initial_cell_states)
    packed_initial_cell_states = layers.Lambda(pack, name='initial_state_pack')(split_initial_cell_states)
    return packed_initial_cell_states
