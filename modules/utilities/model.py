# TODO - DOC

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


def build_lstm_layer(lstm_size, return_sequences=False):
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
        return_state=False,
        go_backwards=False,
        stateful=False,
        time_major=False,
        unroll=False
    )


def build_bidirectional_lstm_layer(lstm_size, return_sequences=False):
    return layers.Bidirectional(
        layer=build_lstm_layer(lstm_size, return_sequences=return_sequences),
        merge_mode="concat"
    )


def build_stacked_rnn_layers(layers_sizes, type="lstm"):
    assert(len(layers_sizes) > 0)

    if type == "lstm":
        rnn_build_fn = build_lstm_layer
    elif type == "bidirectional_lstm":
        rnn_build_fn = build_bidirectional_lstm_layer
    else:
        raise ValueError("RNN Layer of type {} not supporter".format(type))

    rnn_layers = []
    for i, layer_size in enumerate(layers_sizes):
        is_last_layer = (i == len(layers_sizes) - 1)
        curr_layer = rnn_build_fn(layer_size, return_sequences=not is_last_layer)
        rnn_layers.append(curr_layer)

    return rnn_layers


def call_stacked_rnn_layers(inputs, rnn_layers, initial_cell_state=None):
    output = rnn_layers[0](inputs, initial_state=initial_cell_state)
    for i, bidirectional_lstm_layer in enumerate(rnn_layers, start=1):
        output = rnn_layers[i](output)

    return output


def initial_cell_state_from_embedding(layer_size, embedding):
    initial_hidden_state = layers.Dense(
        units=layer_size,
        activation='tanh',
        use_bias=True,
        kernel_initializer=initializers.RandomNormal(stddev=0.001),
        name="z_to_initial_state"
    )(embedding)
    initial_cell_state = K.zeros(layer_size)
    return initial_hidden_state, initial_cell_state
