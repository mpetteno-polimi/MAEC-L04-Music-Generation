import logging as log
from tensorflow import keras as ks


class HierarchicalDecoder(ks.Model):  # from GrooVAE Hierarchical Decoder

    def __init__(self, config):  # todo: default config object
        super(HierarchicalDecoder, self).__init__()
        """ Constructor for the Lstm decoders. """
        self.latent_inputs = ks.Input(config.LATENT_DIMENSION)

        # one layer linear dense layer of DECODER_DENSE_TANH_HID_SIZE * config.DECODER_NUM_LAYERS size
        self.tanh_layer1 = ks.layers.Dense(config.DECODER_DENSE_TANH_HID_SIZE * config.DECODER_NUM_LAYERS,
                                           activation=ks.activations.tanh)

        # two layers LSTM layer of config.DECODER_LSTM_SIZE size
        self.lstm_layer1 = ks.layers.LSTM(config.DECODER_LSTM_SIZE, dropout=config.DROPOUT_RATE, return_sequences=True)
        self.lstm_layer2 = ks.layers.LSTM(config.DECODER_LSTM_SIZE, dropout=config.DROPOUT_RATE, )

        # three layers linear dense layer of config.DECODER_DENSE_TANH_HID_SIZE * config.DECODER_NUM_LAYERS size
        self.tanh_layer2 = ks.layers.Dense(config.DECODER_DENSE_TANH_HID_SIZE * config.DECODER_NUM_LAYERS,
                                           activation=ks.activations.tanh)
        self.tanh_layer3 = ks.layers.Dense(config.DECODER_DENSE_TANH_HID_SIZE * config.DECODER_NUM_LAYERS,
                                           activation=ks.activations.tanh)
        # one layer linear dense layer of config.DECODER_DENSE_TANH_HID_SIZE size
        self.tanh_layer4 = ks.layers.Dense(config.DECODER_DENSE_TANH_HID_SIZE, activation=ks.activations.tanh)

        # two layers LSTM layer of config.DECODER_LSTM_SIZE size
        self.lstm_layer3 = ks.layers.LSTM(config.DECODER_LSTM_SIZE, dropout=config.DROPOUT_RATE, return_sequences=True)
        self.lstm_layer4 = ks.layers.LSTM(config.DECODER_LSTM_SIZE, dropout=config.DROPOUT_RATE, )

        # one layer linear dense layer of DECODER_DENSE_TANH_HID_SIZE * config.DECODER_NUM_LAYERS size

        # grooVAE comments on the code:
        # !!! ReLU for the oldest models, Sigmoid for the more recent ones.
        # According to roberts, this should be a Softmax activation (and not RELU as here P&L did). but why softmax?
        # I put sigmoid instead... To me it makes more sense because in the end we want 0s and 1s.
        self.relu_layer = ks.layers.Dense(config.DECODER_DENSE_TANH_HID_SIZE * config.DECODER_NUM_LAYERS,
                                          activation=ks.activations.relu)

        # Visualization
        # self.decoder_model.summary()

    def call(self, inputs, training=None, mask=None):
        # todo: implementation
        keras_in = self.latent_inputs(inputs)
        x = self.tanh_layer1(keras_in)
        output = self.lstm_layers(x)
        return output
