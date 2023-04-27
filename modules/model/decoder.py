# TODO - DOC

from keras import layers

from definitions import ConfigSections
from modules import utilities


class HierarchicalDecoder(layers.Layer):

    def __init__(self, name="decoder", **kwargs):
        super(HierarchicalDecoder, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)

        lstm_dec_size = self._model_config.get("dec_rnn_size")

        # Conductor layer - Stacked LSTM
        conductor = utilities.model.build_stacked_rnn_layers(
            layers_sizes=self._model_config.get("dec_rnn_size"),
            type="lstm"
        )

        # TODO - Decoder middle layers

        # Core decoder
        decoder = utilities.model.build_stacked_rnn_layers(
            layers_sizes=self._model_config.get("dec_rnn_size"),
            type="lstm"
        )

    def call(self, inputs, *args, **kwargs):
        # TODO - Decoder call function
        pass
