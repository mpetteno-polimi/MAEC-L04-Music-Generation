# TODO - DOC

from keras import layers
from keras import backend as K

from definitions import ConfigSections
from modules import utilities


class ConductorLayer(layers.Layer):

    def __init__(self, name="conductor", **kwargs):
        super(ConductorLayer, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)
        self.conductor = utilities.model.build_stacked_rnn_layers(
            layers_sizes=self._model_config.get("dec_rnn_size"),
            type="lstm"
        )

    def call(self, inputs, training=False, *args, **kwargs):
        z_ssm_embedding, pianoroll, ssm = inputs

        # Conductor forward pass
        conductor_sequence_length = self._model_config.get("conductor_length")
        conductor_initial_state = utilities.model.initial_cell_state_from_embedding(
            embedding=z_ssm_embedding,
            layer_size=self._model_config.get("dec_rnn_size")[0]
        )
        # TODO - Create tensor for conductor input based on pianoroll tensor y dimension it should be
        #  [batch_size x conductor_seq_len x pianoroll_y]
        conductor_input = None
        conductor_output = utilities.model.call_stacked_rnn_layers(
            inputs=conductor_input,
            rnn_layers=self.conductor,
            initial_cell_state=conductor_initial_state
        )
        return conductor_output


class CoreDecoderLayer(layers.Layer):

    def __init__(self, name="hierarchical_decoder", **kwargs):
        super(CoreDecoderLayer, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)
        self.decoder = utilities.model.build_stacked_rnn_layers(
            layers_sizes=self._model_config.get("dec_rnn_size"),
            type="lstm"
        )

    def call(self, inputs, training=False, *args, **kwargs):

        def _is_teacher_force_enabled():
            if self._model_config.get("teacher_forcing") and training:
                return K.random.random() < self._model_config.get("teacher_forcing_prob")
            else:
                return False

        conductor_output, pianoroll = inputs

        decoder_sequence_length = self._model_config.get("decoder_length")
        assert (conductor_output.shape[1] * decoder_sequence_length == pianoroll.shape[1])
        # TODO - Core decoder forward pass (See function 'forward_tick_rnn' in master thesis)

        return


class HierarchicalDecoder(layers.Layer):

    def __init__(self, name="hierarchical_decoder", **kwargs):
        super(HierarchicalDecoder, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)
        self.conductor = ConductorLayer()
        self.core_decoder = CoreDecoderLayer()
        # TODO - HierarchicalDecoder output layer init

    def call(self, inputs, training=False, *args, **kwargs):
        z_ssm_embedding, pianoroll, ssm = inputs
        conductor_output = self.conductor(inputs, training=False)
        decoder_output = self.core_decoder((conductor_output, pianoroll))
        # TODO - HierarchicalDecoder output layer call
        return
