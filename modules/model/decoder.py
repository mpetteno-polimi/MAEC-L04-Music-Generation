# TODO - DOC
# todo: passing hidden states?
#  https://stackoverflow.com/questions/42415909/initializing-lstm-hidden-state-tensorflow-keras
import keras
from keras import layers
from keras import backend as K
from tensorflow import split, concat, shape, unstack
from definitions import ConfigSections
from modules import utilities
from keras.initializers.initializers_v2 import RandomNormal


class ConductorLayer(layers.Layer):

    def __init__(self, name="conductor", **kwargs):
        super(ConductorLayer, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)

        self.state_initializer = layers.Dense(
            units=4*self._model_config.get("dec_rnn_size"),
            activation='tanh',
            use_bias=True,
            kernel_initializer=RandomNormal(stddev=0.001),
            name="z_to_initial_state"
        )

        self.conductor = utilities.model.build_stacked_rnn_layers(
            layers_sizes=self._model_config.get("dec_rnn_size"),
            type="lstm"
        )

    def call(self, inputs, training=False, *args, **kwargs):
        z_ssm_embedding, pianoroll, ssm = inputs
        z_size = self._model_config.get("z_size")
        batch_size = self._model_config.get("batch_size")

        # Conductor forward pass
        conductor_sequence_length = self._model_config.get("conductor_length")

        # conductor_initial_state = utilities.model.initial_cell_state_from_embedding(
        #     embedding=z_ssm_embedding,
        #     layer_size=self._model_config.get("dec_rnn_size")[0]
        # )
        # TODO - Create tensor for conductor input based on pianoroll tensor y dimension it should be
        #  [batch_size x conductor_seq_len x pianoroll_y (+ ssm)]
        conductor_input = K.zeros(
            shape=[batch_size, conductor_sequence_length, shape(z_ssm_embedding)[1]])  # todo: check 3rd dimension size
        fc_init = self.state_initializer(conductor_input)
        h1, c1, h2, c2 = split(fc_init, 4, axis=1)

        conductor_output = utilities.model.call_stacked_rnn_layers(
            inputs=conductor_input,
            rnn_layers=self.conductor,
            initial_cell_state=[[h1, c1], [h2, c2]]
        )
        return conductor_output


class CoreDecoderLayer(layers.Layer):

    def __init__(self, name="hierarchical_decoder", training=False, **kwargs):
        super(CoreDecoderLayer, self).__init__(name=name, **kwargs)
        self.training = training
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)
        output_depth = self._model_config.get('piano_min_midi_pitch') - self._model_config.get(
            'piano_min_midi_pitch') + 1

        self.state_initializer = layers.Dense(
            units=4 * self._model_config.get("dec_rnn_size"),
            activation='tanh',
            use_bias=True,
            kernel_initializer=RandomNormal(stddev=0.001),
            name="z_to_initial_state"
        )
        self.decoder = utilities.model.build_stacked_rnn_layers(
            layers_sizes=self._model_config.get("dec_rnn_size"),
            type="lstm"
        )
        self.output_layer = layers.Dense(
            units=output_depth,
            activation='tanh',
            use_bias=True,
            kernel_initializer=RandomNormal(stddev=0.001),
            name="z_to_initial_state"
        )

    def call(self, inputs, training=False, *args, **kwargs):
        """
        The decoder
        LSTM uses the embedding for the current bar concatenated with the output of the
        following fully-connected layer as its input for the next time step
        :param inputs:
        :param training:
        :param args:
        :param kwargs:
        :return:
        """

        batch_size = self._model_config.get("batch_size")
        dec_rnn_size = self._model_config.get("dec_rnn_size")
        output_depth = self._model_config.get('piano_min_midi_pitch') - self._model_config.get(
            'piano_min_midi_pitch') + 1

        # get output from previous bar (fir the very first iteration input is zero) todo: check dimension
        previous = kwargs.pop('previous', keras.Input(shape=(batch_size, dec_rnn_size + output_depth)))

        def _is_teacher_force_enabled():
            if self._model_config.get("teacher_forcing") and training:
                return K.random.random() < self._model_config.get("teacher_forcing_prob")
            else:
                return False

        conductor_embedding, pianoroll = inputs
        decoder_sequence_length = self._model_config.get("decoder_length")

        assert (conductor_embedding.shape[1] * decoder_sequence_length == pianoroll.shape[1])
        # TODO - Core decoder forward pass (See function 'forward_tick_rnn' in master thesis)

        # takes conductor output i through fc layer and passes
        states = self.state_initializer(conductor_embedding)
        # splits it in four to initialize states
        h1, c1, h2, c2 = split(states, 4, axis=1)

        outputs = []
        for i in range(self._model_config.get("slice_bars")):
            if _is_teacher_force_enabled:
                # todo: set teacher forcing previous = ground_truth[i-1]
                continue

            core_dec_in = concat(conductor_embedding, previous)
            utilities.model.call_stacked_rnn_layers(core_dec_in, self.decoder, [c1, c2])
            previous = utilities.model.call_stacked_rnn_layers(
                inputs=core_dec_in,
                rnn_layers=self.conductor,
                initial_cell_state=[[h1, c1], [h2, c2]]
            )

            outputs.append(previous)

        return outputs, previous


class HierarchicalDecoder(layers.Layer):

    def __init__(self, name="hierarchical_decoder", **kwargs):
        super(HierarchicalDecoder, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)
        self.conductor = ConductorLayer()
        self.core_decoder = CoreDecoderLayer()
        # TODO - HierarchicalDecoder output layer init

    def call(self, inputs, training=False, *args, **kwargs):
        z_ssm_embedding, pianoroll, ssm = inputs
        output_depth = self._model_config.get('piano_min_midi_pitch') - self._model_config.get(
            'piano_min_midi_pitch') + 1
        batch_size = self._model_config.get("batch_size")

        # conductor - out [batch_size, conductor_length, dec_rnn_size]
        conductor_output = self.conductor(inputs, training=False)
        conductor_output = unstack(conductor_output, axis=1)

        # decode conductor embeddings
        outputs = []
        previous = K.zeros(shape=(batch_size, output_depth))

        for embedding in conductor_output:
            decoder_out, last_out = self.core_decoder(embedding, previous=previous, training=False) # todo: add training param
            outputs.extend(decoder_out)
            previous = last_out


        decoder_output = self.core_decoder((conductor_output, pianoroll))
        # TODO - HierarchicalDecoder output layer call
        return
