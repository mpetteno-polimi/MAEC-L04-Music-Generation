# TODO - DOC

import tensorflow as tf
from keras import layers
from keras import backend as K
from keras.initializers import initializers

from definitions import ConfigSections
from modules.utilities import config, rnn


class InitialCellStateFromEmbeddingLayer(layers.Layer):

    def __init__(self, layers_sizes, name="initial_cell_state", **kwargs):

        def split(cell_states):
            # TODO - Generic Keras backend implementation
            return tf.split(cell_states, num_or_size_splits=flatten_state_sizes, axis=1)

        def pack(cell_states):
            return list(zip(cell_states[::2], cell_states[1::2]))

        super(InitialCellStateFromEmbeddingLayer, self).__init__(name=name, **kwargs)
        cell_state_sizes = [(int(layer_size), int(layer_size)) for layer_size in layers_sizes]
        flatten_state_sizes = [x for cell_state_size in cell_state_sizes for x in cell_state_size]
        self._initial_cell_states = layers.Dense(
            units=sum(flatten_state_sizes),
            activation='tanh',
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            name="z_to_initial_state"
        )
        self._split_initial_cell_states = layers.Lambda(split, name='initial_state_split')
        self._pack_initial_cell_states = layers.Lambda(pack, name='initial_state_pack')

    def call(self, inputs, training=False, *args, **kwargs):
        initial_cell_states = self._initial_cell_states(inputs, training=training)
        split_initial_cell_states = self._split_initial_cell_states(initial_cell_states, training=training)
        packed_initial_cell_states = self._pack_initial_cell_states(split_initial_cell_states, training=training)
        return packed_initial_cell_states


class HierarchicalDecoder(layers.Layer):

    def __init__(self, output_depth, name="hierarchical_decoder", **kwargs):
        super(HierarchicalDecoder, self).__init__(name=name, **kwargs)
        self._model_config = config.load_configuration_section(ConfigSections.MODEL)
        self._training_config = config.load_configuration_section(ConfigSections.TRAINING)

        self._batch_size = self._training_config.get("batch_size")
        self._layers_sizes = self._model_config.get("dec_rnn_size")

        # Init conductor layer
        self.conductor_initial_cell_state = InitialCellStateFromEmbeddingLayer(layers_sizes=self._layers_sizes,
                                                                               name="conductor_initial_cell_state")
        self.conductor = rnn.build_lstm_layers(
            layers_sizes=self._layers_sizes,
            bidirectional=False,
            return_sequences=False,
            return_state=True,
            name="conductor"
        )

        # Init core-decoder layer
        self.decoder_initial_cell_state = InitialCellStateFromEmbeddingLayer(layers_sizes=self._layers_sizes,
                                                                             name="core_decoder_initial_cell_state")
        self.core_decoder = rnn.build_lstm_layers(
            layers_sizes=self._layers_sizes,
            bidirectional=False,
            return_sequences=False,
            return_state=True,
            name="core_decoder"
        )

        self.output_projection = layers.Dense(
            units=output_depth,
            activation="sigmoid",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="output_projection"
        )

    def call(self, inputs, training=False, *args, **kwargs):

        def _is_teacher_forcing_enabled():
            if self._model_config.get("teacher_forcing") and training:
                return K.random.random() < self._model_config.get("teacher_forcing_prob")
            else:
                return False

        z_ssm_embedding, pianoroll, ssm = inputs
        reconstructed_pianoroll = []
        conductor_sequence_length = self._model_config.get("conductor_seq_length")
        decoder_sequence_length = self._model_config.get("decoder_seq_length")
        assert (conductor_sequence_length * decoder_sequence_length == pianoroll.shape[1])

        conductor_initial_input = K.zeros(shape=(self._batch_size, 1, self._layers_sizes[0]))
        conductor_states = self.conductor_initial_cell_state(z_ssm_embedding, training=training)
        decoder_input = K.zeros(shape=(self._batch_size, 1, pianoroll.shape[2]))
        for i in range(conductor_sequence_length):
            cond_emb_output, *conductor_states = self.conductor(conductor_initial_input,
                                                                initial_state=conductor_states,
                                                                training=training)
            decoder_states = self.decoder_initial_cell_state(cond_emb_output, training=training)
            # Expand conductor embedding dims to allow for concatenation with the decoder input
            cond_emb_output = K.expand_dims(cond_emb_output, axis=1)
            for j in range(decoder_sequence_length):
                decoder_input = K.concatenate(tensors=[decoder_input, cond_emb_output], axis=-1)
                dec_emb_output, *decoder_states = self.core_decoder(decoder_input,
                                                                    initial_state=decoder_states,
                                                                    training=training)
                if _is_teacher_forcing_enabled():
                    note_emb_out = pianoroll[:, i * decoder_sequence_length + j, :]
                else:
                    note_emb_out = self.output_projection(dec_emb_output, training=training)
                decoder_input = K.expand_dims(note_emb_out, axis=1)
                reconstructed_pianoroll.append(note_emb_out[:, None, :])

        return K.concatenate(reconstructed_pianoroll, axis=1)
