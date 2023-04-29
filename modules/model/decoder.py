# TODO - DOC

from keras import layers
from keras import backend as K

from definitions import ConfigSections
from modules import utilities


class LstmDecoderLayer(layers.Layer):

    def __init__(self, name="lstm_decoder", return_sequences=True, return_state=True, **kwargs):
        super(LstmDecoderLayer, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)
        self.decoder = utilities.model.build_stacked_rnn_layers(
            layers_sizes=self._model_config.get("dec_rnn_size"),
            type_="lstm",
            return_sequences=return_sequences,
            return_state=return_state
        )

    def call(self, inputs, training=False, *args, **kwargs):
        input_, initial_states = inputs
        output = utilities.model.call_stacked_rnn_layers(
            inputs=input_,
            rnn_layers=self.decoder,
            initial_cell_states=initial_states,
            training=training
        )

        return output


class HierarchicalDecoder(layers.Layer):

    def __init__(self, output_depth, name="hierarchical_decoder", **kwargs):
        super(HierarchicalDecoder, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)
        self.conductor = LstmDecoderLayer(name="conductor")
        self.core_decoder = LstmDecoderLayer(name="core_decoder")
        # TODO - Review output projection activation function
        self.output_projection = layers.Dense(
            units=output_depth,
            activation="softmax",
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
        batch_size = pianoroll.shape[0]
        reconstructed_pianoroll = []
        conductor_sequence_length = self._model_config.get("conductor_seq_length")
        decoder_sequence_length = self._model_config.get("decoder_seq_length")
        assert (conductor_sequence_length * decoder_sequence_length == pianoroll.shape[1])

        conductor_states = utilities.model.initial_cell_states_from_embedding(
            embedding=z_ssm_embedding,
            layers_sizes=self._model_config.get("dec_rnn_size")
        )
        conductor_input = K.zeros(shape=(batch_size, 1, 512))  # TODO - what is the second dimension?
        decoder_input = K.zeros(shape=(batch_size, 1, 176))  # TODO - what is the second dimension?
        for i in range(conductor_sequence_length):
            conductor_inputs = conductor_input, conductor_states
            cond_emb_output, conductor_initial_states = self.conductor(conductor_inputs, training)
            decoder_states = utilities.model.initial_cell_states_from_embedding(
                embedding=cond_emb_output,
                layers_sizes=self._model_config.get("dec_rnn_size")
            )
            for j in range(decoder_sequence_length):
                decoder_input = layers.Concatenate((decoder_input, cond_emb_output), axis=-1)
                decoder_inputs = decoder_input, decoder_states
                dec_emb_output, decoder_states = self.core_decoder(decoder_inputs, training)
                if _is_teacher_forcing_enabled():
                    note_emb_out = pianoroll[:, i * decoder_sequence_length + j, :]
                else:
                    note_emb_out = self.output_projection(dec_emb_output, training)
                reconstructed_pianoroll.append(note_emb_out)

        return reconstructed_pianoroll
