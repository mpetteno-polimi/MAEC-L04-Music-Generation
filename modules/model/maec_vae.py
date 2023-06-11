# TODO - DOC

import keras
from keras import layers
from keras import backend as K

from definitions import ConfigSections
from modules.model.kl_divergence import KLDivergenceLayer
from modules.utilities import config, math


class MaecVAE(keras.Model):

    def __init__(self, encoder, decoder, cnn, name="maec_vae", **kwargs):
        super().__init__(name=name, **kwargs)
        self._model_config = config.load_configuration_section(ConfigSections.MODEL)
        self._representation_config = config.load_configuration_section(ConfigSections.REPRESENTATION)
        self.encoder = encoder
        self.decoder = decoder
        self.cnn = cnn
        self._ssm_layer = layers.Lambda(lambda args: math.pairwise_distance(args[0], args[0], args[1]))
        self._concatenation_layer = layers.Concatenate(axis=-1, name="z_concat")
        self._kl_divergence_layer = KLDivergenceLayer()

    def call(self, inputs, training=None, mask=None):
        pianoroll = inputs
        ssm = self._ssm_layer((pianoroll, self._model_config.get("ssm_function")), training=training, mask=mask)
        z_mean, z_log_var, z = self.encoder(pianoroll, training=training, mask=mask)
        if training:
            _ = self._kl_divergence_layer((z_mean, z_log_var, self.optimizer.iterations), training=training)
        else:
            _ = self._kl_divergence_layer((z_mean, z_log_var, 0), training=training)

        ssm_embedding = self.cnn(ssm, training=training, mask=mask)
        z_ssm_embedding = self._concatenation_layer((z, ssm_embedding), training=training)
        reconstruction = self.decoder((z_ssm_embedding, pianoroll, ssm), training=training, mask=mask)
        return reconstruction

    def sample(self, inputs, z_sample=None, use_pianoroll_input=False):
        """
        Inference method
        :param inputs: ssm tensor
        :param use_pianoroll_input: if false input should be a batch of ssm tensors
                                if true it should be a batcch of pianoroll tensors
        """

        z_size = self._model_config.get("z_size")

        if use_pianoroll_input:
            pianoroll = inputs
            ssm = self._ssm_layer((pianoroll, self._model_config.get("ssm_function")), training=False, mask=None)

        else:
            pianoroll_features_num = 2 * (int(self._representation_config.get('piano_max_midi_pitch')) - int(
                self._representation_config.get('piano_min_midi_pitch')) + 1)

            ssm = inputs
            pianoroll = K.zeros(shape=(inputs.shape[0], inputs.shape[1], pianoroll_features_num))

        ssm_embedding = self.cnn(ssm, training=False)
        if z_sample is None:
            z_sample = K.random_normal(shape=(2, z_size), mean=0.0, stddev=1.0)

        z_ssm_embedding = self._concatenation_layer((z_sample, ssm_embedding), training=False)
        return self.decoder((z_ssm_embedding, pianoroll, ssm), training=False)
