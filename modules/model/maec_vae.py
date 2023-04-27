# TODO - DOC

import keras
from keras import layers, losses
from keras import backend as K

from definitions import ConfigSections
from modules import utilities
from modules.model.kl_divergence import KLDivergenceLayer


class MaecVAE(keras.Model):

    def __init__(self, encoder, decoder, cnn, name="maec_vae", **kwargs):
        super().__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)
        self.encoder = encoder
        self.decoder = decoder
        self.cnn = cnn
        self.concatenation_layer = layers.Concatenate(axis=-1, name="z_concat")
        self.kl_divergence_layer = KLDivergenceLayer()

    def call(self, inputs, training=None, mask=None):
        pianoroll, ssm = inputs
        z_mean, z_log_var, z = self.encoder(pianoroll, training=training, mask=mask)
        z_mean, z_log_var = self.kl_divergence_layer((z_mean, z_log_var))
        ssm_embedding = self.cnn(ssm, training=training, mask=mask)
        decoder_input = self.concatenation_layer([z, ssm_embedding])
        reconstruction = self.decoder([decoder_input, pianoroll, ssm], training=training, mask=mask)
        return reconstruction, z_mean, z_log_var

    def sample(self, inputs):
        ssm = inputs
        z_size = self._model_config.get("z_size")
        ssm_embedding = self.cnn(ssm, training=False)
        z_sample = K.random_normal(shape=z_size, mean=0.0, stddev=1.0)
        decoder_input = self.concatenation_layer([z_sample, ssm_embedding])
        return self._decoder.decode(decoder_input, training=False)

    def loss_fn(self):
        # TODO - Reconstruction loss computation check
        return losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
