# TODO - DOC

import keras
from keras import layers, losses
from keras import backend as K

from definitions import ConfigSections
from modules.model.kl_divergence import KLDivergenceLayer
from modules.utilities import config, math


class MaecVAE(keras.Model):

    def __init__(self, encoder, decoder, cnn, name="maec_vae", **kwargs):
        super().__init__(name=name, **kwargs)
        self._model_config = config.load_configuration_section(ConfigSections.MODEL)
        self.encoder = encoder
        self.decoder = decoder
        self.cnn = cnn
        # TODO - Check function for computing SSM
        self._compute_ssm_layer = layers.Lambda(math.pairwise_cosine_sim)
        self._concatenation_layer = layers.Concatenate(axis=-1, name="z_concat")
        self._kl_divergence_layer = KLDivergenceLayer()

    def call(self, inputs, training=None, mask=None):
        pianoroll = inputs
        ssm = self._compute_ssm_layer((pianoroll, pianoroll))
        z_mean, z_log_var, z = self.encoder(pianoroll, training=training, mask=mask)
        _ = self._kl_divergence_layer((z_mean, z_log_var, self.optimizer.iterations))
        ssm_embedding = self.cnn(ssm, training=training, mask=mask)
        z_ssm_embedding = self._concatenation_layer((z, ssm_embedding))
        reconstruction = self.decoder((z_ssm_embedding, pianoroll, ssm), training=training, mask=mask)
        return reconstruction

    def sample(self, inputs):
        ssm = inputs
        z_size = self._model_config.get("z_size")
        ssm_embedding = self.cnn(ssm, training=False)
        z_sample = K.random_normal(shape=z_size, mean=0.0, stddev=1.0)
        decoder_input = self.concatenation_layer((z_sample, ssm_embedding))
        return self._decoder.decode(decoder_input, training=False)

    def loss_fn(self):
        return losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
