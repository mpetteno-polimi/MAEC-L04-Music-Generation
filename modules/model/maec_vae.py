# TODO - DOC

import keras
from keras import layers
from keras import backend as K

from definitions import ConfigSections
from modules import utilities


class MaecVAE(keras.Model):

    def __init__(self, encoder, decoder, cnn, name="maec_vae", **kwargs):
        super().__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)
        self.encoder = encoder
        self.decoder = decoder
        self.cnn = cnn
        self.concatenation_layer = layers.Concatenate(axis=-1, name="z_concat")

    def call(self, inputs, training=None, mask=None):
        pianoroll = inputs[0]
        ssm = inputs[1]
        z_mean, z_log_var, z = self.encoder(pianoroll, training=training, mask=mask)
        ssm_embedding = self.cnn(ssm, training=training, mask=mask)
        decoder_input = self.concatenation_layer([z, ssm_embedding])
        reconstructed = self.decoder(decoder_input, training=training, mask=mask)
        return reconstructed

    def sample(self, inputs):
        ssm = inputs
        z_size = self._model_config.get("z_size")
        ssm_embedding = self.cnn(ssm, training=False)
        z_sample = K.random_normal(shape=z_size, mean=0.0, stddev=1.0)
        decoder_input = self.concatenation_layer([z_sample, ssm_embedding])
        return self._decoder.decode(decoder_input, training=False)

    def loss_function(self, inputs, outputs):
        free_bits = self._model_config.get("free_bits")
        max_beta = self._model_config.get("max_beta")
        beta_rate = self._model_config.get("beta_rate")

        # Reconstruction loss (depends on the used decoder)
        reconstruction_loss = self._decoder.reconstruction_loss(inputs, outputs)

        # KL divergence (explicit formula) - uses free bits as regularization parameter
        kl_div = -0.5 * K.mean(1 + self._z_log_var - K.square(self._z_mean) - K.exp(self._z_log_var), axis=-1)
        free_nats = free_bits * K.log(2.0)
        kl_loss = K.maximum(kl_div - free_nats, 0)

        beta = (1.0 - K.pow(beta_rate, self._model.optimizers.iterations)) * max_beta
        return reconstruction_loss + beta * kl_loss
