# TODO - DOC

import tensorflow as tf
import keras
from keras import layers, losses, metrics
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
        # Metrics
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        pianoroll, ssm = inputs
        z_mean, z_log_var, z = self.encoder(pianoroll, training=training, mask=mask)
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

    def loss_function(self, inputs, outputs, z_mean, z_log_var):
        free_bits = self._model_config.get("free_bits")
        max_beta = self._model_config.get("max_beta")
        beta_rate = self._model_config.get("beta_rate")

        # Reconstruction loss # TODO - Reconstruction loss computation check
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(losses.binary_crossentropy(inputs, outputs), axis=(1, 2)))

        # KL divergence (explicit formula) - uses free bits as regularization parameter
        kl_div = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        free_nats = free_bits * K.log(2.0)
        kl_loss = K.maximum(kl_div - free_nats, 0)

        bits = kl_div / K.log(2.0)
        beta = (1.0 - K.pow(beta_rate, self.optimizer.iterations)) * max_beta
        return reconstruction_loss, kl_loss, bits, beta

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
        pianoroll, ssm = data

        with tf.GradientTape() as tape:
            reconstruction, z_mean, z_log_var = self(data, training=True)
            reconstruction_loss, kl_loss, kl_bits, kl_beta = self.loss_function(pianoroll, reconstruction, z_mean,
                                                                                z_log_var)
            total_loss = reconstruction_loss + kl_beta * kl_loss,

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_bits_tracker.update_state(kl_bits)
        self.kl_beta_tracker.update_state(kl_beta)

        return {
            "losses/total_loss": self.total_loss_tracker.result(),
            "losses/reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "losses/kl_loss": self.kl_loss_tracker.result(),
            'losses/kl_bits': self.kl_bits_tracker.result(),
            'losses/kl_beta': self.kl_beta_tracker().result()
        }
