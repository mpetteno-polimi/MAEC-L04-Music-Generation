
from keras import layers
from keras import backend as K


class KLDivergenceLayer(layers.Layer):
    """ Identity transform layer that adds KL divergence to the final model loss. """

    def call(self, inputs, training=None, mask=None):

        z_mean, z_log_var = inputs

        # Get configuration parameters
        free_bits = self._model_config.get("free_bits")
        max_beta = self._model_config.get("max_beta")
        beta_rate = self._model_config.get("beta_rate")

        # Compute and add KL-Divergence loss
        kl_div_batch = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        free_nats = free_bits * K.log(2.0)
        kl_cost_batch = K.maximum(kl_div_batch - free_nats, 0)
        beta = (1.0 - K.pow(beta_rate, self.optimizer.iterations)) * max_beta
        kl_loss_batch = beta * K.mean(kl_cost_batch)
        self.add_loss(kl_loss_batch)

        # Add metrics
        self.add_metric(kl_loss_batch, name="losses/kl_loss")
        self.add_metric(K.mean(kl_div_batch) / K.log(2.0), name="losses/kl_bits")
        self.add_metric(beta, name="losses/kl_beta")

        return inputs
