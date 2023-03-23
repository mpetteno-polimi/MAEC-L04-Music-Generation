import logging as log
import tensorflow as tf
from tensorflow import keras as ks


# todo: tune hparams
class LstmEncoder(ks.Model):
    def __init__(self, config):  # todo: default config object
        super(LstmEncoder, self).__init__()
        # prepare keras-compatible tensor-like input data
        input_shape = (config.IN_SEQ_LEN, config.AVAILABLE_NOTES)
        self.inputsKs = tf.keras.Input(input_shape)

        # one lstm layer of size config.HIDDEN_SIZE
        self.LSTM = ks.layers.LSTM(config.HIDDEN_SIZE)

        self.dense_mean = ks.layers.Dense(config.LATENT_DIMENSION, name="z_mean")
        self.dense_log_var = ks.layers.Dense(config.LATENT_DIMENSION, name="z_log_var")

    def call(self, inputs, training=None, mask=None):
        keras_in = self.inputsKs(inputs)
        x = self.LSTM(keras_in)
        z_mean = self.denseMean(x)
        z_log_var = self.denseLogVar(x)
        return [z_mean, z_log_var]

        # VISUALIZATION FZ
        # keras.utils.plot_model(self.model, "encoder_model_info.png", show_shapes=True)
        # self.model.summary()

        # Clear (MNIST) vae that offers a slightly more complex encoder: https://keras.io/examples/generative/vae/

    # ================================= PRIVATE METHODS =========================================
