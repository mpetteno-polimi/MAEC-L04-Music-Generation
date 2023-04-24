
import tensorflow as tf
import tf_slim as slim
import tf_slim.nets as slim_nets

from magenta.models.music_vae import CategoricalLstmDecoder


class MaecDecoder(CategoricalLstmDecoder):

    def __init__(self, cnn: str):
        super().__init__()
        self._batch_size = None
        self._max_seq_length = None
        self._cnn_id = cnn

    def build(self, hparams, output_depth, is_training=True):
        self._max_seq_length = hparams.max_seq_len
        self._batch_size = hparams.batch_size
        super().build(hparams, output_depth, is_training)

    def build_cnn(self):
        cnn_inputs = self.get_pianoroll_smm_tensors()
        if self._cnn_id == 'inceptionv3':
            return self.build_inception_cnn(cnn_inputs)

    def build_inception_cnn(self, inputs):
        with slim.arg_scope(slim_nets.inception.inception_v3_arg_scope()):
            with tf.compat.v1.variable_scope('InceptionV3', [inputs], reuse=None) as scope:
                net, end_points = slim_nets.inception.inception_v3_base(
                    inputs=inputs,
                    scope=scope,
                    min_depth=16,
                    depth_multiplier=1.0)
                with tf.compat.v1.variable_scope('Logits'):
                    # Global average pooling.
                    net = tf.reduce_mean(input_tensor=net, axis=[1, 2], keepdims=False, name='GlobalPool')
                    end_points['global_pool'] = net
                    return net, end_points

    def reconstruction_loss(self, x_input, x_target, x_length, z=None, c_input=None):
        cnn_output, _ = self.build_cnn()
        z_concat = tf.concat([z, cnn_output], axis=1)
        return super().reconstruction_loss(x_input, x_target, x_length, z_concat, c_input)

    # TODO - Move to data converter class
    def get_pianoroll_smm_tensors(self):
        batch_size, height, width, channels = self._batch_size, self._max_seq_length, self._max_seq_length, 1
        return tf.compat.v1.random_uniform([batch_size, height, width, channels], maxval=1)
