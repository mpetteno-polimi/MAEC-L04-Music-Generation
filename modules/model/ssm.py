# TODO - DOC

from keras import layers
from keras import backend as K

from modules.utilities import math


class SSMLayer(layers.Layer):

    def __init__(self, similarity_function, name="ssm", **kwargs):
        super(SSMLayer, self).__init__(name=name, **kwargs)
        # TODO - Check function for computing SSM
        self._similarity_function = similarity_function

    def call(self, inputs, training=False, *args, **kwargs):
        pianoroll = inputs
        pianoroll_t = K.permute_dimensions(pianoroll, (0, 2, 1))
        if self._similarity_function == 'cosine':
            return math.cosine_similarity(pianoroll, pianoroll_t)
        elif self._similarity_function == 'dot':
            return math.dot_similarity(pianoroll, pianoroll_t)
        elif self._similarity_function == 'manhattan':
            return math.minkowsky_distance(pianoroll, pianoroll_t, 1)
        elif self._similarity_function == 'euclidean':
            return math.minkowsky_distance(pianoroll, pianoroll_t, 2)
        else:
            raise ValueError("SSM function {} not supported".format(self._similarity_function))
