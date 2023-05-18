# TODO - DOC

from keras import layers

from modules.utilities import math


class SSMLayer(layers.Layer):

    def __init__(self, similarity_function, name="ssm", **kwargs):
        super(SSMLayer, self).__init__(name=name, **kwargs)
        self._similarity_function = similarity_function

    def call(self, inputs, training=False, *args, **kwargs):
        # TODO - Is it better to use similarity or dissimilarity elements in the SSM?
        # TODO - Which function is better for computing SSM?
        # TODO - Do we need normalization?
        pianoroll = inputs
        return math.pairwise_distance(pianoroll, pianoroll, self._similarity_function)
