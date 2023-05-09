from scipy.spatial.distance import *
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import layers


def normalize_tensor01(tensor):
    # TODO: DOC
    tensor1 = layers.subtract(
        [tensor,
         K.min(tensor)]
    ),
    tensor2 = layers.subtract(
        [K.max(tensor),
         K.min(tensor)]
    )

    return layers.Lambda(lambda x: x[0] / x[1])([tensor1, tensor2])


def compute_ssm(input_tensor, metric='manhattan', axis=1, p=None):
    """
    Computes the self similarity matrix of a given input.
    Parameters:
      :param input_tensor: tensor which ssm is computed upon
      :param: (optional) float used only for minkowsy metric. See minkowsky_similarity_func(u, v, p)
      :param axis:
      :param metric: (optional) string indicating which distance metric to use for the ssm
              available: 'manhattan', 'norm1', 'minkowsky', 'lpnorm'
              non case sensitive
      :param p: (optional) integer that needs to be used for minkwsky distance

    Returns:
      :returns ssm: bidimensional matrix (M x M) with values in  [0, 1) range

    """

    def manhattan_similarity_func(input1, input2):
        """
        Computes manhattan (L1) similarity matrix between mono dimensional tensors.
        Parameters:
          :param input1: first array for similarity
          :param input2: second array for similarity
        """
        return K.sum(K.abs(input1 - input2), axis=axis, keepdims=True)

    def minkowsky_sqrt_similarity_func(input1, input2):
        """
        Computes similarity matrix between mono dimensional tensors using sqrt distance.
        Parameters:
          :param input1: first array for similarity
          :param input2: second array for similarity
        """

        return K.sum(K.sqrt(K.abs(input1 - input2)), axis=axis, keepdims=True)

    def p_norm_similarity_func(input1, input2):
        """
        Computes similarity matrix between mono dimensional tensors using minkowsy distance metric (LP norm).
        Power is determined by compute_ssm() parameters p. P MUST BE INTEGER

        Parameters:
          :param input1: first array for similarity
          :param input2: second array for similarity
        """
        assert p is not None
        return K.sum(K.pow(K.abs(input1 - input2), p), axis=axis, keepdims=True)

    input_t = K.transpose(input)

    metric = metric.lower()
    print('metric: ', metric)

    # self distance matrix
    if metric == 'manhattan' or metric == 'l1norm':
        ssm = manhattan_similarity_func(input_t, input_t)

    elif metric == 'root' or metric == 'minkowsky_sqrt' or metric == 'sqrt':
        ssm = minkowsky_sqrt_similarity_func(input_t, input_t)

    elif metric == 'intMinkowsky' or metric == 'lpIntNorm':
        ssm = p_norm_similarity_func(input_t, input_t)

    else:
        print('ERROR: metric: [', metric, '] not recongnised')
        return -1

    ssm_norm = normalize_tensor01(ssm)
    return ssm_norm


class SSMLayer(layers.Layer):

    def __init__(self, metric='manhattan', p=0.5):
        # TODO: doc
        super(SSMLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        # TODO: doc
        ssm = compute_ssm(inputs)
        return ssm


# ======================== Equivalent np array operations =========================

def compute_matrix_ssm(piano_roll_input, metric='manhattan', p=0.5):
    """
    Computes the self similarity matrix of a given input.
    Parameters:
      piano_roll_input: bidimensional matrix (M x N) with values in  [0, 1) range
      p: (optional) float used only for minkowsy metric. See minkowsky_similarity_func(u, v, p)
      metric: (optional) string indicating which distance metric to use for the ssm
              available: 'manhattan', 'norm1', 'minkowsky', 'lpnorm'
              non case sensitive
    Returns:
      ssm: bidimensional matrix (M x M) with values in  [0, 1) range

    """

    def manhattan_matrix_similarity_func(u, v):
        """
        Computes manhattan (L1) similarity matrix between input arrays.
        Parameters:
          u: first array for similarity
          v: second array for similarity
        """
        sim = 0
        for idx, a in enumerate(u):
            sim = sim + (1 - np.abs(a - v[idx]))
        return sim

    def minkowsky_matrix_similarity_func(u, v):
        """
        Computes similarity matrix between input arrays using minkowsy distance metric (LP norm).
        Power is determined by compute_ssm() parameters p
        \left(\sum _{{i=1}}^{n}|x_{i}-y_{i}|^{p}\right)^{{1/p}}
        Parameters:
          u: first array for similarity
          v: second array for similarity
        """
        sim = 0
        for idx, a in enumerate(u):
            sim = sim + (1 - np.abs(a - v[idx])) ** p
        return sim

    input_t = np.transpose(piano_roll_input)

    metric = metric.lower()
    print('metric: ', metric)
    # self distance matrix
    if metric == 'manhattan' or metric == 'norm1':
        ssm = cdist(input_t, input_t, manhattan_matrix_similarity_func)
        ssm_norm = (ssm - np.min(ssm)) / (np.max(ssm) - np.min(ssm))

    elif metric == 'minkowsky' or metric == 'lpnorm':
        ssm = cdist(input_t, input_t, minkowsky_matrix_similarity_func)
        ssm_norm = (ssm - np.min(ssm)) / (np.max(ssm) - np.min(ssm))
    else:
        print('ERROR: metric: [', metric, '] not recongnised')
        return -1

    return ssm_norm


def ssm_np_to_tf_tensor(ssm_np):
    """
    Converts ssm from np array to tf tensor.
    Parameters:
      ssm_np: ssm of np array_type
    Returns:
      ssm_tf: ssm of tf tensor type
    """
    ssm_tf = tf.convert_to_tensor(ssm_np)
    return ssm_tf