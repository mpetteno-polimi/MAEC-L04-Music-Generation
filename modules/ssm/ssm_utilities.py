from scipy.spatial.distance import *
import tensorflow as tf
from keras import backend as K
from keras import layers
import numpy as np


def normalize_tensor01(tensor):
    # TODO: DOC
    minimum = K.min(tensor)
    maximum = K.max(tensor)
    tensor1 = tensor - minimum
    tensor2 = maximum - minimum

    return layers.Lambda(lambda x: x[0] / x[1])([tensor1, tensor2])


class SSMLayer(layers.Layer):

    def compute_ssm(self, input_tensor, axis=1, p=None):
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
            Computes manhattan (L1) similarity matrix between bidimensional tensors.

            Parameters:
              :param input1: first array for similarity
              :param input2: second array for similarity
            """
            distance = np.zeros((input1.shape[0], input2.shape[1]), dtype=np.float32)

            for i in range(input1.shape[0]):
                # each row is compared with each column
                for j in range(input2.shape[1]):
                    diff = input1[i, :] - input2[:, j]
                    distance[i][j] = K.sum(K.abs(diff), keepdims=False)

            distance = K.constant(distance,
                                  dtype="float32",
                                  shape=(input1.shape[0], input2.shape[1]))
            return distance

        def minkowsky_sqrt_similarity_func(input1, input2):
            """
            Computes similarity matrix between bidimensional tensors using sqrt distance.

            Parameters:
              :param input1: first array for similarity
              :param input2: second array for similarity
            """
            distance = np.zeros((input1.shape[0], input2.shape[1]), dtype=np.float32)

            for i in range(input1.shape[0]):
                # each row is compared with each column
                for j in range(input2.shape[1]):
                    diff = input1[i, :] - input2[:, j]
                    distance[i][j] = K.sum(K.sqrt(K.abs(diff)), keepdims=False)

            distance = K.constant(distance,
                                  dtype="float32",
                                  shape=(input1.shape[0], input2.shape[1]))
            return distance

        def lp_norm_similarity_func(input1, input2):
            """
            Computes similarity matrix between bidimensional tensors using minkowsy distance metric (LP norm).
            Power is determined by compute_ssm() parameters p.
            Parameters:
              :param input1: first array for similarity
              :param input2: second array for similarity
            """
            # assert p is int
            distance = np.zeros((input1.shape[0], input2.shape[1]), dtype=np.float32)
            for i in range(input1.shape[0]):
                # each row is compared with each column
                for j in range(input2.shape[1]):
                    diff = input1[i, :] - input2[:, j]
                    absolute = K.abs(diff)
                    power = K.pow(absolute, self.p)
                    distance[i][j] = K.sum(power, keepdims=False)

            distance = K.constant(distance,
                                  dtype="float32",
                                  shape=(input1.shape[0], input2.shape[1]))
            return distance

        def prod_similarity_func(input1, input2):
            """
            Computes dot similarity between two input matricees

            Parameters:
              :param input1: first array for similarity
              :param input2: second array for similarity
            """
            return K.dot(input1, input2)

        input_tensor_t = K.transpose(input_tensor)
        print('metric: ', self.metric)

        # self distance matrix
        if self.metric == 'manhattan' or self.metric == 'l1norm':
            dist_matrix = manhattan_similarity_func(input_tensor_t, input_tensor)

        elif self.metric == 'root' or self.metric == 'minkowsky_sqrt' or self.metric == 'sqrt':
            dist_matrix = minkowsky_sqrt_similarity_func(input_tensor_t, input_tensor)

        elif self.metric == 'int_minkowsky' or self.metric == 'lp_int_norm':
            dist_matrix = lp_norm_similarity_func(input_tensor_t, input_tensor)

        elif self.metric == 'dot' or self.metric == 'multiplication' or self.metric == 'dot_norm':
            dist_matrix = prod_similarity_func(input_tensor_t, input_tensor)

        else:
            print('ERROR: metric: [', self.metric, '] not recongnised')
            return -1

        dist_matrix_norm = normalize_tensor01(dist_matrix)
        ssm = 1 - dist_matrix_norm
        return ssm

    def __init__(self, metric, p=2):
        # TODO: doc
        super(SSMLayer, self).__init__()
        self.metric = metric.lower()
        self.p = p

    def call(self, inputs, *args, **kwargs):
        # TODO: doc
        ssm = self.compute_ssm(inputs)
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