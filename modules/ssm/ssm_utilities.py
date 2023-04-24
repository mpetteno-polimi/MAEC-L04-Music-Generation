from scipy.spatial.distance import *
import numpy as np
import tensorflow as tf


def compute_ssm(piano_roll_input, metric='manhattan', p=0.5):
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

    def manhattan_similarity_func(u, v):
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

    def minkowsky_similarity_func(u, v):
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
        ssm = cdist(input_t, input_t, manhattan_similarity_func)
        ssm_norm = (ssm - np.min(ssm)) / (np.max(ssm) - np.min(ssm))

    elif metric == 'minkowsky' or metric == 'lpnorm':
        ssm = cdist(input_t, input_t, minkowsky_similarity_func)
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