
from keras import backend as K


def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + K.exp(log_variance / 2) * epsilon
    return random_sample


def l2_norm(x, axis=None):
    """
    Takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm


def cosine_similarity(tensor_a, tensor_b):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x d x m] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    a_l2_norm = l2_norm(tensor_a, axis=2)
    b_l2_norm = l2_norm(tensor_b, axis=1)
    cosine_sim = K.batch_dot(tensor_a, tensor_b) / (a_l2_norm * b_l2_norm)

    return cosine_sim


def dot_similarity(tensor_a, tensor_b):
    """
    Computes dot similarity between two input matrices

    Parameters:
      :param tensor_a: first array for similarity
      :param tensor_b: second array for similarity
    """

    return K.batch_dot(tensor_a, K.permute_dimensions(tensor_b, (0, 2, 1)))


def minkowsky_distance(tensor_a, tensor_b, p):
    """
    Computes similarity matrix between bidimensional tensors using sqrt distance.

    Parameters:
      :param tensor_a: first array for similarity
      :param tensor_b: second array for similarity
    """

    return K.pow(K.sum(K.pow(K.abs(tensor_a-tensor_b), p), axis=-1, keepdims=True), 1/p)
