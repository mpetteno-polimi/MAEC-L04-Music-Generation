
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


def pairwise_cosine_sim(tensors):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    tensor_a, tensor_b = tensors
    a_l2_norm = l2_norm(tensor_a, axis=2)
    b_l2_norm = l2_norm(tensor_b, axis=2)
    num = K.batch_dot(tensor_a, K.permute_dimensions(tensor_b, (0, 2, 1)))
    den = (a_l2_norm * K.permute_dimensions(b_l2_norm, (0, 2, 1)))
    dist_mat = num / den

    return dist_mat
