
from keras import backend as K


def reparametrization_trick(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + K.exp(log_variance / 2) * epsilon
    return random_sample


def minkowski_distance(x1, x2, p):
    """
    Computes the Minkowski distance between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)
    p -- Parameter for the Minkowski distance.

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """

    # Expand dimensions to enable broadcasting
    x1_expand = K.expand_dims(x1, axis=2)
    x2_expand = K.expand_dims(x2, axis=1)

    diff = K.abs(x1_expand - x2_expand)
    distances = K.pow(K.sum(K.pow(diff, p), axis=-1), 1.0 / p)
    return distances


def dot_product_distance(x1, x2):
    """
    Computes the dot product distance between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """

    x2_transposed = K.permute_dimensions(x2, (0, 2, 1))
    dot_product = K.batch_dot(x1, x2_transposed)
    distances = 1.0 - dot_product
    return distances


def cosine_distance(x1, x2):
    """
    Computes the cosine distance between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """

    x1_norm = K.l2_normalize(x1, axis=-1)
    x2_norm = K.l2_normalize(x2, axis=-1)
    distances = dot_product_distance(x1_norm, x2_norm)
    return distances


def sqrt_euclidean_distance(x1, x2):
    """
    Computes the square root Euclidean distance between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """

    # Expand dimensions to enable broadcasting
    x1_expand = K.expand_dims(x1, axis=2)
    x2_expand = K.expand_dims(x2, axis=1)

    squared_diff = K.square(x1_expand - x2_expand)
    sum_squared_diff = K.sum(squared_diff, axis=-1)
    distances = K.sqrt(sum_squared_diff)
    return distances


def pairwise_distance(x1, x2, metric='euclidean'):
    """
    Computes pairwise distances between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)
    metric -- Distance metric to use (default: 'euclidean')

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """

    if metric == 'euclidean':
        distances = minkowski_distance(x1, x2, p=2)
    elif metric == 'manhattan':
        distances = minkowski_distance(x1, x2, p=1)
    elif metric == 'cosine':
        distances = cosine_distance(x1, x2)
    elif metric == 'sqrt_euclidean':
        distances = sqrt_euclidean_distance(x1, x2)
    elif metric == 'dot_product':
        distances = dot_product_distance(x1, x2)
    else:
        raise ValueError('Metric {} not supported. Available options: euclidean, cosine, manhattan, '
                         'sqrt_euclidean, dot_product'.format(metric))

    return distances
