
import numpy as np
from scipy.spatial import distance


def mean_points_distance(points):
    mean_distance = np.mean(distance.pdist(points, metric='euclidean', out=None))
    # TODO - remove
    min_distance = np.min(distance.pdist(points, metric='euclidean', out=None))
    sigma = mean_distance / 3
    print('mean distance:', mean_distance)
    print('min distance:', min_distance)
    print('sigma:', sigma)

    return mean_distance
