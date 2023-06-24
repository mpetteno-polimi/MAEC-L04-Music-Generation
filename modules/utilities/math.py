
import numpy as np
from scipy.spatial import distance


def mean_points_distance(points):
    mean_distance = np.mean(distance.pdist(points, metric='euclidean', out=None))
    return mean_distance


def min_points_distance(points):
    min_distance = np.min(distance.pdist(points, metric='euclidean', out=None))
    return min_distance
