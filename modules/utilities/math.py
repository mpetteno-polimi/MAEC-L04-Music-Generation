import numpy as np
from itertools import combinations, chain
from scipy.special import comb
from scipy.spatial import distance


def mean_points_distance(points):
    mean_distance = np.mean(distance.pdist(points, metric='euclidean', out=None))
    return mean_distance


def min_points_distance(points):
    min_distance = np.min(distance.pdist(points, metric='euclidean', out=None))
    return min_distance


def find_couples_subset(input_set):
    n = np.shape(input_set)[0]
    count = comb(n, 2, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), 2)),
                        int, count=count*2)
    index_list = index.reshape(-1, 2)
    pairs = [[input_set[indices[0]], input_set[indices[1]]] for indices in index_list]
    return np.asarray(pairs)

'''
todo: generic subset implementation
def find_subset(input_set, subset_len):
    n = np.shape(input_set)[0]
    count = comb(n, subset_len, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), subset_len)),
                        int, count=count*subset_len)
    index_list = index.reshape(-1, subset_len)
    # return [[input_set[indices[i]] for i in indices] for indices in index_list]
    return [[input_set[indices[0]], input_set[indices[1]]] for indices in index_list]

'''