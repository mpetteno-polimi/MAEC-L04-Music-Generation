
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc


def latin_hypercube_sampling(d, grid_width, n_grid_points, rand_seed):
    grid_min = int(-grid_width / 2)
    grid_min = np.full(shape=d, fill_value=grid_min)
    grid_max = int(grid_width / 2)
    grid_max = np.full(shape=d, fill_value=grid_max)

    latin_hypercube_sampler = qmc.LatinHypercube(d, centered=False, seed=rand_seed)
    grid_points = latin_hypercube_sampler.random(n_grid_points)
    grid_points = qmc.scale(grid_points, grid_min, grid_max, reverse=False)

    return grid_points


def batch_gaussian_sampling(d, grid_points, samples_per_point, sigma, rand_seed):
    np.random.seed(rand_seed)
    batched_samples = np.zeros(shape=(samples_per_point, grid_points.shape[0], d))
    for idx, mean in enumerate(grid_points):
        samples = [sigma * np.random.randn(d) + mean for _ in range(samples_per_point)]
        batched_samples[:, idx, :] = samples

    return np.asarray(batched_samples)


def get_sigma_from_grid_points(grid_points, k_sigma):
    dm = cdist(grid_points, grid_points, metric='euclidean')
    dm = dm[dm > 0]
    dist_min = np.amin(dm)
    half_dist_min = dist_min / 2
    chosen_sigma = half_dist_min / k_sigma
    return chosen_sigma, dist_min
