# TODO - Doc

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.stats import pearsonr

from definitions import ConfigSections
from modules.utilities import config as config_file

logging = tf.compat.v1.logging
script_config = config_file.load_configuration_section(ConfigSections.LATENT_SPACE_SAMPLING)


def load_matrices(run_folder_path):
    grid_points_coordinates = []
    sample_coordinates = []
    sample_complexities = []

    for grid_point_folder_name in os.listdir(run_folder_path):

        # Load grid points coordinates
        grid_point_coord_file_name = 'mean_pt_coord_%s.npy' % grid_point_folder_name.split(sep="_")[-1]
        grid_point_folder_path = os.path.join(run_folder_path, grid_point_folder_name)
        grid_point_file_path = os.path.join(grid_point_folder_path, grid_point_coord_file_name)
        grid_point_coord = np.load(grid_point_file_path)
        grid_points_coordinates.append(grid_point_coord)

        grid_point_samples_complexities = []
        grid_point_samples_coord = []
        for sample_folder_name in os.listdir(grid_point_folder_path):
            sample_folder_path = os.path.join(grid_point_folder_path, sample_folder_name)
            if os.path.isdir(sample_folder_path):
                sample_folder_idx = sample_folder_name.split(sep='_')[-1]

                # Load sample complexities
                sample_complexities_file_name = 'sample_complexities_%s.npy' % sample_folder_idx
                sample_complexity_file_path = os.path.join(sample_folder_path, sample_complexities_file_name)
                sample_complexity = np.load(sample_complexity_file_path)
                grid_point_samples_complexities.append(sample_complexity)

                # Load sample coordinates
                sample_coord_file_name = 'sample_coord_%s.npy' % sample_folder_idx
                sample_coord_file_path = os.path.join(sample_folder_path, sample_coord_file_name)
                sample_coord = np.load(sample_coord_file_path)
                grid_point_samples_coord.append(sample_coord)

        sample_coordinates.append(grid_point_samples_coord)
        sample_complexities.append(grid_point_samples_complexities)

    return np.asarray(grid_points_coordinates), np.asarray(sample_coordinates), np.asarray(sample_complexities)


def evaluate(grid_points_coordinates, sample_coordinates, sample_complexities):
    correlation(sample_coordinates, sample_complexities)
    multi_dimensional_scaling(sample_coordinates, sample_complexities)


def correlation(sample_coordinates, sample_complexities):
    # TODO - Probably there's a quicker way to do this
    # flatten to 2 dim
    coordinates = sample_coordinates.reshape(-1, sample_coordinates.shape[-1])
    coordinates = np.transpose(coordinates, (1, 0))
    complexities = sample_complexities.reshape(-1, sample_complexities.shape[-1])
    complexities = np.transpose(complexities, (1, 0))

    # for each dimension compute pearson correlation and store then in a array
    pearson_correlations = []
    for complexity in complexities:
        correlations = []
        for coord in coordinates:
            corr = pearsonr(coord, complexity)
            correlations.append(corr)
        pearson_correlations.append(correlations)

    # TODO - Do something with correlation coefficients

    return np.asarray(pearson_correlations)


def multi_dimensional_scaling(sample_coordinates, sample_complexities):
    # TODO - Review
    # flatten to 2 dim
    sample_coordinates_unrolled = sample_coordinates.reshape(-1, sample_coordinates.shape[-1])
    sample_complexities = sample_complexities.reshape(-1, sample_complexities.shape[-1])

    sample_complexities_norm = sample_complexities[:, 0]
    # sample_complexities_norm = (sample_complexities_norm - np.min(sample_complexities_norm)) /
    # (np.max(sample_complexities_norm) - np.min(sample_complexities_norm))

    multi_dim_scaler = MDS(n_components=2, verbose=2, max_iter=100)
    scaled_sample_coordinates = multi_dim_scaler.fit_transform(sample_coordinates_unrolled[0:100])

    print('sample_coordinates_2d.shape', sample_coordinates.shape)
    print('scaled_sample_coordinates', scaled_sample_coordinates.shape)
    print('scaled_sample_coordinates max', np.min(scaled_sample_coordinates))
    print('scaled_sample_coordinates min', np.min(scaled_sample_coordinates))
    print('sample_complexities_norm', sample_complexities_norm.shape)
    print('sample_complexities_norm max', np.max(sample_complexities_norm))
    print('sample_complexities_norm min', np.min(sample_complexities_norm))

    plt.figure(0)
    fig, ax = plt.subplots()

    ax.scatter(scaled_sample_coordinates[0:100, 0], scaled_sample_coordinates[0:100, 1],
               c=sample_complexities_norm[0:100], s=sample_complexities_norm[0:100], vmin=0, vmax=50)
    ax.set(xlim=(-100, 100), xticks=np.arange(-100, 100),
           ylim=(-100, 100), yticks=np.arange(-100, 100))

    plt.show()


def run():
    output_dir = os.path.expanduser(script_config.get("output_dir"))
    model_config = script_config.get("model_config")
    rand_seed = script_config.get("rand_seed")
    run_folder_name = 'config_%s_seed_%d' % (model_config, rand_seed)
    run_folder_path = os.path.join(output_dir, run_folder_name)

    grid_points_coordinates, sample_coordinates, sample_complexities = load_matrices(run_folder_path)

    # Save matrices (for sharing purposes)
    np.save(os.path.join(run_folder_path, 'grid_points_coordinates'), grid_points_coordinates)
    np.save(os.path.join(run_folder_path, 'sample_coordinates'), sample_coordinates)
    np.save(os.path.join(run_folder_path, 'sample_complexities'), sample_complexities)

    evaluate(grid_points_coordinates, sample_coordinates, sample_complexities)


def main(_):
    logging.set_verbosity(script_config.get("log"))
    run()


def console_entry_point():
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.app.run(main)


if __name__ == '__main__':
    console_entry_point()
