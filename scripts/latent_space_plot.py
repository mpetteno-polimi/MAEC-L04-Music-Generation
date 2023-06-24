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


def run_mds():
    output_dir = os.path.expanduser(script_config.get("output_dir"))
    model_config = script_config.get("model_config")
    rand_seed = script_config.get("rand_seed")

    grid_pt_coords, sample_coordinates, sample_complexities = load_samples(output_dir)

    # flatten to 2 dim
    sample_coordinates_unrolled = sample_coordinates.reshape(-1, sample_coordinates.shape[-1])
    sample_complexities = sample_complexities.reshape(-1, sample_complexities.shape[-1])

    sample_complexities_norm = sample_complexities[:, 0]
    # sample_complexities_norm = (sample_complexities_norm - np.min(sample_complexities_norm)) / (np.max(sample_complexities_norm) - np.min(sample_complexities_norm))

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
    # plot
    fig, ax = plt.subplots()

    ax.scatter(scaled_sample_coordinates[0:100, 0], scaled_sample_coordinates[0:100, 1], c=sample_complexities_norm[0:100], s=sample_complexities_norm[0:100], vmin=0, vmax=50)
    ax.set(xlim=(-100, 100), xticks=np.arange(-100, 100),
           ylim=(-100, 100), yticks=np.arange(-100, 100))

    plt.show()


# correlazione (coeff. di Pearson) tra il vettore di N complessità e ciascuna delle 256 dimensioni della matrice z. (modificato)
def run_pearson():
    output_dir = os.path.expanduser(script_config.get("output_dir"))
    model_config = script_config.get("model_config")
    rand_seed = script_config.get("rand_seed")

    grid_pt_coords, sample_coordinates, sample_complexities = load_samples(output_dir)
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

    pearson_correlations = np.asarray(pearson_correlations)
    print('pearson_corr shape', pearson_correlations.shape)


def load_samples(output_dir):

    grid_pt_coords = np.load(os.path.join(output_dir, 'grid_points_coordinates.npy'), allow_pickle=True,
                             fix_imports=True)
    sample_coordinates = np.load(os.path.join(output_dir, 'sample_coordinates.npy'), allow_pickle=True,
                                 fix_imports=True)
    sample_complexities = np.load(os.path.join(output_dir, 'sample_complexities.npy'), allow_pickle=True,
                                  fix_imports=True)
    return grid_pt_coords, sample_coordinates, sample_complexities




def main(_):
    logging.set_verbosity(script_config.get("log"))
    run_pearson()


def console_entry_point():
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.app.run(main)


def test():
    rand_arr_coord = np.random.randn(3200, 2)
    rand_arr_compl = np.random.uniform(10, 20, 3200)

    plt.figure(0)
    # plot
    fig, ax = plt.subplots()

    ax.scatter(rand_arr_coord[:, 0], rand_arr_coord[:, 1], c=rand_arr_compl, vmin=0, vmax=100)
    ax.set(xlim=(-8, 8), xticks=np.arange(-8, 8),
           ylim=(-8, 8), yticks=np.arange(-8, 8))

    plt.show()


if __name__ == '__main__':
    # test()
    console_entry_point()
