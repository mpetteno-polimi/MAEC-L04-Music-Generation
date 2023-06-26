# TODO - Doc

import os
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from magenta.models.music_vae.configs import CONFIG_MAP
from scipy.stats import pearsonr
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.manifold import MDS

import latent_space_complexities
from definitions import ConfigSections
from modules.maec.maec_trained_model import MAECTrainedModel
from modules.utilities import config as config_file, file_system

logging = tf.compat.v1.logging
script_config = config_file.load_configuration_section(ConfigSections.LATENT_SPACE_SAMPLING)
complexities_methods = ['toussaint', 'note density', 'pitch range', 'contour']


def load_matrices(grid_folder_path):
    grid_points_coordinates = []
    sample_coordinates = []
    sample_complexities = []

    for grid_point_folder_name in os.listdir(grid_folder_path):
        # Load grid points coordinates
        grid_point_coord_file_name = 'mean_pt_coord_%s.npy' % grid_point_folder_name.split(sep="_")[-1]
        grid_point_folder_path = os.path.join(grid_folder_path, grid_point_folder_name)
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
                if os.path.isfile(sample_complexity_file_path):
                    sample_complexity = np.load(sample_complexity_file_path)
                    grid_point_samples_complexities.append(sample_complexity)
                else:
                    grid_point_samples_complexities.append([np.nan, np.nan, np.nan, np.nan])

                # Load sample coordinates
                sample_coord_file_name = 'sample_coord_%s.npy' % sample_folder_idx
                sample_coord_file_path = os.path.join(sample_folder_path, sample_coord_file_name)
                sample_coord = np.load(sample_coord_file_path)
                grid_point_samples_coord.append(sample_coord)

        sample_coordinates.append(grid_point_samples_coord)
        sample_complexities.append(grid_point_samples_complexities)

    return np.asarray(grid_points_coordinates), np.asarray(sample_coordinates), np.asarray(sample_complexities)


def evaluate(output_dir, grid_points_coordinates, sample_coordinates, sample_complexities, model):
    correlation(output_dir, grid_points_coordinates, sample_coordinates, sample_complexities)
    multi_dimensional_scaling(output_dir, grid_points_coordinates, sample_complexities)
    histograms(output_dir, grid_points_coordinates, sample_complexities, model)


def correlation(output_dir, grid_points_coordinates, sample_coordinates, sample_complexities):
    def compute_correlation_coefficients(coordinates, complexities, dimensions):
        coefficients = np.zeros(dimensions)
        for i in range(dimensions):
            coefficients[i], __ = pearsonr(coordinates[:, i], complexities)
        return coefficients

    def plot_correlation_coefficients(coefficients, output_folder_path):
        plt.figure()
        plt.stem(coefficients)
        plt.xlabel('Latent dimension')
        plt.ylabel('Pearson coefficient')
        plt.grid(linestyle=':')
        plt.savefig(output_folder_path)

    correlation_folder_path = os.path.join(output_dir, 'correlation')
    tf.compat.v1.gfile.MakeDirs(correlation_folder_path)
    n_dimension = sample_coordinates.shape[2]
    sample_complexities_mean = np.nanmean(sample_complexities, axis=1)

    # Flatten first two dimensions
    flatten_coordinates = sample_coordinates.reshape(-1, n_dimension)
    flatten_complexities = sample_complexities.reshape(-1, sample_complexities.shape[2])

    # Filter NaN values from flatten arrays
    flatten_coordinates = flatten_coordinates[~np.isnan(flatten_complexities).any(axis=1), :]
    flatten_complexities = flatten_complexities[~np.isnan(flatten_complexities).any(axis=1), :]

    # Plot correlation between all complexities methods
    cmat = np.corrcoef(flatten_complexities.T)
    plt.figure()
    sns.heatmap(cmat, annot=True, linewidth=1., xticklabels=complexities_methods, yticklabels=complexities_methods)
    plt.savefig(os.path.join(correlation_folder_path, 'complexities_methods_correlation.png'))

    # Plot correlation between latent space dimensions and complexity methods
    for complexity_id, complexities_method in enumerate(complexities_methods):
        # Create output directory
        complexity_folder_path = os.path.join(correlation_folder_path, complexities_method.lower())
        tf.compat.v1.gfile.MakeDirs(complexity_folder_path)

        # Plot correlation coefficient for current complexity values
        current_complexity_values = flatten_complexities[:, complexity_id]
        rho_x = compute_correlation_coefficients(flatten_coordinates, current_complexity_values, n_dimension)
        output_path = os.path.join(complexity_folder_path, '%s_values_correlation.png' % complexities_method)
        plot_correlation_coefficients(rho_x, output_path)

        # Plot correlation coefficient for current complexities mean
        current_complexity_mean = sample_complexities_mean[:, complexity_id]
        rho = compute_correlation_coefficients(grid_points_coordinates, current_complexity_mean, n_dimension)
        output_path = os.path.join(complexity_folder_path, '%s_mean_correlation.png' % complexities_method)
        plot_correlation_coefficients(rho, output_path)

        # Plot mean complexities of the two most correlated dimensions in 2D
        rho_s = np.argsort(np.abs(rho))
        zx = rho_s[-1]
        zy = rho_s[-2]
        plt.figure()
        plt.scatter(grid_points_coordinates[:, zx], grid_points_coordinates[:, zy],
                    c=current_complexity_mean,
                    cmap=plt.cm.bwr)
        plt.xlabel(f'z_{zx}')
        plt.ylabel(f'z_{zy}')
        plt.colorbar()
        plt.grid(linestyle=':')
        plt.savefig(os.path.join(complexity_folder_path, '%s_top2_corr_mean_2d.png' % complexities_method))

        # Plot mean complexities of the two most correlated dimensions in 3D
        xyz = {'x': grid_points_coordinates[:, zx], 'y': grid_points_coordinates[:, zy], 'z': current_complexity_mean}
        # put the data into a pandas DataFrame (this is what my data looks like)
        df = pd.DataFrame(xyz, index=range(len(xyz['x'])))
        # re-create the 2D-arrays
        x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()))
        y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
        x2, y2 = np.meshgrid(x1, y1)
        z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='cubic')
        fig = plt.figure(dpi=150)
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=plt.cm.bwr, linewidth=1., antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=8)
        ax.view_init(20, 45)
        plt.savefig(os.path.join(complexity_folder_path, '%s_top2_corr_mean_3d.png' % complexities_method))


def multi_dimensional_scaling(output_dir, grid_points_coordinates, sample_complexities):
    mds_folder_path = os.path.join(output_dir, 'mds')
    tf.compat.v1.gfile.MakeDirs(mds_folder_path)
    sample_complexities_mean = np.nanmean(sample_complexities, axis=1)

    for complexity_id, complexities_method in enumerate(complexities_methods):
        # Create output directory
        complexity_folder_path = os.path.join(mds_folder_path, complexities_method.lower())
        tf.compat.v1.gfile.MakeDirs(complexity_folder_path)

        current_complexity_mean = sample_complexities_mean[:, complexity_id]

        # Plot MDS
        reducer = MDS(verbose=1, random_state=1)
        r = reducer.fit_transform(grid_points_coordinates)
        plt.figure()
        plt.scatter(r[:, 0], r[:, 1], c=current_complexity_mean, cmap='bwr')
        plt.colorbar()
        plt.grid(linestyle=':')
        plt.savefig(os.path.join(complexity_folder_path, '%s_mds.png' % complexities_method))

        # Plot MDS with tricontour
        xx, yy, zz = r[:, 0], r[:, 1], current_complexity_mean
        fig, ax = plt.subplots(1)
        ax.tricontour(xx, yy, zz, levels=21, linewidths=0.2, colors='k')
        cntr2 = ax.tricontourf(xx, yy, zz, levels=21, cmap="bwr", antialiased=True)
        fig.colorbar(cntr2, ax=ax)
        ax.plot(xx, yy, 'ko', ms=4)
        plt.tight_layout()
        plt.savefig(os.path.join(complexity_folder_path, '%s_mds_tricontour.png' % complexities_method))


def histograms(output_dir, grid_points_coordinates, sample_complexities, model):
    sample_complexities_mean = np.nanmean(sample_complexities, axis=1)
    histograms_folder_name = 'histograms'
    histograms_folder_path = os.path.join(output_dir, histograms_folder_name)

    for complexity_id, complexities_method in enumerate(complexities_methods):
        # Create complexity method output folder
        complexity_folder_path = os.path.join(histograms_folder_path, complexities_method)
        tf.compat.v1.gfile.MakeDirs(complexity_folder_path)

        # Divide complexity mean values in three ranges (low, mid, high)
        ranges_names = ['low', 'mid', 'high']
        current_complexity_mean = sample_complexities_mean[:, complexity_id]
        sorted_complexities_mean_idx = np.argsort(current_complexity_mean, axis=0)
        ranges_idx = np.array_split(sorted_complexities_mean_idx, 3)

        range_complexities = np.zeros(shape=(sample_complexities.shape[1], 3))
        for idx, range_idx in enumerate(ranges_idx):
            # Create range output folder
            range_folder_path = os.path.join(complexity_folder_path, ranges_names[idx])

            if not os.path.exists(range_folder_path):
                # Create main directory
                tf.compat.v1.gfile.MakeDirs(range_folder_path)

                # Pick three complexity values from the range and find their corresponding coordinates
                # random_points_idx = np.random.choice(range_idx, size=3)
                # random_points = grid_points_coordinates[random_points_idx]
                #
                # # Compute the centroid
                # centroid = np.mean(random_points, axis=0)
                # z_grid = np.asarray([centroid])

                # Choose the grid point - TODO temporary
                if idx == 0:
                    # Low range
                    chosen_point_idx = range_idx[0]
                elif idx == 1:
                    # Mid range
                    chosen_point_idx = range_idx[int((len(range_idx) - 1) / 2)]
                elif idx == 2:
                    # High range
                    chosen_point_idx = range_idx[-1]
                else:
                    chosen_point_idx = 0
                z_grid = np.asarray([grid_points_coordinates[chosen_point_idx]])

                # Batch sampling around chosen point
                logging.info('Sampling latent space...')
                results, batched_gaussian_samples = model.grid_sample(grid_points=z_grid,
                                                                      n_samples_per_grid_point=
                                                                      sample_complexities.shape[1],
                                                                      sigma=1)
                logging.info('Saving results...')
                range_samples_folder_path = file_system.save_grid_sampling_results(range_folder_path, results, z_grid,
                                                                                   batched_gaussian_samples)
            else:
                range_samples_folder_path = os.path.join(range_folder_path, "samples")

            # Compute complexities
            complexity_values = latent_space_complexities.run(range_samples_folder_path, metrics=[complexities_method])
            range_complexities[:, idx] = complexity_values[:, 0]

            # Plot continuous histogram of complexity
            plt.figure()
            sns.histplot(range_complexities[:, idx], bins=20, kde=True, legend=False)
            plt.xlabel(complexities_method)
            plt.legend(title='Ranges', loc='upper right', labels=[ranges_names[idx]])
            plt.savefig(os.path.join(range_folder_path, '%s_%s_histogram.png' % (ranges_names[idx],
                                                                                 complexities_method)))

        # Plot continuous histogram of complexities
        plt.figure()
        sns.histplot(range_complexities, bins=20, kde=True, legend=False)
        plt.xlabel(complexities_method)
        plt.legend(title='Ranges', loc='upper right', labels=ranges_names)
        plt.savefig(os.path.join(complexity_folder_path, '%s_ranges_histogram.png' % complexities_method))

    # Compute statistical test pairwise
    # scipy.stats.ttest_ind(complexity_values['low'], complexity_values['high'], equal_var=False)
    # scipy.stats.mannwhitneyu()


def run(config_map):
    output_dir = os.path.expanduser(script_config.get("output_dir"))
    model_config = script_config.get("model_config")
    checkpoint_dir_or_path = os.path.expanduser(script_config.get("checkpoint_file"))
    n_samples_per_grid_point = script_config.get("n_samples_per_grid_point")
    rand_seed = script_config.get("rand_seed")

    # Load model
    if model_config not in config_map:
        raise ValueError('Invalid config name: %s' % model_config)
    config = config_map[model_config]
    config.data_converter.max_tensors_per_item = None

    logging.info('Loading model...')
    model = MAECTrainedModel(config,
                             batch_size=n_samples_per_grid_point,
                             checkpoint_dir_or_path=checkpoint_dir_or_path)

    # Load matrices
    run_folder_name = 'config_%s_seed_%d' % (model_config, rand_seed)
    samples_folder_name = 'samples'
    samples_folder_path = os.path.join(output_dir, run_folder_name, samples_folder_name)
    evaluation_folder_path = os.path.join(output_dir, run_folder_name, 'evaluation')
    tf.compat.v1.gfile.MakeDirs(evaluation_folder_path)
    grid_points_coordinates, sample_coordinates, sample_complexities = load_matrices(samples_folder_path)

    # Save matrices (for sharing purposes)
    matrices_folder_path = os.path.join(evaluation_folder_path, 'matrices')
    tf.compat.v1.gfile.MakeDirs(matrices_folder_path)
    np.save(os.path.join(matrices_folder_path, 'grid_points_coordinates'), grid_points_coordinates)
    np.save(os.path.join(matrices_folder_path, 'sample_coordinates'), sample_coordinates)
    np.save(os.path.join(matrices_folder_path, 'sample_complexities'), sample_complexities)

    evaluate(evaluation_folder_path, grid_points_coordinates, sample_coordinates, sample_complexities, model)


def main(_):
    logging.set_verbosity(script_config.get("log"))
    run(CONFIG_MAP)


def console_entry_point():
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.app.run(main)


if __name__ == '__main__':
    console_entry_point()
