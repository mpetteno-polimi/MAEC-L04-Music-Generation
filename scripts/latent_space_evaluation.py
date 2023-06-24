import os
import numpy as np
import tensorflow as tf

from definitions import ConfigSections
from modules.utilities import config as config_file

logging = tf.compat.v1.logging
script_config = config_file.load_configuration_section(ConfigSections.LATENT_SPACE_SAMPLING)


def run():
    output_dir = os.path.expanduser(script_config.get("output_dir"))
    model_config = script_config.get("model_config")
    rand_seed = script_config.get("rand_seed")

    run_folder_name = 'config_%s_seed_%d' % (model_config, rand_seed)
    run_folder_path = os.path.join(output_dir, run_folder_name)

    grid_points_coordinates = []
    sample_coordinates = []
    sample_complexities = []

    for grid_point_folder_name in os.listdir(run_folder_path):
        # save grid pt coordinates
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
                sample_folder_idx = sample_folder_name.split(sep='_')[-1].split(sep='.')[0]
                # save coords
                sample_coord_file_name = 'sample_complexities_%s.npy' % sample_folder_idx
                sample_coord_file_path = os.path.join(sample_folder_path, sample_coord_file_name)
                sample_coord = np.load(sample_coord_file_path)
                grid_point_samples_complexities.append(sample_coord)

                # save complexity
                sample_complexity_file_name = 'sample_coord_%s.npy' % sample_folder_idx
                sample_complexity_file_path = os.path.join(sample_folder_path, sample_complexity_file_name)
                sample_complexity = np.load(sample_complexity_file_path)
                grid_point_samples_coord.append(sample_complexity)

        sample_coordinates.append(grid_point_samples_coord)
        sample_complexities.append(grid_point_samples_complexities)

    grid_points_coordinates = np.asarray(grid_points_coordinates)
    sample_coordinates = np.asarray(sample_coordinates)
    sample_complexities = np.asarray(sample_complexities)

    np.save(os.path.join(output_dir, 'grid_points_coordinates'), grid_points_coordinates)
    np.save(os.path.join(output_dir, 'sample_coordinates'), sample_coordinates)
    np.save(os.path.join(output_dir, 'sample_complexities'), sample_complexities)

def main(_):
    logging.set_verbosity(script_config.get("log"))
    run()


def console_entry_point():
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.app.run(main)


if __name__ == '__main__':
    console_entry_point()
