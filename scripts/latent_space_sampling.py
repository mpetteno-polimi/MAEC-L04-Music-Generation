# TODO - Doc

import os

import note_seq
import numpy as np
import tensorflow as tf

from magenta.models.music_vae.configs import CONFIG_MAP

from definitions import ConfigSections
from modules.utilities import config as config_file
from modules.maec.maec_trained_model import MAECTrainedModel

logging = tf.compat.v1.logging
script_config = config_file.load_configuration_section(ConfigSections.LATENT_SPACE_SAMPLING)


def run(config_map):
    """Load model params, save config file and start trainer.

        Args:
            config_map: Dictionary mapping configuration name to Config object.

        Raises:
            ValueError: if required flags are missing or invalid.
    """
    checkpoint_dir_or_path = os.path.expanduser(script_config.get("checkpoint_file"))
    model_config = script_config.get("model_config")
    n_grid_points = script_config.get("n_grid_points")
    n_samples_per_grid_point = script_config.get("n_samples_per_grid_point")
    temperature = script_config.get("temperature")
    rand_seed = script_config.get("rand_seed")
    output_dir = os.path.expanduser(script_config.get("output_dir"))

    if model_config not in config_map:
        raise ValueError('Invalid config name: %s' % model_config)
    config = config_map[model_config]
    config.data_converter.max_tensors_per_item = None

    logging.info('Loading model...')
    model = MAECTrainedModel(config,
                             batch_size=n_samples_per_grid_point,
                             checkpoint_dir_or_path=checkpoint_dir_or_path)

    logging.info('Sampling latent space...')
    results, z_grid, batched_gaussian_samples = model.grid_sample(n_grid_points=n_grid_points,
                                                                  n_samples_per_grid_point=n_samples_per_grid_point,
                                                                  length=config.hparams.max_seq_len,
                                                                  temperature=temperature)

    logging.info('Saving results...')

    # Create main run folder
    run_folder_name = 'config_%s_seed_%d' % (model_config, rand_seed)
    run_folder_path = os.path.join(output_dir, run_folder_name)
    tf.compat.v1.gfile.MakeDirs(run_folder_path)

    for i in range(n_grid_points):
        # Create current grid point folder
        grid_point_folder_name = 'grid_point_%d' % i
        grid_point_folder_path = os.path.join(run_folder_path, grid_point_folder_name)
        tf.compat.v1.gfile.MakeDirs(grid_point_folder_path)

        # Save current mean point coordinate file
        mean_point_coord_file_name = 'mean_pt_coord_%d.npy' % i
        mean_point_coord_file_path = os.path.join(grid_point_folder_path, mean_point_coord_file_name)
        mean_point_coord = z_grid[i, :]
        np.save(mean_point_coord_file_path, mean_point_coord)

        # Save gaussian samples results
        for j in range(n_samples_per_grid_point):
            # Create current sample folder
            sample_folder_name = 'sample_%d' % j
            sample_folder_path = os.path.join(grid_point_folder_path, sample_folder_name)
            tf.compat.v1.gfile.MakeDirs(sample_folder_path)

            # Save current sample coordinate file
            sample_coord_file_name = 'sample_coord_%d.npy' % j
            sample_coord_file_path = os.path.join(sample_folder_path, sample_coord_file_name)
            sample_coord = batched_gaussian_samples[j, i, :]
            np.save(sample_coord_file_path, sample_coord)

            # Save current sample MIDI output
            sample_midi_file_name = 'sample_midi_out_%d.mid' % j
            sample_midi_file_path = os.path.join(sample_folder_path, sample_midi_file_name)
            sample_note_sequence = results[i*j]
            note_seq.sequence_proto_to_midi_file(sample_note_sequence, sample_midi_file_path)


def main(_):
    logging.set_verbosity(script_config.get("log"))
    run(CONFIG_MAP)


def console_entry_point():
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.app.run(main)


if __name__ == '__main__':
    console_entry_point()
