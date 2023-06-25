# TODO - Doc

import os

import tensorflow as tf

from magenta.models.music_vae.configs import CONFIG_MAP

from definitions import ConfigSections
from modules.utilities import config as config_file, sampling, file_system
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
    grid_width = script_config.get("grid_width")
    n_samples_per_grid_point = script_config.get("n_samples_per_grid_point")
    k_sigma = script_config.get('k_sigma')
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
    z_grid = sampling.latin_hypercube_sampling(
        d=config.hparams.z_size,
        grid_width=grid_width,
        n_grid_points=n_grid_points,
        rand_seed=rand_seed
    )
    sigma, min_point_distance = sampling.get_sigma_from_grid_points(z_grid, k_sigma)
    tf.compat.v1.logging.info("Minimum distance between grid points is %s" % min_point_distance)
    tf.compat.v1.logging.info("Setting sigma to %s" % sigma)
    results, batched_gaussian_samples = model.grid_sample(grid_points=z_grid,
                                                          n_samples_per_grid_point=n_samples_per_grid_point,
                                                          sigma=sigma,
                                                          length=config.hparams.max_seq_len,
                                                          temperature=temperature)

    logging.info('Saving results...')
    run_folder_name = 'config_%s_seed_%d' % (model_config, rand_seed)
    file_system.save_grid_sampling_results(output_dir=os.path.join(output_dir, run_folder_name),
                                           results=results,
                                           z_grid=z_grid,
                                           batched_gaussian_samples=batched_gaussian_samples)


def main(_):
    logging.set_verbosity(script_config.get("log"))
    run(CONFIG_MAP)


def console_entry_point():
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.app.run(main)


if __name__ == '__main__':
    console_entry_point()
