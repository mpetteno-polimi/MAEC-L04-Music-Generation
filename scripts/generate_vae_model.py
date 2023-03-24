from magenta.models.music_vae.music_vae_generate import _slerp

from definitions import ConfigSections
from modules import utilities

import magenta.models.music_vae.configs as config
from magenta.models.music_vae import TrainedModel
import numpy as np
import note_seq
import logging
import time
import os


def load_model(config_map):
    """ returns a TrainedModel with specified parameters

      Args:
        config_map: Dictionary mapping configuration name to Config object.

    """
    logging.info('Loading model...')
    checkpoint_dir_or_path = os.path.expanduser(config_file.get('model_checkpoint_file_path'))
    model = TrainedModel(
        config_map,
        batch_size=min(config_file.get('batch_size'), config_file.get('num_output_files')),
        checkpoint_dir_or_path=checkpoint_dir_or_path
    )
    return model


def generate(model, config_map):
    """ Generate midi sequences """
    result = None
    if config_file.get('model_mode') == 'interpolate':
        logging.info('Interpolating...')
        input_midi_1 = os.path.expanduser(config_file.get('midi_input_path_1'))
        input_midi_2 = os.path.expanduser(config_file.get('midi_input_path_2'))
        input_1 = note_seq.midi_file_to_note_sequence(input_midi_1)
        input_2 = note_seq.midi_file_to_note_sequence(input_midi_2)

        _, mu, _ = model.encode([input_1, input_2])
        z = np.array([
            _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, config_file.get('num_output_files'))])
        results = model.decode(length=config_map.hparams.max_seq_len, z=z, )
        # temperature=FLAGS.temperature) # todo: Support temperature: conditioning????
    elif config_file.get('model_mode') == 'sample':
        logging.info('Sampling...')
        results = model.sample(
            n=config_file.get('num_output_files'),
            length=config_map.hparams.max_seq_len, )
        # temperature=FLAGS.temperature) # todo: Support temperature: conditioning????

    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    basename = os.path.join(
        config_file.get('output_directory'),
        '%s_%s_%s-*-of-%03d.mid' %
        (config_file.get('config_map_name'), config_file.get('model_mode'), date_and_time,
         config_file.get('num_output_files'))
    )
    logging.info('Outputting %d files as `%s`...', config_file.get('num_output_files'), basename)
    if results is None:
        raise Exception("model not able to process ")
    for i, ns in enumerate(results):
        note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

    logging.info('Done.')


if __name__ == '__main__':
    """Load model params, load TrainedModel and generates midi sequences."""

    utilities.check_configuration_section(ConfigSections.MODEL)
    config_file = utilities.load_configuration_section(ConfigSections.MODEL)

    config_map = config.CONFIG_MAP[config_file.get('config_map_name')]
    config_map.data_converter.max_tensors_per_item = None

    model = load_model(config_map)
    generate(model, config_map)
