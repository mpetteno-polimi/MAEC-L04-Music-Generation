""" Utility methods for other modules.

The module contains all methods that are not related to any specific function or service such as methods to handle
services configuration and file system access.

"""

import ast
import configparser
import os
import logging
import sys
from datetime import time
from typing import Mapping, Any

import note_seq
from magenta.common import merge_hparams
from magenta.models.music_vae import MusicVAE, data, lstm_models

from definitions import Paths
import tensorflow as tf
import magenta.models.music_vae.configs as configs

from magenta.contrib import training as contrib_training
HParams = contrib_training.HParams



def load_configuration_file() -> configparser.ConfigParser:
    """Loads the configuration file specified in modules.constants.Paths.MAIN_CONFIG_FILE.
    Files that cannot be opened are silently ignored.

    Returns:
        configparser.ConfigParser:
            An object representing the configuration file.

    """

    config_file = configparser.ConfigParser()
    config_file.read(Paths.MAIN_CONFIG_FILE)
    return config_file


def load_configuration_section(section_name: str) -> Mapping[str, Any]:
    """Loads a section of the configuration file specified in modules.constants.Paths.MAIN_CONFIG_FILE.

    Args:
        section_name (str):
            The name of the section to load.

    Returns:
        The loaded config section

    """

    config_file = load_configuration_file()
    file_section = dict()
    for option in config_file.options(section_name):
        option_value = config_file.get(section_name, option)
        try:
            file_section[option] = ast.literal_eval(option_value)
        except (SyntaxError, ValueError):
            file_section[option] = option_value
    return file_section


def _check_extract_examples(input_midi, path, input_number, config_file):
    """Make sure each input returns exactly one example from the converter.
        todo: finish DOC
        Args:
            input_midi:
            path:
            input_number:
            config_file:
    """
    input_ns = note_seq.midi_file_to_note_sequence(input_midi)
    config_map = config.CONFIG_MAP[config_file.get('config_map_name')]
    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    tensors = config_map.data_converter.to_tensors(input_ns).outputs
    if not tensors:
        print(
            'MusicVAE configs have very specific input requirements. Could not '
            'extract any valid inputs from `%s`. Try another MIDI file.' % path)
        sys.exit()
    elif len(tensors) > 1:
        base_dir_name = os.path.join(
            config_file.get('output_directory'),
            '%s_input%d-extractions_%s-*-of-%03d.mid' %
            (config_file.get('config_map_name'), input_number, date_and_time, len(tensors)))
        for i, ns in enumerate(config_map.data_converter.from_tensors(tensors)):
            note_seq.sequence_proto_to_midi_file(ns, base_dir_name.replace('*', '%03d' % i))
        print('%d valid inputs extracted from `%s`. Outputting these potential inputs as `%s`. Call script again with '
              'one of these instead.' % (len(tensors), path, base_dir_name))
        sys.exit()


def check_configuration_section(section_name: str):
    """From a specific section of the config.ini file, check if all needed parameters are set
    Args:
        section_name (str):
            The name of the section to check.

    Raises:
        ValueError: if required fields are missing or invalid.
    """
    config_file = load_configuration_section(section_name)
    if section_name is 'Model':
        if config_file.get('model_checkpoint_file_path') is None:
            raise ValueError('`model_checkpoint_file_path` should be specified in the config.ini file')
        if config_file.get('output_directory') is None:
            raise ValueError('`output_directory` is required in the config.ini file.')
        tf.io.gfile.mkdir(config_file.get('output_directory'))
        if config_file.get('model_mode') != 'sample' and config_file.get('model_mode') != 'interpolate':
            raise ValueError('Invalid value for `model_mode`: %s' % config_file.get('model_mode'))

        if config_file.get('config_map_name') not in configs.CONFIG_MAP:
            raise ValueError('Invalid `config_map_name`: %s' % config_file.get('config_map_name').config)
        if config_file.get('config_mode') == 'interpolate':
            if config_file.get('midi_input_path_1') is None or config_file.get('midi_input_path_2') is None:
                raise ValueError('`midi_input_path_1` and `midi_input_path_2` must be specified in `interpolate` mode.')
            input_midi_1 = os.path.expanduser(config_file.get('midi_input_path_1'))
            input_midi_2 = os.path.expanduser(config_file.get('midi_input_path_2'))
            if not os.path.exists(input_midi_1):
                raise ValueError('`midi_input_path_1` not found: %s' % config_file.get('midi_input_path_1'))
            if not os.path.exists(input_midi_2):
                raise ValueError('`midi_input_path_2` not found: %s' % config_file.get('midi_input_path_2'))
            logging.info(
                'Attempting to extract examples from input MIDIs using config `%s`...',
                config_file.get('config_map_name'))
            _check_extract_examples(input_midi_1, config_file.get('midi_input_path_1'), 1, config_file)
            _check_extract_examples(input_midi_2, config_file.get('midi_input_path_2'), 2, config_file)


    if section_name is 'Training':
        if config_file.get('model_checkpoint_file_dir') is None:
            raise ValueError('`model_checkpoint_file_dir` should be specified in the config.ini file')
        if config_file.get('model_mode') not in ['train', 'eval']:
            raise ValueError('Invalid mode: %s' % config_file.get('model_mode'))

        if config_file.get('config_map_name') not in configs.CONFIG_MAP:
            raise ValueError('Invalid config: %s' % config_file.get('config_map_name'))
        configuration = configs.CONFIG_MAP['config_map_name']
        if config_file.get('h_params'):
            configuration.hparams.parse(config_file.get('h_params'))
        config_update_map = {}


    #    if FLAGS.examples_path:
    #        config_update_map['%s_examples_path' % FLAGS.mode] = os.path.expanduser(
    #            FLAGS.examples_path)
    #
    #    # todo: do we need this? No i think
    #    if FLAGS.tfds_name:
    #        if FLAGS.examples_path:
    #            raise ValueError(
    #                'At most one of --examples_path and --tfds_name can be set.')
    #        config_update_map['tfds_name'] = FLAGS.tfds_name
    #        config_update_map['eval_examples_path'] = None
    #        config_update_map['train_examples_path'] = None

    # TODO: multithreading
    #    if FLAGS.num_sync_workers:
    #        config.hparams.batch_size //= FLAGS.num_sync_workers

        if config_file.get('model_mode') == 'train':
            is_training = True
        elif config_file.get('model_mode') == 'eval':
            is_training = False
        else:
            raise ValueError('Invalid mode: {}'.format(config_file.get('model_mode')))


# TODO: add method able to generate a magenta config from config.ini file settings
def add_magenta_config(config_name, model, hparams):
    """
    Add a map to the list of available config maps of magenta configs
    """
    from magenta.models.music_vae.configs import CONFIG_MAP, Config

    CONFIG_MAP[config_name] = Config(
        model=model,
        hparams=merge_hparams(
            lstm_models.get_default_hparams(),
            hparams
        ),
        note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
        data_converter=data.OneHotMelodyConverter(
            valid_programs=data.MEL_PROGRAMS,
            skip_polyphony=False,
            max_bars=100,  # Truncate long melodies before slicing.
            slice_bars=2,
            steps_per_quarter=4),
        train_examples_path=None,
        eval_examples_path=None,
    )
