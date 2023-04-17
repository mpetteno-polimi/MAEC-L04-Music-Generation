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
from pathlib import Path

import note_seq
from magenta.common import merge_hparams
import numpy as np
from magenta.models.music_vae import data, lstm_models

from definitions import Paths, ConfigSections
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


def get_tfrecords_path_for_source_datasets(source_datasets, input_path: Path, mode_label: str, collection_name: str) \
    -> [str]:
    """ TODO - Function DOC """

    tfrecord_file_patterns = map(lambda source_dataset: source_dataset.name + "-{}_{}.tfrecord", source_datasets)
    tfrecord_paths = [list(input_path.glob(e.format(mode_label, collection_name))) for e in tfrecord_file_patterns]
    tfrecord_paths = [j for i in tfrecord_paths for j in i]
    tfrecord_paths = [str(path) for path in tfrecord_paths]
    return tfrecord_paths


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
    config_map = configs.CONFIG_MAP[config_file.get('config_map_name')]
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
        ValueError: if required fields are missing or invalid or if config section name is not valid.
    """
    if section_name not in ConfigSections.__dict__.values():
        raise ValueError('Configuration section name not Valid')

    if section_name is 'Model':
        check_model_config()

    if section_name is 'Training':
        check_training_config()

    if section_name is 'Testing':
        check_testing_config()

    if section_name is 'Generation':
        check_generation_config()


def check_model_config():
    model_config = load_configuration_section(ConfigSections.MODEL)
    if model_config.get('config_map_name') not in configs.CONFIG_MAP:
        raise ValueError('Invalid `config_map_name`: %s' % model_config.get('config_map_name'))

    if not isinstance(model_config.get('batch_size'), int):
        raise ValueError('Invalid batch size`: %s' % model_config.get('batch_size'))


def check_training_config():
    config_file = load_configuration_section(ConfigSections.TRAINING)
    check_model_config()
    model_config = load_configuration_section(ConfigSections.MODEL)


    if config_file.get('model_checkpoint_file_dir') is None:
        raise ValueError('`model_checkpoint_file_dir` should be specified in the config.ini file')

    configuration = configs.CONFIG_MAP[model_config.get('config_map_name')]
    if config_file.get('h_params'):
        configuration.hparams.parse(config_file.get('h_params'))

    if config_file.get('checkpoints_max_num') is None:
        raise ValueError('`checkpoints_max_num` should be specified in the config.ini file')

    if config_file.get('hours_between_checkpoints') is None:
        raise ValueError('`hours_between_checkpoints` should be specified in the config.ini file')

    if config_file.get('num_steps') is None:
        raise ValueError('`num_steps` should be specified in the config.ini file')
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


def check_testing_config():
    config_file = load_configuration_section(ConfigSections.TESTING)

    if config_file.get('num_batches') is not None \
        and not isinstance(config_file.get('num_batches'), int):
        raise ValueError('`num_batches` should be specified in the config.ini file')


def check_generation_config():
    config_file = load_configuration_section(ConfigSections.GENERATION)

    if config_file.get('model_checkpoint_file_path') is None:
        raise ValueError('`model_checkpoint_file_path` should be specified in the config.ini file')

    if config_file.get('output_directory') is None:
        raise ValueError('`output_directory` is required in the config.ini file.')
    tf.io.gfile.mkdir(config_file.get('output_directory'))

    if config_file.get('num_output_files') is None:
        raise ValueError('`num_output_files` is required in the config.ini file.')

    if config_file.get('model_mode') != 'sample' and config_file.get('model_mode') != 'interpolate':
        raise ValueError('Invalid value for `model_mode`: %s' % config_file.get('model_mode'))

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

def slerp(p0, p1, t):
    """Spherical linear interpolation."""
    omega = np.arccos(
        np.dot(np.squeeze(p0 / np.linalg.norm(p0)),
               np.squeeze(p1 / np.linalg.norm(p1))))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1
