""" Utility methods for other modules.

The module contains all methods that are not related to any specific function or service such as methods to handle
services configuration and file system access.

"""

import ast
import configparser
from typing import Mapping, Any
from definitions import Paths
import tensorflow._api.v2.compat.v1 as tf


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


def trial_summary(hparams, examples_path, output_dir):
    """Writes a tensorboard text summary of the trial."""

    examples_path_summary = tf.summary.text(
        'examples_path', tf.constant(examples_path, name='examples_path'),
        collections=[])

    hparams_dict = hparams.values()

    # Create a markdown table from hparams.
    header = '| Key | Value |\n| :--- | :--- |\n'
    keys = sorted(hparams_dict.keys())
    lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
    hparams_table = header + '\n'.join(lines) + '\n'

    hparam_summary = tf.summary.text(
        'hparams', tf.constant(hparams_table, name='hparams'), collections=[])

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
        writer.add_summary(examples_path_summary.eval())
        writer.add_summary(hparam_summary.eval())
        writer.close()


def get_input_tensors(dataset, config):
    """Get input tensors from dataset."""
    batch_size = config.hparams.batch_size
    iterator = tf.data.make_one_shot_iterator(dataset)
    (input_sequence, output_sequence, control_sequence, sequence_length) = iterator.get_next()
    input_sequence.set_shape(
        [batch_size, None, config.data_converter.input_depth])
    output_sequence.set_shape(
        [batch_size, None, config.data_converter.output_depth])
    if not config.data_converter.control_depth:
        control_sequence = None
    else:
        control_sequence.set_shape(
            [batch_size, None, config.data_converter.control_depth])
    sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())

    return {
        'input_sequence': input_sequence,
        'output_sequence': output_sequence,
        'control_sequence': control_sequence,
        'sequence_length': sequence_length
    }
