""" Utility methods for other modules.

The module contains all methods that are not related to any specific function or service such as methods to handle
services configuration and file system access.

"""

import ast
import configparser
import hashlib
from pathlib import Path
from typing import Mapping, Any
from definitions import Paths


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


def generate_note_sequence_id(self, filename, collection_name, source_type):
    """Generates a unique ID for a sequence.
    The format is:'/id/<type>/<collection name>/<hash>'.
    Args:
      filename: The string path to the source file relative to the root of the
          collection.
      collection_name: The collection from which the file comes.
      source_type: The source type as a string (e.g. "midi" or "abc").
    Returns:
      The generated sequence ID as a string.
    """
    filename_fingerprint = hashlib.sha1(filename.encode('utf-8'))
    return '/id/%s/%s/%s' % (
        source_type.lower(), collection_name, filename_fingerprint.hexdigest())