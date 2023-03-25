""" Module that contains all necessary definitions.

Definitions are separated into classes in order to make the module more readable and maintainable.

"""
import os
from pathlib import Path


class ConfigSections:
    """ Constants that indicate the sections present in the main configuration file. """

    DATASETS = 'Datasets'


class Paths:
    """ Constants that represent useful file system paths.

     This class use pathlib library to declare the paths.

     """

    ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_DIR = ROOT_DIR / 'config'
    MAIN_CONFIG_FILE = CONFIG_DIR / 'config.ini'
    RESOURCES_DIR = ROOT_DIR / 'resources'
    DATA_DIR = RESOURCES_DIR / 'data'
    DATA_SOURCES_DIR = DATA_DIR / 'sources'
    DATA_RECORDS_DIR = DATA_DIR / 'records'
    DATA_NOTESEQ_RECORDS_DIR = DATA_RECORDS_DIR / 'noteseq'
    DATA_PIANOROLL_RECORDS_DIR = DATA_RECORDS_DIR / 'pianoroll'
