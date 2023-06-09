""" Module that contains all necessary definitions.

Definitions are separated into classes in order to make the module more readable and maintainable.

"""
import os
from pathlib import Path


class ConfigSections:
    """ Constants that indicate the sections present in the main configuration file. """

    REPRESENTATION = 'Representation'
    DATASETS = 'Datasets'
    MODEL = 'Model'
    TRAINING = 'Training'
    TEST = 'Test'
    CHECKPOINTS = 'Checkpoints'
    EARLY_STOPPING = 'EarlyStopping'
    BACKUP = 'Backup'
    TENSORBOARD = 'Tensorboard'


class Paths:
    """ Constants that represent useful file system paths.

     This class use pathlib library to declare the paths.

     """

    ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_DIR = ROOT_DIR / 'config'
    MAIN_CONFIG_FILE = CONFIG_DIR / 'config.ini'
    RESOURCES_DIR = ROOT_DIR / 'resources'
    DATA_DIR = RESOURCES_DIR / 'datasets'
    DATA_SOURCES_DIR = DATA_DIR / 'sources'
    DATA_RECORDS_DIR = DATA_DIR / 'records'
    DATA_PLOTS_DIR = DATA_DIR / 'plots'
    DATA_NOTESEQ_RECORDS_DIR = DATA_RECORDS_DIR / 'noteseq'
    TRAIN_DIR = RESOURCES_DIR / "train"
    TRAIN_CHECK_DIR = TRAIN_DIR / "checkpoints"
    TRAIN_BACKUP_DIR = TRAIN_DIR / "backup"
    LOG_DIR = RESOURCES_DIR / "logs"
