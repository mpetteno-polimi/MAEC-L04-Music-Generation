""" TODO - Module DOC """

import tensorflow as tf
import pandas as pd
from abc import ABC
import utilities
from definitions import ConfigSections, Paths


class Dataset(ABC):

    def __init__(self):

        self._train_dataset = None
        self._validation_dataset = None
        self._test_dataset = None

        # Load configuration file
        config_file = utilities.load_configuration_section(ConfigSections.DATASETS)
        self._config_file = config_file


class MaestroDataset(Dataset):

    def __init__(self):
        super().__init__()
        self._init_paths()
        self.download()
        self.make()

    def _init_paths(self):
        self._version = 'v' + self._config_file.get('maestro_version')
        self._name = 'maestro-' + self._version
        self._path = Paths.DATASETS_DIR / self._name
        self._metadata = self._path / (self._name + '.csv')

    def download(self):
        midi_only = '-midi' if self._config_file.get('maestro_midi_only') else ''
        maestro_zip_name = self._name + midi_only + '.zip'
        maestro_url = '/'.join([self._config_file.get('maestro_url'), self._version, maestro_zip_name])
        if not self._path.exists():
            tf.keras.utils.get_file(
                fname=maestro_zip_name,
                origin=maestro_url,
                extract=True,
                cache_dir=Paths.RESOURCES_DIR,
                cache_subdir=Paths.DATASETS_DIR,
            )

    def make(self):
        dataset = pd.read_csv(self._metadata)
        self._train_dataset = dataset[dataset.split == 'train']
        self._validation_dataset = dataset[dataset.split == 'validation']
        self._test_dataset = dataset[dataset.split == 'test']


if __name__ == '__main__':
    maestroData = MaestroDataset()
