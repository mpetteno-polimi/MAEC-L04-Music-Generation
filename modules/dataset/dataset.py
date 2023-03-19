""" TODO - Module DOC """

import tensorflow as tf
from abc import ABC
import utilities
from definitions import ConfigSections, Paths


class Dataset(ABC):

    def __init__(self):

        # Load configuration file
        config_file = utilities.load_configuration_section(ConfigSections.DATASETS)
        self._config_file = config_file


class MaestroDataset(Dataset):

    def __init__(self):
        super().__init__()
        self._download()

    def _download(self):
        data_dir = Paths.DATASETS_DIR
        maestro_version = 'v' + self._config_file.get('maestro_version')
        maestro_dirname = 'maestro-' + maestro_version
        midi_only = '-midi' if self._config_file.get('maestro_midi_only') else ''
        maestro_zip_name = maestro_dirname + midi_only + '.zip'
        maestro_dir = data_dir / maestro_dirname
        maestro_url = '/'.join([self._config_file.get('maestro_url'), maestro_version, maestro_zip_name])
        if not maestro_dir.exists():
            tf.keras.utils.get_file(
                fname=maestro_zip_name,
                origin=maestro_url,
                extract=True,
                cache_dir=Paths.RESOURCES_DIR,
                cache_subdir=data_dir,
            )


if __name__ == '__main__':
    maestroData = MaestroDataset()
