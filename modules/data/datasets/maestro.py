""" TODO - Module DOC """

from pathlib import Path
from typing import Any

import tensorflow as tf
import pandas as pd

from dataset import SourceDataset
from definitions import Paths


class MaestroDataset(SourceDataset):
    """ TODO - Class DOC """

    def download(self) -> None:
        maestro_zip_name = self.name + ('-midi' if self._config_file.get('maestro_midi_only') else '') + '.zip'
        maestro_url = '/'.join([self._config_file.get('maestro_url'), self.version, maestro_zip_name])
        if not self.path.exists():
            tf.keras.utils.get_file(
                fname=maestro_zip_name,
                origin=maestro_url,
                extract=True,
                cache_dir=Paths.RESOURCES_DIR,
                cache_subdir=Paths.DATA_SOURCES_DIR
            )

    @property
    def name(self) -> str:
        return 'maestro-' + self.version

    @property
    def path(self) -> Path:
        return Paths.DATA_SOURCES_DIR / self.name

    @property
    def version(self) -> str:
        return 'v' + self._config_file.get('maestro_version')

    @property
    def _metadata(self) -> Any:
        return pd.read_csv(self.path / (self.name + '.csv'))

    @property
    def _train_metadata(self) -> [str]:
        train_metadata = self._metadata[self._metadata.split == 'train']
        return train_metadata.midi_filename

    @property
    def _validation_metadata(self) -> [str]:
        validation_metadata = self._metadata[self._metadata.split == 'validation']
        return validation_metadata.midi_filename

    @property
    def _test_metadata(self) -> [str]:
        test_metadata = self._metadata[self._metadata.split == 'test']
        return test_metadata.midi_filename
