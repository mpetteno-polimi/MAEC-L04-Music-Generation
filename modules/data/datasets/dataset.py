""" TODO - Module DOC """

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import utilities
from data.converters.dataconverter import DataConverter
from data.loaders.dataloader import DataLoader
from definitions import ConfigSections, Paths
from noteseqconverter import NoteSequenceConverter


class Dataset(ABC):
    """ TODO - Class DOC """

    def __init__(self):
        self._config_file = utilities.load_configuration_section(ConfigSections.DATASETS)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def path(self) -> Path:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass


class RecordsDataset(Dataset):

    def __init__(self, train_tfrecord_paths: [Path], validation_tfrecord_paths: [Path], test_tfrecord_paths: [Path]):
        super().__init__()
        self.test_dataset = None
        self.validation_dataset = None
        self.train_dataset = None
        self.train_tfrecord_paths = train_tfrecord_paths
        self.validation_tfrecord_paths = validation_tfrecord_paths
        self.test_tfrecord_paths = test_tfrecord_paths

    def load(self) -> None:
        if self.data_loader:
            self.train_dataset = self.data_loader.load_train(self.train_tfrecord_paths)
            self.validation_dataset = self.data_loader.load_validation(self.validation_tfrecord_paths)
            self.test_dataset = self.data_loader.load_test(self.test_tfrecord_paths)

    @property
    @abstractmethod
    def data_loader(self) -> DataLoader:
        pass


class SourceDataset(Dataset):

    def convert(self) -> None:
        if self.data_converter:
            self.data_converter.convert_train(self._train_metadata)
            self.data_converter.convert_validation(self._validation_metadata)
            self.data_converter.convert_test(self._test_metadata)

    @abstractmethod
    def download(self) -> None:
        pass

    @property
    def data_converter(self) -> DataConverter:
        return NoteSequenceConverter(self.path, Paths.DATA_RECORDS_DIR, self.name)

    @property
    @abstractmethod
    def _metadata(self) -> Any:
        pass

    @property
    @abstractmethod
    def _train_metadata(self) -> Any:
        pass

    @property
    @abstractmethod
    def _validation_metadata(self) -> Any:
        pass

    @property
    @abstractmethod
    def _test_metadata(self) -> Any:
        pass
