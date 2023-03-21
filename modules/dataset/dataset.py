""" TODO - Module DOC """

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import tensorflow as tf
import pretty_midi as pm

import utilities
from definitions import ConfigSections


class Dataset(ABC):
    """ TODO - Class DOC """

    def __init__(self):
        self._config_file = utilities.load_configuration_section(ConfigSections.DATASETS)
        self._download()
        self.train_dataset = self._process_dataset_metadata(self._train_dataset_metadata)
        self.validation_dataset = self._process_dataset_metadata(self._validation_dataset_metadata)
        self.test_dataset = self._process_dataset_metadata(self._test_dataset_metadata)

    @abstractmethod
    def _download(self) -> None:
        pass

    @abstractmethod
    def _process_dataset_metadata(self, dataset_metadata: Any) -> tf.data.Dataset:
        pass

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

    @property
    @abstractmethod
    def _train_dataset_metadata(self) -> Any:
        pass

    @property
    @abstractmethod
    def _validation_dataset_metadata(self) -> Any:
        pass

    @property
    @abstractmethod
    def _test_dataset_metadata(self) -> Any:
        pass


class MidiDataset(Dataset, ABC):
    """ TODO - Class DOC """

    def _process_dataset_metadata(self, midi_files: [str]) -> tf.data.Dataset:
        for midi_file in midi_files:
            midi_file_path = str(self.path/midi_file)
            midi = pm.PrettyMIDI(midi_file_path)
            deb = midi.instruments[0].get_piano_roll()
            return
