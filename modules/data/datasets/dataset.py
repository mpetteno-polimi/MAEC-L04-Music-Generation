""" TODO - Module DOC """

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from definitions import ConfigSections, Paths
from modules import utilities
from modules.data.converters.dataconverter import DataConverter
from modules.data.converters.noteseq import NoteSequenceConverter
from modules.data.plotter.dataplotter import DataPlotter
from modules.data.plotter.midi import MidiDataPlotter


class Dataset(ABC):
    """ TODO - Class DOC """

    def __init__(self):
        self._config_file = utilities.load_configuration_section(ConfigSections.DATASETS)
        self.dataset = None
        self._plot_data = self._config_file.get('plot_data')

    def convert(self) -> None:
        if self.data_converter:
            self.data_converter.convert_train(self._train_data)
            self.data_converter.convert_validation(self._validation_data)
            self.data_converter.convert_test(self._test_data)

    def plot(self) -> None:
        if self.data_plotter and self._plot_data:
            self.data_plotter.plot(self._train_data)
            self.data_plotter.plot(self._validation_data)
            self.data_plotter.plot(self._test_data)

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
    def metadata(self) -> Any:
        pass

    @property
    @abstractmethod
    def _train_data(self) -> [str]:
        pass

    @property
    @abstractmethod
    def _validation_data(self) -> [str]:
        pass

    @property
    @abstractmethod
    def _test_data(self) -> [str]:
        pass

    @property
    @abstractmethod
    def data_converter(self) -> DataConverter:
        pass

    @property
    @abstractmethod
    def data_plotter(self) -> DataPlotter:
        pass


class SourceDataset(Dataset, ABC):

    def __init__(self):
        super().__init__()
        self.download()

    @abstractmethod
    def download(self) -> None:
        pass

    @property
    def data_converter(self) -> DataConverter:
        return NoteSequenceConverter(self.path, Paths.DATA_NOTESEQ_RECORDS_DIR, self.name)


class MidiDataset(SourceDataset, ABC):

    @property
    def data_plotter(self) -> DataPlotter:
        return MidiDataPlotter(self.path, Paths.DATA_PLOTS_DIR, self.name)
