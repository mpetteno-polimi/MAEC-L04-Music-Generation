""" TODO - Module DOC """

from pathlib import Path

from definitions import Paths
from modules.data.converters.dataconverter import DataConverter
from modules.data.converters.pianoroll import PianoRollConverter
from modules.data.datasets.dataset import TFRecordsDataset


class PianoRollDataset(TFRecordsDataset):
    """ TODO - Class DOC """

    @property
    def name(self) -> str:
        return "pianoroll"

    @property
    def path(self) -> Path:
        return Paths.DATA_PIANOROLL_RECORDS_DIR

    @property
    def version(self) -> str:
        return "v1.0.0"

    @property
    def data_converter(self) -> DataConverter:
        return PianoRollConverter(Paths.DATA_NOTESEQ_RECORDS_DIR, Paths.DATA_PIANOROLL_RECORDS_DIR, self.name)
