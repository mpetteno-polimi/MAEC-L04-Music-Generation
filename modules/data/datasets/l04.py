""" TODO - Module DOC """

from pathlib import Path

from data.loaders.dataloader import DataLoader
from data.loaders.noteseqloader import NoteSequenceLoader
from dataset import RecordsDataset
from definitions import Paths


class L04Dataset(RecordsDataset):
    """ TODO - Class DOC """

    @property
    def data_loader(self) -> DataLoader:
        return NoteSequenceLoader(Paths.DATA_RECORDS_DIR)

    @property
    def name(self) -> str:
        return 'l04-' + self.version

    @property
    def path(self) -> Path:
        return Paths.DATA_RECORDS_DIR / self.name

    @property
    def version(self) -> str:
        return 'v1.0.0'
