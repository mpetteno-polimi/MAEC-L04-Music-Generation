""" TODO - Module DOC """

from abc import ABC, abstractmethod
from pathlib import Path


class DataPlotter(ABC):
    """ TODO - Class DOC """

    def __init__(self, input_path: Path, output_path: Path, collection_name: str):
        self.input_path = input_path
        self.output_path = output_path
        self.collection_name = collection_name

    @abstractmethod
    def plot(self, files_names: [str]) -> None:
        pass
