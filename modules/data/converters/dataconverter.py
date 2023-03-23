""" TODO - Module DOC """

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DataConverter(ABC):
    """ TODO - Class DOC """

    def __init__(self, input_path: Path, output_path: Path, collection_name: str):
        self.input_path = input_path
        self.output_path = output_path
        self.collection_name = collection_name

    @abstractmethod
    def convert_train(self, files_to_convert: Any) -> None:
        pass

    @abstractmethod
    def convert_validation(self, files_to_convert: Any) -> None:
        pass

    @abstractmethod
    def convert_test(self, files_to_convert: Any) -> None:
        pass
