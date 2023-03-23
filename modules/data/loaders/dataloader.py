""" TODO - Module DOC """

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DataLoader(ABC):
    """ TODO - Class DOC """

    def __init__(self, input_path: Path):
        self.input_path = input_path

    @abstractmethod
    def load_train(self, train_metadata: Any = None) -> None:
        pass

    @abstractmethod
    def load_validation(self, validation_metadata: Any = None) -> None:
        pass

    @abstractmethod
    def load_test(self, test_metadata: Any = None) -> None:
        pass
