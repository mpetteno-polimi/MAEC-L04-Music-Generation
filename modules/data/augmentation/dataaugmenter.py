""" TODO - Module DOC """

from abc import ABC, abstractmethod
from typing import Any


class DataAugmenter(ABC):
    """ TODO - Class DOC """

    @abstractmethod
    def augment(self, data: Any) -> Any:
        pass
