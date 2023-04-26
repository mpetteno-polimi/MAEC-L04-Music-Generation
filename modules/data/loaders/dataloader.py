""" TODO - Module DOC """

from abc import ABC, abstractmethod
from typing import AnyStr, Any

from definitions import ConfigSections
from modules import utilities


class DataLoader(ABC):
    """ TODO - Class DOC """

    def __init__(self):
        self._training_config = utilities.load_configuration_section(ConfigSections.TRAINING)

    @abstractmethod
    def load(self, filenames: AnyStr) -> Any:
        pass
