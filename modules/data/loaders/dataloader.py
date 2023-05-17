""" TODO - Module DOC """

from abc import ABC, abstractmethod
from typing import AnyStr, Any

from definitions import ConfigSections
from modules.utilities import config


class DataLoader(ABC):
    """ TODO - Class DOC """

    def __init__(self):
        self._training_config = config.load_configuration_section(ConfigSections.TRAINING)

    @abstractmethod
    def load(self, filenames: AnyStr) -> Any:
        pass
