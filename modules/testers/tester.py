from abc import ABC, abstractmethod
import magenta.models.music_vae.configs as configs
import tensorflow._api.v2.compat.v1 as tf
from definitions import ConfigSections
from modules import utilities


# TODO: Module DOC


class Tester(ABC):
    """
    Abstract base class which will serve as a NN tester
    """

    # TODO: Class DOC

    def __init__(self, model, dataset):
        """
        Trainer class constructor

        :param model:
        """
        self.model = model
        self.dataset = dataset
        self._model_config = utilities.load_configuration_section(ConfigSections.MODEL)
        self._train_config = utilities.load_configuration_section(ConfigSections.TRAINING)
        self._test_config = utilities.load_configuration_section(ConfigSections.TEST)
        self.config_map = configs.CONFIG_MAP[self._model_config.get('config_map_name')]

    @abstractmethod
    def test(self):
        """ Retrieves the current settings and launches the test loop """
        pass

    @abstractmethod
    def _test_loop(self, train_dir, eval_dir, config, num_batches, master=''):

        """Evaluate the model repeatedly."""

        pass
