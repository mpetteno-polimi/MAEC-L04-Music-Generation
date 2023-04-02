from abc import ABC, abstractmethod
import magenta.models.music_vae.configs as configs
import tensorflow._api.v2.compat.v1 as tf
from definitions import ConfigSections
from modules import utilities


# TODO: Module DOC


class Trainer(ABC):
    """
    Abstract base class which will serve as a NN trainer
    """

    # TODO: Class DOC

    def __init__(self, model, dataset):
        """
        Trainer class constructor

        :param model:
        """
        self.model = model
        self.dataset = dataset
        self._train_config = utilities.load_configuration_section(ConfigSections.TRAINING)
        self._model_config = utilities.load_configuration_section(ConfigSections.MODEL)
        self.config_map = configs.CONFIG_MAP[self._model_config.get('config_map_name')]


    @abstractmethod
    def train(self):
        """ Retrieves the current settings and launches the train loop """
        pass

    @abstractmethod
    def _train_loop(self, train_dir, config, checkpoints_to_keep=5, keep_checkpoint_every_n_hours=1,
                    num_steps=None, master='', num_sync_workers=0, num_ps_tasks=0, task=0):
        """ Train loop. """

        pass


