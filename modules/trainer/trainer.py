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

    def _get_input_tensors(self, dataset_fn, config):
        """Get input tensors from dataset."""
        batch_size = config.hparams.batch_size
        iterator = tf.data.make_one_shot_iterator(dataset_fn)
        (input_sequence, output_sequence, control_sequence, sequence_length) = iterator.get_next()
        input_sequence.set_shape(
            [batch_size, None, config.data_converter.input_depth])
        output_sequence.set_shape(
            [batch_size, None, config.data_converter.output_depth])
        if not config.data_converter.control_depth:
            control_sequence = None
        else:
            control_sequence.set_shape(
                [batch_size, None, config.data_converter.control_depth])
        sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())

        return {
            'input_sequence': input_sequence,
            'output_sequence': output_sequence,
            'control_sequence': control_sequence,
            'sequence_length': sequence_length
        }

    @abstractmethod
    def train(self):
        """ Retrieves the current settings and launches the train loop """
        pass

    def _trial_summary(self, hparams, examples_path, output_dir):
        """Writes a tensorboard text summary of the trial."""

        examples_path_summary = tf.summary.text(
            'examples_path', tf.constant(examples_path, name='examples_path'),
            collections=[])

        hparams_dict = hparams.values()

        # Create a markdown table from hparams.
        header = '| Key | Value |\n| :--- | :--- |\n'
        keys = sorted(hparams_dict.keys())
        lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
        hparams_table = header + '\n'.join(lines) + '\n'

        hparam_summary = tf.summary.text(
            'hparams', tf.constant(hparams_table, name='hparams'), collections=[])

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
            writer.add_summary(examples_path_summary.eval())
            writer.add_summary(hparam_summary.eval())
            writer.close()
