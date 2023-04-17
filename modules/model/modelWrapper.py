from modules.utilities import slerp
import magenta.models.music_vae.configs as configs
from magenta.models.music_vae import TrainedModel
from modules.trainer.VAETrainer import VAETrainer
from definitions import ConfigSections
from modules import utilities
import numpy as np
import note_seq
import logging
import time
import os
# TODO: DOC


class MaecModelAPI(object):
    """
    Simple API for launching model's operations

    Pro

    Methods:
        @load_model loads model on @self.model
        @generate
        @train
    """

    def __init__(self, model=None, dataset=None):
        self.model_config_file = utilities.load_configuration_section(ConfigSections.MODEL)
        self.train_config_file = utilities.load_configuration_section(ConfigSections.TRAINING)
        self.config_map = configs.CONFIG_MAP[self.model_config_file.get('config_map_name')]
        self.config_map.data_converter.max_tensors_per_item = None
        self.model = model
        self.dataset = dataset

    def load_trained_model(self):
        """
        mounts model on self.model
        """
        logging.info('Loading model...')
        checkpoint_dir = os.path.expanduser(self.model_config_file.get('model_checkpoint_file_dir'))
        self.model = TrainedModel(
            self.config_map,
            batch_size=min(self.model_config_file.get('batch_size'), self.model_config_file.get('num_output_files')),
            checkpoint_dir_or_path=checkpoint_dir
        )

    def generate_midi(self):
        """ Generate midi sequences """
        if self.model is None:
            logging.debug('model is', self.model, '- trying to load it...')
            self.load_trained_model()

        results = None
        if self.model_config_file.get('model_mode') == 'interpolate':
            logging.info('Interpolating...')
            input_midi_1 = os.path.expanduser(self.model_config_file.get('midi_input_path_1'))
            input_midi_2 = os.path.expanduser(self.model_config_file.get('midi_input_path_2'))
            input_1 = note_seq.midi_file_to_note_sequence(input_midi_1)
            input_2 = note_seq.midi_file_to_note_sequence(input_midi_2)

            _, mu, _ = self.model.encode([input_1, input_2])
            z = np.array([
                slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, self.model_config_file.get('num_output_files'))])
            results = self.model.decode(
                length=self.config_map.hparams.max_seq_len,
                z=z,
            )
            # temperature=FLAGS.temperature) # todo: Support temperature: conditioning????
        elif self.model_config_file.get('model_mode') == 'sample':
            logging.info('Sampling...')
            results = self.model.sample(
                n=self.model_config_file.get('num_output_files'),
                length=self.config_map.hparams.max_seq_len,
            )
            # temperature=FLAGS.temperature) # todo: Support temperature: conditioning????

        date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
        basename = os.path.join(
            self.model_config_file.get('output_directory'),
            '%s_%s_%s-*-of-%03d.mid' %
            (self.model_config_file.get('config_map_name'), self.model_config_file.get('model_mode'), date_and_time,
             self.model_config_file.get('num_output_files'))
        )
        logging.info('Outputting %d files as `%s`...', self.model_config_file.get('num_output_files'), basename)
        if results is None:
            raise Exception("model not able to process ")
        for i, ns in enumerate(results):
            note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

        logging.info('Done.')

    def train(self):
        """
        Launches training on self.model

        :param dataset:
        """

        if self.model is None:
            raise ValueError('ERROR: no model found - model: ', self.model)
        if self.dataset is None:
            raise ValueError('ERROR: no dataset available for training - dataset: ', self.dataset)

        trainer = VAETrainer(
            model=self.model,
            dataset=self.dataset,
        )

        trainer.train()

        

