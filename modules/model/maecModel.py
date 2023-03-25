from magenta.models.music_vae.music_vae_generate import _slerp
import magenta.models.music_vae.configs as config
from magenta.models.music_vae import TrainedModel
from definitions import ConfigSections
from modules import utilities
import numpy as np
import note_seq
import logging
import time
import os
# TODO: DOC


class MaecModel(object):

    def __init__(self):
        self.config_file = utilities.load_configuration_section(ConfigSections.MODEL)
        self.config_map = config.CONFIG_MAP[self.config_file.get('config_map_name')]
        self.config_map.data_converter.max_tensors_per_item = None
        self.model = None

    def load_model(self):
        """
        """
        logging.info('Loading model...')
        checkpoint_dir_or_path = os.path.expanduser(self.config_file.get('model_checkpoint_file_path'))
        self.model = TrainedModel(
            self.config_map,
            batch_size=min(self.config_file.get('batch_size'), self.config_file.get('num_output_files')),
            checkpoint_dir_or_path=checkpoint_dir_or_path
        )

    def generate(self):
        """ Generate midi sequences """
        if self.model is None:
            logging.debug('model is', self.model, '- trying to load it...')
            self.load_model()

        result = None
        if self.config_file.get('model_mode') == 'interpolate':
            logging.info('Interpolating...')
            input_midi_1 = os.path.expanduser(self.config_file.get('midi_input_path_1'))
            input_midi_2 = os.path.expanduser(self.config_file.get('midi_input_path_2'))
            input_1 = note_seq.midi_file_to_note_sequence(input_midi_1)
            input_2 = note_seq.midi_file_to_note_sequence(input_midi_2)

            _, mu, _ = self.model.encode([input_1, input_2])
            z = np.array([
                _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, self.config_file.get('num_output_files'))])
            results = self.model.decode(length=self.config_map.hparams.max_seq_len, z=z, )
            # temperature=FLAGS.temperature) # todo: Support temperature: conditioning????
        elif self.config_file.get('model_mode') == 'sample':
            logging.info('Sampling...')
            results = self.model.sample(
                n=self.config_file.get('num_output_files'),
                length=self.config_map.hparams.max_seq_len, )
            # temperature=FLAGS.temperature) # todo: Support temperature: conditioning????

        date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
        basename = os.path.join(
            self.config_file.get('output_directory'),
            '%s_%s_%s-*-of-%03d.mid' %
            (self.config_file.get('config_map_name'), self.config_file.get('model_mode'), date_and_time,
             self.config_file.get('num_output_files'))
        )
        logging.info('Outputting %d files as `%s`...', self.config_file.get('num_output_files'), basename)
        if results is None:
            raise Exception("model not able to process ")
        for i, ns in enumerate(results):
            note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

        logging.info('Done.')