# TODO - DOC
# TODO: delete print
# todo: each note is a new note right now [fixed, needs to be tested further]

from keras import backend as K
import tensorflow as tf
import numpy as np
import pretty_midi
import math
import os

from definitions import ConfigSections, Paths
from modules.utilities import config


class Evaluator(object):

    def __init__(self, model):
        self._model = model

        representation_config = config.load_configuration_section(ConfigSections.REPRESENTATION)
        test_config = config.load_configuration_section(ConfigSections.TEST)
        self._n_test_files = int(test_config.get('n_tests'))
        self._test_batch_size = int(test_config.get('test_batch_size'))
        self._slice_bars = int(representation_config.get('slice_bars'))
        self._input_steps_len = int(representation_config.get('num_bars')) * self._slice_bars
        self._input_feature_len = 2 * (int(representation_config.get('piano_max_midi_pitch')) - int(
            representation_config.get('piano_min_midi_pitch')) + 1)
        self._bpm = int(test_config.get('bpm_out'))
        self._z_samples_file_path = test_config.get('z_samples_file_path')

    def create_midi_files(self, pianoroll_batches, max_outputs=None):
        # TODO : Remove commented lines when cleaning code
        # rand_t = [K.random_uniform_variable(shape=(2, 256, 176), low=0.0, high=1.0, dtype="float32"),
        #           K.random_uniform_variable(shape=(2, 256, 176), low=0.0, high=1.0, dtype="float32")]
        #
        # for idx, elem in enumerate(rand_t):
        #     rand_t[idx] = tf.where(elem > 0.99, 1, 0)
        #
        # print(len(rand_t), end=' x ')
        # print(rand_t[0].shape)

        tatum_seconds = (60.0 / self._bpm) / self._slice_bars
        # per ogni elem della lista
        for batch_idx, batch in enumerate(pianoroll_batches):
            for pr_idx, pianoroll in enumerate(batch):
                # per ogni elem della batch
                midi_file = pretty_midi.PrettyMIDI(initial_tempo=self._bpm)
                piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
                piano_instr = pretty_midi.Instrument(program=piano_program)

                # per ogni elem della serie temporale
                # for time_idx, time_step in enumerate(pianoroll):
                #     for note_idx, note in enumerate(time_step):
                #         if note_idx % 2 == 0 and K.eval(note) != 0.0:
                #             start_time_sec = time_idx * tatum_seconds
                #             end_time_sec = start_time_sec + tatum_seconds
                #             note_midi_number = int(note_idx / 2)
                #             velocity = int(math.ceil(K.eval(note) * 127))
                #             assert 0 <= note_midi_number <= 127
                #             assert 0 <= math.ceil(K.eval(note) * 127) <= 127
                #             assert start_time_sec < end_time_sec
                #             note = pretty_midi.Note(
                #                 velocity=velocity,
                #                 pitch=note_midi_number,
                #                 start=start_time_sec,
                #                 end=end_time_sec  # todo: each note is a new note right now
                #             )
                #             piano_instr.notes.append(note)
                # for each note add it each time it is present in the temporal series
                for note_idx, note in enumerate(pianoroll[-1]):
                    if note_idx % 2 == 0 and K.eval(note) != 0.0:
                        time_step_idx = 0
                        while time_step_idx < pianoroll.shape[0]:  # todo: clean (refactor using iterator)
                            start_time_sec = time_step_idx * tatum_seconds
                            end_time_sec = start_time_sec + tatum_seconds
                            # increase length if sustained note and skip replay
                            note_time_steps_len = int(1)
                            for time_step in range(time_step_idx, pianoroll.shape[0]):
                                if K.eval(pianoroll[time_step][note_idx+1]):
                                    end_time_sec += tatum_seconds
                                    note_time_steps_len += 1
                            time_step_idx += note_time_steps_len

                            note_midi_number = int(note_idx / 2)
                            velocity = int(math.ceil(K.eval(note) * 127))
                            assert 0 <= note_midi_number <= 127
                            assert 0 <= math.ceil(K.eval(note) * 127) <= 127
                            assert start_time_sec < end_time_sec
                            note = pretty_midi.Note(
                                velocity=velocity,
                                pitch=note_midi_number,
                                start=start_time_sec,
                                end=end_time_sec  # todo: each note is a new note right now
                            )
                            piano_instr.notes.append(note)
                            time_step_idx += 1


                midi_file.instruments.append(piano_instr)
                file_num = int(pr_idx+batch_idx)
                if file_num >= max_outputs:
                    return
                print('./../resources/eval/midi_out/' + str(file_num) + '.mid')
                midi_file.write('./../resources/eval/midi_out/' + str(file_num) + '.mid')

    def evaluate(self, test_dataset, max_outputs=None):
        """
        The number of created midi files is max(max_outputs, batch_size, z_samples)
        """
        def fetch_z_sample_batches():
            # load z from file if available
            z_samples = None
            if self._z_samples_file_path is not None and self._z_samples_file_path != '' and self._z_samples_file_path != 'None':
                z_path = os.path.abspath(self._z_samples_file_path)

                assert os.path.isfile(z_path)
                print('Sampling from file...', z_path)
                z_samples = np.load(z_path, allow_pickle=True, fix_imports=True, encoding='latin1')
            else:
                print('WARNING: Random inference sampling')
            return z_samples

        z_sample_batches = fetch_z_sample_batches()
        # Start inference
        print('Predicting... ')
        results = []
        for idx, test_batch in enumerate(test_dataset):
            if idx >= self._n_test_files:
                break
            z = tf.convert_to_tensor(z_sample_batches[idx * 2:(idx * 2) + 2, :])
            results.append(self._model.sample(
                test_batch[0],  # test_batch is a list of 2 identical 'data sources' for testing
                pianoroll_format=True,
                z_sample=z
            ))

        print('Done')
        # print('results:', results)
        self.create_midi_files(pianoroll_batches=results, max_outputs=max_outputs)

