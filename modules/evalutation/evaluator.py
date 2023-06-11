# TODO - DOC
# TODO: delete print

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
        self._min_velocity_threshold = int(test_config.get('min_velocity_threshold'))  # 0..127 scale

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
                # NO SUSTAINED NOTES VERSION
                # each note is a new note right now
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
                #                 end=end_time_sec
                #             )
                #             piano_instr.notes.append(note)

                file_num = int(pr_idx + batch_idx)
                if file_num >= max_outputs:
                    return
                # pianoroll note first
                pianoroll = tf.transpose(pianoroll, perm=[1, 0])
                # for each pianoroll note scroll all time steps and add available ones
                for note_idx, note_time_steps in enumerate(pianoroll):
                    if note_idx % 2 == 0:
                        time_step_idx = 0
                        while time_step_idx < tf.size(note_time_steps):
                            note_vel = note_time_steps[time_step_idx]
                            # for time_step_idx, note_vel in enumerate(note_time_steps):
                            velocity = int(math.ceil(K.eval(note_vel) * 127))
                            # print(note_idx, time_step_idx, velocity)
                            time_step_idx += 1

                            if velocity >= self._min_velocity_threshold:
                                start_time_sec = time_step_idx * tatum_seconds
                                end_time_sec = start_time_sec + tatum_seconds
                                while time_step_idx < tf.size(note_time_steps) \
                                    and round(K.eval(pianoroll[note_idx + 1][time_step_idx])) == 1:
                                    time_step_idx += 1
                                    end_time_sec += tatum_seconds

                                note_midi_number = int(note_idx / 2)
                                assert 0 <= note_midi_number <= 127
                                assert 0 <= velocity <= 127
                                assert start_time_sec < end_time_sec
                                note = pretty_midi.Note(
                                    velocity=velocity,
                                    pitch=note_midi_number,
                                    start=start_time_sec,
                                    end=end_time_sec
                                )
                                piano_instr.notes.append(note)
                                # print(note)
                                # print(time_step_idx, note_idx)

                midi_file.instruments.append(piano_instr)

                print('./../resources/eval/midi_out/' + str(file_num) + '.mid')
                midi_file.write('./../resources/eval/midi_out/' + str(file_num) + '.mid')

    def evaluate(self, inputs, use_pianoroll_input, max_outputs=None):
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

        assert inputs is not None
        z_sample_batches = fetch_z_sample_batches()
        # Start inference
        print('Predicting... ')
        results = []
        for idx, input_batch in enumerate(inputs):
            if idx >= self._n_test_files:
                break
            z = tf.convert_to_tensor(z_sample_batches[idx * 2:(idx * 2) + 2, :])

            if use_pianoroll_input:
                sampling_inputs = input_batch[0]
            else:
                sampling_inputs = input_batch
            results.append(self._model.sample(
                sampling_inputs,
                use_pianoroll_input=use_pianoroll_input,
                z_sample=z,
            ))

        print('Done')
        # print('results:', results)
        self.create_midi_files(pianoroll_batches=results, max_outputs=max_outputs)
