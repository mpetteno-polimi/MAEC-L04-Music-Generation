# TODO - Doc

import os
import glob

import pretty_midi

import numpy as np
import tensorflow as tf

from definitions import ConfigSections
from modules.utilities import config as config_file
from modules.utilities import complexity_measures

logging = tf.compat.v1.logging
script_config = config_file.load_configuration_section(ConfigSections.LATENT_SPACE_SAMPLING)
complexities_methods = ['toussaint', 'note density', 'pitch range', 'contour']


def run(samples_folder_path, metrics=None, save=True):
    if metrics is None:
        metrics = complexities_methods

    num_bars = script_config.get("num_bars")
    file_name_pattern = samples_folder_path + "/**/*.mid"
    empty_midi_file_count = 0
    samples_complexities = []
    for idx, file_path in enumerate(glob.glob(file_name_pattern, recursive=True)):
        # Load MIDI file
        midi_file = pretty_midi.PrettyMIDI(file_path)
        try:
            midi_file = complexity_measures.sanitize(midi_file, num_bars)
        except AssertionError as e:
            logging.warning("Skipping MIDI file %s. Cause is: %s" % (file_path, e.args))
            empty_midi_file_count = empty_midi_file_count + 1
            continue

        # Complexities computation
        sample_complexities = []
        for metric in metrics:
            if metric == 'toussaint':
                toussaint = complexity_measures.toussaint(midi_file, num_bars, binary=True)
                sample_complexities.append(toussaint)
            elif metric == 'note density':
                note_density = complexity_measures.note_density(midi_file, num_bars, binary=True)
                sample_complexities.append(note_density)
            elif metric == 'pitch range':
                pitch_range = complexity_measures.pitch_range(midi_file)
                sample_complexities.append(pitch_range)
            elif metric == 'contour':
                contour = complexity_measures.contour(midi_file)
                sample_complexities.append(contour)
            else:
                continue
        samples_complexities.append(np.asarray(sample_complexities))

        # Save current sample complexities file
        if save:
            folder_path = os.path.dirname(file_path)
            folder_name = os.path.basename(folder_path)
            sample_num = folder_name.split(sep="_")[-1]
            sample_complexities_file_name = 'sample_complexities_%s.npy' % sample_num
            sample_complexities_file_path = os.path.join(folder_path, sample_complexities_file_name)
            np.save(sample_complexities_file_path, sample_complexities)

    logging.info("Found %d empty MIDI files." % empty_midi_file_count)

    return np.asarray(samples_complexities)


def main(_):
    logging.set_verbosity(script_config.get("log"))
    output_dir = os.path.expanduser(script_config.get("output_dir"))
    model_config = script_config.get("model_config")
    rand_seed = script_config.get("rand_seed")
    run_folder_name = 'config_%s_seed_%d' % (model_config, rand_seed)
    samples_folder_name = 'samples'
    samples_folder_path = os.path.join(output_dir, run_folder_name, samples_folder_name)
    run(samples_folder_path)


def console_entry_point():
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.app.run(main)


if __name__ == '__main__':
    console_entry_point()
