""" TODO - Module DOC """

import tensorflow as tf
from visual_midi import Plotter
from visual_midi import Preset
from pretty_midi import PrettyMIDI

from modules.data.plotter.dataplotter import DataPlotter


class MidiDataPlotter(DataPlotter):
    """ TODO - Class DOC """

    def plot(self, files_names: [str]) -> None:
        for idx, file in enumerate(files_names):
            try:
                full_file_path = self.input_path / file
                pm = PrettyMIDI(str(full_file_path))
                preset = Preset(plot_width=850)
                plotter = Plotter(preset, plot_max_length_bar=4)
                output_file_path = self.output_path / self.collection_name / file
                tf.io.gfile.makedirs(output_file_path.parent)
                plotter.save(pm, str(output_file_path.with_suffix(".html")))
            except Exception as exc:
                tf.compat.v1.logging.fatal('{} generated an exception: {}'.format(file, exc))
