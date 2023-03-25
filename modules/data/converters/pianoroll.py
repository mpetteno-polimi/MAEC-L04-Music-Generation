""" TODO - Module DOC """

from pathlib import Path

from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_model
from magenta.pipelines import pipeline

from modules.data.converters.dataconverter import DataConverter
from modules.data.pipelines import pipelines


class PianoRollConverter(DataConverter):
    """ TODO - Class DOC """

    def convert_train(self, files_to_convert: [Path]) -> None:
        self._convert_to_piano_roll(files_to_convert)

    def convert_validation(self, files_to_convert: [Path]) -> None:
        self._convert_to_piano_roll(files_to_convert)

    def convert_test(self, files_to_convert: [Path]) -> None:
        self._convert_to_piano_roll(files_to_convert)

    def _convert_to_piano_roll(self, files_to_convert: [Path]) -> None:
        for file_to_convert in files_to_convert:
            in_tfrecord_file_path = self.input_path / file_to_convert
            out_trecord_file_name = in_tfrecord_file_path.stem.replace("_noteseq", "")
            out_tfrecord_file = self.output_path / out_trecord_file_name

            if not out_tfrecord_file.exists():
                pipeline_instance = pipelines.pianoroll_pipeline(
                    config=pianoroll_rnn_nade_model.default_configs['rnn-nade'],  # TODO - Change
                    min_steps=80,  # 5 measures
                    max_steps=2048,
                    out_filename=self.collection_name
                )

                pipeline.run_pipeline_serial(pipeline_instance,
                                             pipeline.tf_record_iterator(str(in_tfrecord_file_path),
                                                                         pipeline_instance.input_type),
                                             str(self.output_path), out_trecord_file_name)
