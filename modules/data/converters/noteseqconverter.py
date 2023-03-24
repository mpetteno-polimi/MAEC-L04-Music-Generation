""" TODO - Module DOC """

import tensorflow as tf
import note_seq as ns
from magenta.scripts.convert_dir_to_note_sequences import generate_note_sequence_id

from modules.data.converters.dataconverter import DataConverter


class NoteSequenceConverter(DataConverter):
    """ TODO - Class DOC """

    def convert_train(self, files_to_convert: [str]) -> None:
        self._convert("train", files_to_convert)

    def convert_validation(self, files_to_convert: [str]) -> None:
        self._convert("validation", files_to_convert)

    def convert_test(self, files_to_convert: [str]) -> None:
        self._convert("test", files_to_convert)

    def _convert(self, tfrecord_file_label: str, files_to_convert: [str]) -> None:
        tfrecord_file_name = '{}-{}.tfrecord'.format(self.collection_name, tfrecord_file_label)
        tfrecord_file_path = self.output_path / tfrecord_file_name
        if not tfrecord_file_path.exists():
            tf.io.gfile.mkdir(self.output_path)
            with tf.io.TFRecordWriter(str(tfrecord_file_path)) as writer:
                for idx, file in enumerate(files_to_convert):
                    try:
                        sequence = self._convert_midi_file(file)
                        tf.compat.v1.logging.info('Converted MIDI file {}. {} - Progress: {}/{}'.format(
                            file, tfrecord_file_label.upper(), idx + 1, len(files_to_convert)))
                    except ns.MIDIConversionError as e:
                        tf.compat.v1.logging.warning('Could not parse MIDI file {}. It will be skipped. '
                                                     'Error was: {}'.format(file, e))
                    except Exception as exc:
                        tf.compat.v1.logging.fatal('{} generated an exception: {}'.format(file, exc))
                    else:
                        writer.write(sequence.SerializeToString())
        else:
            tf.compat.v1.logging.info('{} already exists. Moving forward.'.format(str(tfrecord_file_path)))

    def _convert_midi_file(self, midi_file: str):
        """Converts a midi file to a sequence proto.

        Args:
          midi_file: Path of the midi file to convert

        Returns:
          Either a NoteSequence proto or None if the file could not be records.
        """

        full_file_path = str(self.input_path / midi_file)
        sequence = ns.midi_file_to_sequence_proto(full_file_path)
        sequence.collection_name = self.collection_name
        sequence.filename = midi_file
        sequence.id = generate_note_sequence_id(sequence.filename, sequence.collection_name, 'midi')
        return sequence
