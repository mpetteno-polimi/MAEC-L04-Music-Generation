""" TODO - Module DOC """

from typing import Callable

import tensorflow as tf
import note_seq
from magenta.scripts.convert_dir_to_note_sequences import generate_note_sequence_id
from note_seq import abc_parser

from modules.data.converters.dataconverter import DataConverter


class NoteSequenceConverter(DataConverter):
    """ TODO - Class DOC """

    def convert_train(self, files_to_convert: [str]) -> None:
        self._convert_to_noteseq("train", files_to_convert)

    def convert_validation(self, files_to_convert: [str]) -> None:
        self._convert_to_noteseq("validation", files_to_convert)

    def convert_test(self, files_to_convert: [str]) -> None:
        self._convert_to_noteseq("test", files_to_convert)

    def _convert_to_noteseq(self, tfrecord_file_label: str, files_to_convert: [str]) -> None:
        tfrecord_file_name = '{}-{}_noteseq.tfrecord'.format(self.collection_name, tfrecord_file_label)
        tfrecord_file_path = self.output_path / tfrecord_file_name
        if not tfrecord_file_path.exists():
            tf.io.gfile.mkdir(self.output_path)
            with tf.io.TFRecordWriter(str(tfrecord_file_path)) as writer:
                for idx, file in enumerate(files_to_convert):
                    try:
                        if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                            sequences = self._convert_file(file, note_seq.midi_file_to_sequence_proto, 'midi')
                            tf.compat.v1.logging.info('Converted MIDI file {}. {} - Progress: {}/{}'.format(
                                file, tfrecord_file_label.upper(), idx + 1, len(files_to_convert)))
                        elif file.lower().endswith('.xml') or file.lower().endswith('.mxl'):
                            sequences = self._convert_file(file, note_seq.musicxml_file_to_sequence_proto, 'musicxml')
                            tf.compat.v1.logging.info('Converted MusicXML file {}. {} - Progress: {}/{}'.format(
                                file, tfrecord_file_label.upper(), idx + 1, len(files_to_convert)))
                        elif file.lower().endswith('.abc'):
                            sequences = self._convert_abc(file)
                    except note_seq.MIDIConversionError as e:
                        tf.compat.v1.logging.warning('Could not parse MIDI file {}. It will be skipped. '
                                                     'Error was: {}'.format(file, e))
                    except note_seq.MusicXMLConversionError as e:
                        tf.compat.v1.logging.warning('Could not parse MusicXML file {}. It will be skipped. '
                                                     'Error was: {}'.format(file, e))
                    except Exception as exc:
                        tf.compat.v1.logging.fatal('{} generated an exception: {}'.format(file, exc))
                    else:
                        if sequences:
                            for sequence in sequences:
                                writer.write(sequence.SerializeToString())
        else:
            tf.compat.v1.logging.info('{} already exists. Moving forward.'.format(str(tfrecord_file_path)))

    def _convert_file(self, file: str, converter_fn: Callable, source_type: str) -> [note_seq.NoteSequence]:
        """Converts a file to a sequence proto.

        Args:
          file: Path of the file to convert
          converter_fn: The converter function
          source_type: The type of the file to convert (e.g. midi, musicxml, etc...)

        Returns:
          Either a list of NoteSequence proto or None if the file could not be records.
        """

        full_file_path = str(self.input_path / file)
        sequence = converter_fn(full_file_path)
        sequence.collection_name = self.collection_name
        sequence.filename = file
        sequence.id = generate_note_sequence_id(sequence.filename, sequence.collection_name, source_type)
        return [sequence]

    def _convert_abc(self, file: str) -> [note_seq.NoteSequence]:
        """Converts an abc file to a sequence proto.

        Args:
          file: Path of the file to convert

        Returns:
          Either a NoteSequence proto or None if the file could not be converted.
        """
        try:
            tunebook = tf.io.gfile.GFile(file, 'rb').read()
            tunes, exceptions = abc_parser.parse_abc_tunebook(tunebook)
        except abc_parser.ABCParseError as e:
            tf.compat.v1.logging.warning('Could not parse ABC file %s. It will be skipped. Error was: %s', file, e)
            return None

        for exception in exceptions:
            tf.compat.v1.logging.warning('Could not parse tune in ABC file %s. It will be skipped. Error was: %s', file,
                                         exception)

        sequences = []
        for idx, tune in tunes.items():
            tune.collection_name = self.collection_name
            tune.filename = file
            tune.id = generate_note_sequence_id('{}_{}'.format(tune.filename, idx), tune.collection_name, 'abc')
            sequences.append(tune)
            tf.compat.v1.logging.info('Converted ABC file %s.', file)
        return sequences
