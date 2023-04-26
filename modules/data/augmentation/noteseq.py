from typing import Any

import tensorflow as tf
import note_seq
from note_seq import NoteSequence, sequences_lib

from modules.data.augmentation.dataaugmenter import DataAugmenter


class NoteSequenceAugmenter(DataAugmenter):

    def __init__(self, transpose_range=None, stretch_range=None):
        self._transpose_range = transpose_range
        self._stretch_range = stretch_range

    def augment(self, note_sequence: NoteSequence) -> Any:
        """Python implementation that augments the NoteSequence.
        Args:
          note_sequence: A NoteSequence proto to be augmented.
        Returns:
          The randomly augmented NoteSequence.
        """
        transpose_min, transpose_max = self._transpose_range if self._transpose_range else (0, 0)
        stretch_min, stretch_max = self._stretch_range if self._stretch_range else (1.0, 1.0)

        return sequences_lib.augment_note_sequence(
            note_sequence,
            stretch_min,
            stretch_max,
            transpose_min,
            transpose_max,
            delete_out_of_range_notes=True)

    def tf_augment(self, note_sequence_scalar):
        """TF op that augments the NoteSequence."""

        def _augment_str(note_sequence_str):
            note_sequence = note_seq.NoteSequence.FromString(note_sequence_str.numpy())
            augmented_ns = self.augment(note_sequence)
            return [augmented_ns.SerializeToString()]

        augmented_note_sequence_scalar = tf.py_function(
            _augment_str,
            inp=[note_sequence_scalar],
            Tout=tf.string,
            name='augment')
        augmented_note_sequence_scalar.set_shape(())
        return augmented_note_sequence_scalar
