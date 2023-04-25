""" TODO - Module DOC """

import functools

import tensorflow as tf
import note_seq
import numpy as np
from magenta.pipelines import statistics
from magenta.models.music_vae import data as music_vae_data
from note_seq import sequences_lib

from modules.data.representation.pianoroll_encoder_decoder import PianorollEncoderDecoder
from modules.data.representation.pianoroll_sequence import PianorollSequence


def extract_pianoroll_sequences(quantized_sequence, start_step=0, min_steps_discard=None,
                                max_steps_discard=None, max_steps_truncate=None):
    """Extracts a polyphonic track from the given quantized NoteSequence.

    Currently, this extracts only one pianoroll from a given track.

    Args:
      quantized_sequence: A quantized NoteSequence.
      start_step: Start extracting a sequence at this time step. Assumed
          to be the beginning of a bar.
      min_steps_discard: Minimum length of tracks in steps. Shorter tracks are
          discarded.
      max_steps_discard: Maximum length of tracks in steps. Longer tracks are
          discarded. Mutually exclusive with `max_steps_truncate`.
      max_steps_truncate: Maximum length of tracks in steps. Longer tracks are
          truncated. Mutually exclusive with `max_steps_discard`.

    Returns:
      pianoroll_seqs: A python list of PianorollSequence instances.
      stats: A dictionary mapping string names to `statistics.Statistic` objects.

    Raises:
      ValueError: If both `max_steps_discard` and `max_steps_truncate` are
          specified.
    """

    if (max_steps_discard, max_steps_truncate).count(None) == 0:
        raise ValueError(
            'Only one of `max_steps_discard` and `max_steps_truncate` can be '
            'specified.')
    sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

    # pylint: disable=g-complex-comprehension
    stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
                 ['pianoroll_tracks_truncated_too_long',
                  'pianoroll_tracks_discarded_too_short',
                  'pianoroll_tracks_discarded_too_long',
                  'pianoroll_tracks_discarded_more_than_1_program'])
    # pylint: enable=g-complex-comprehension

    steps_per_bar = sequences_lib.steps_per_bar_in_quantized_sequence(
        quantized_sequence)

    # Create a histogram measuring lengths (in bars not steps).
    stats['pianoroll_track_lengths_in_bars'] = statistics.Histogram(
        'pianoroll_track_lengths_in_bars',
        [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, 1000])

    # Allow only 1 program.
    programs = set()
    for note in quantized_sequence.notes:
        programs.add(note.program)
    if len(programs) > 1:
        stats['pianoroll_tracks_discarded_more_than_1_program'].increment()
        return [], list(stats.values())

    # Translate the quantized sequence into a PianorollSequence.
    pianoroll_seq = PianorollSequence(quantized_sequence=quantized_sequence,
                                      start_step=start_step)

    pianoroll_seqs = []
    num_steps = pianoroll_seq.num_steps

    if min_steps_discard is not None and num_steps < min_steps_discard:
        stats['pianoroll_tracks_discarded_too_short'].increment()
    elif max_steps_discard is not None and num_steps > max_steps_discard:
        stats['pianoroll_tracks_discarded_too_long'].increment()
    else:
        if max_steps_truncate is not None and num_steps > max_steps_truncate:
            stats['pianoroll_tracks_truncated_too_long'].increment()
            pianoroll_seq.set_length(max_steps_truncate)
        pianoroll_seqs.append(pianoroll_seq)
        stats['pianoroll_track_lengths_in_bars'].increment(
            num_steps // steps_per_bar)
    return pianoroll_seqs, list(stats.values())


class PianoRollConverter(music_vae_data.BaseNoteSequenceConverter):
    """ TODO - Class DOC """

    def __init__(self, min_pitch=music_vae_data.PIANO_MIN_MIDI_PITCH, max_pitch=music_vae_data.PIANO_MAX_MIDI_PITCH,
                 min_steps_discard=None, max_steps_discard=None, max_bars=None, slice_bars=None, add_end_token=False,
                 steps_per_quarter=4, quarters_per_bar=4, pad_to_total_time=False, max_tensors_per_notesequence=None,
                 presplit_on_time_changes=True):
        self._min_pitch = min_pitch
        self._max_pitch = max_pitch
        self._min_steps_discard = min_steps_discard
        self._max_steps_discard = max_steps_discard
        self._steps_per_quarter = steps_per_quarter
        self._steps_per_bar = steps_per_quarter * quarters_per_bar
        self._slice_steps = self._steps_per_bar * slice_bars if slice_bars else None
        self._pad_to_total_time = pad_to_total_time

        self._pianoroll_extractor_fn = functools.partial(
            extract_pianoroll_sequences,
            start_step=0,
            min_steps_discard=self._min_steps_discard,
            max_steps_discard=self._max_steps_discard,
            max_steps_truncate=self._steps_per_bar * max_bars if max_bars else None
        )

        # We have two classes for event: one for the velocity and the other for the repeated flag
        num_classes = 2 * (max_pitch - min_pitch + 1)

        self._pr_encoder_decoder = PianorollEncoderDecoder(input_size=num_classes + add_end_token)

        input_depth = num_classes + add_end_token
        output_depth = num_classes + add_end_token

        super(PianoRollConverter, self).__init__(
            input_depth=input_depth,
            input_dtype=np.bool,
            output_depth=output_depth,
            output_dtype=np.bool,
            end_token=output_depth - 1 if add_end_token else None,
            presplit_on_time_changes=presplit_on_time_changes,
            max_tensors_per_notesequence=max_tensors_per_notesequence
        )

    def _to_tensors_fn(self, note_sequence):
        """Converts NoteSequence to unique sequences."""

        # Quantize sequence
        try:
            quantized_sequence = note_seq.quantize_note_sequence(note_sequence, self._steps_per_quarter)
            if note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) != self._steps_per_bar:
                return music_vae_data.ConverterTensors()
        except (note_seq.BadTimeSignatureError, note_seq.NonIntegerStepsPerBarError, note_seq.NegativeTimeError):
            return music_vae_data.ConverterTensors()

        # TODO - Keep only notes for valid programs

        # Extract piano-roll events
        event_lists, stats = self._pianoroll_extractor_fn(quantized_sequence)
        # tf.compat.v1.logging.info("Piano-roll Sequence Stats: \n{}".format(stats))

        # Pad events with silence to total quantization time if the sequence is shorter
        if self._pad_to_total_time:
            for e in event_lists:
                if len(e) < self._slice_steps:
                    e.set_length(self._slice_steps)

        # Slice events in bars If sequence is not multiple of slice_steps it will be truncated
        if self._slice_steps:
            sliced_event_tuples = []
            for l in event_lists:
                for i in range(0, len(l) - self._slice_steps + 1, self._slice_steps):
                    sliced_event_tuples.append(tuple(l[i: i + self._slice_steps]))
        else:
            sliced_event_tuples = [tuple(l) for l in event_lists]

        unique_event_tuples = list(set(sliced_event_tuples))
        unique_event_tuples = music_vae_data.maybe_sample_items(unique_event_tuples,
                                                                self.max_tensors_per_notesequence,
                                                                self.is_training)

        # TODO - What is the purpose of an end token? Variable length inputs?
        rolls = []
        for t in unique_event_tuples:
            if self.end_token is not None:
                t_roll = list(t) + [self._pr_encoder_decoder.input_size - 1]
            else:
                t_roll = t
            rolls.append(np.vstack([self._pr_encoder_decoder.events_to_input(t_roll, i) for i in range(len(t_roll))]))

        input_seqs = rolls
        output_seqs = rolls

        return music_vae_data.ConverterTensors(inputs=input_seqs, outputs=output_seqs)

    def to_tensors(self, item):
        note_sequence = item
        return music_vae_data.split_process_and_combine(note_sequence,
                                                        self._presplit_on_time_changes,
                                                        self.max_tensors_per_notesequence,
                                                        self.is_training, self._to_tensors_fn)

    def from_tensors(self, samples, controls=None):
        output_sequences = []
        for s in samples:
            if self.end_token is not None:
                end_i = np.where(s[:, self.end_token])
                if len(end_i):
                    s = s[:end_i[0]]

            pr_sequence = PianorollSequence(
                events_list=self._pr_encoder_decoder.input_to_events(s),
                steps_per_quarter=self._steps_per_quarter,
                min_pitch=self._min_pitch,
                max_pitch=self._max_pitch
            )

            output_sequence = pr_sequence.to_sequence()
            output_sequences.append(output_sequence)

        return output_sequences
