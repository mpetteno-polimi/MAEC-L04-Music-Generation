""" TODO - Module DOC """

import functools

import tensorflow as tf
import note_seq
import numpy as np
from magenta.pipelines import pianoroll_pipeline
from magenta.models.music_vae import data as music_vae_data


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
            pianoroll_pipeline.extract_pianoroll_sequences,
            start_step=0,
            min_steps_discard=self._min_steps_discard,
            max_steps_discard=self._max_steps_discard,
            max_steps_truncate=self._steps_per_bar * max_bars if max_bars else None
        )

        num_classes = max_pitch - min_pitch + 1

        self._pr_encoder_decoder = note_seq.PianorollEncoderDecoder(input_size=num_classes + add_end_token)

        input_depth = num_classes + 1 + add_end_token  # TODO - +1 for rest token??
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

        # TODO - Keep only notes for programs

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
            rolls.append(
                np.vstack(
                    [self._pr_encoder_decoder.events_to_input(t_roll, i).astype(np.bool) for i in range(len(t_roll))]
                )
            )

        # TODO - Why expand piano-roll input y-size from 88 to 89?
        input_seqs = [np.append(roll, np.expand_dims(np.all(roll == 0, axis=1), axis=1), axis=1) for roll in rolls]
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
            events_list = [frozenset(np.where(e)[0]) for e in s]

            pr_sequence = note_seq.PianorollSequence(
                events_list=events_list,
                steps_per_quarter=self._steps_per_quarter,
                min_pitch=self._min_pitch,
                max_pitch=self._max_pitch
            )

            output_sequence = pr_sequence.to_sequence(velocity=music_vae_data.OUTPUT_VELOCITY)
            output_sequences.append(output_sequence)

        return output_sequences
