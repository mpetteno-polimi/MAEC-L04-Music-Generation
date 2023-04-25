# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes for converting between pianoroll input and model input/output."""

import numpy as np

from modules.data.representation.pianoroll_sequence import PianorollEvent

MIDI_MAX_VELOCITY = 127


class PianorollEncoderDecoder:
    """An EventSequenceEncoderDecoder that produces a pianoroll encoding. TODO - Modify DOC
      Inputs are binary arrays with active pitches (with some offset) at each step
      set to 1 and inactive pitches set to 0.

      Events are PianorollSequence events, which are tuples of active pitches
      (with some offset) at each step.
      """

    def __init__(self, input_size=88):
        """Initialize a PianorollEncoderDecoder object.

        Args:
          input_size: The size of the input vector.
        """
        self._input_size = input_size

    @property
    def input_size(self):
        return self._input_size

    def events_to_input(self, events, position):
        """Returns the input vector for the given position in the event sequence.

        Args:
          events: A list-like sequence of PianorollSequence events.
          position: An integer event position in the event sequence.

        Returns:
          An input vector, a list of floats.
        """
        input_ = np.zeros(self.input_size, np.float32)
        for pianoroll_event in list(events[position]):
            pitch_index = pianoroll_event.pitch * 2
            input_[pitch_index] = pianoroll_event.velocity / MIDI_MAX_VELOCITY
            # Can't have repeated notes at the beginning of the example
            input_[pitch_index + 1] = pianoroll_event.is_repeated if position != 0 else 0
        return input_

    def input_to_events(self, input_):
        """
        TODO - Function DOC
        """
        events = []
        for frame in input_:
            velocities = frame[::2]
            is_repeated = frame[1::2]
            active_pitches = np.where(velocities)[0]
            events.append(tuple(PianorollEvent(
                pitch=active_pitch,
                velocity=round(velocities[active_pitch] * MIDI_MAX_VELOCITY),
                is_repeated=is_repeated[active_pitch]
            ) for active_pitch in active_pitches))
        return events
