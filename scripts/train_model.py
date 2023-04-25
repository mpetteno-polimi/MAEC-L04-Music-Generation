
from modules.data.converters.pianoroll import PianoRollConverter

if __name__ == "__main__":

    # TODO - Load dataset
    PIANO_MIN_MIDI_PITCH = 21
    PIANO_MAX_MIDI_PITCH = 108
    QUARTER_PER_BARS = 4
    STEPS_PER_QUARTER = 4
    SLICE_BARS = 16
    STEPS_PER_BAR = QUARTER_PER_BARS * STEPS_PER_QUARTER

    pr_16bar_converter = PianoRollConverter(
        min_pitch=PIANO_MIN_MIDI_PITCH,
        max_pitch=PIANO_MAX_MIDI_PITCH,
        min_steps_discard=STEPS_PER_BAR,
        max_steps_discard=None,
        max_bars=None,
        slice_bars=SLICE_BARS,
        steps_per_quarter=STEPS_PER_QUARTER,
        quarters_per_bar=QUARTER_PER_BARS,
        pad_to_total_time=True,
        max_tensors_per_notesequence=None,
        presplit_on_time_changes=True
    )

    # TODO - Launch model train
