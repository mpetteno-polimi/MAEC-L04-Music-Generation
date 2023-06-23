import numpy as np
import pretty_midi


def sanitize(midi: pretty_midi.PrettyMIDI, bars: int):
    assert len(midi.instruments) == 1, \
        f'MusicVAE output is expected to have a single MIDI instrument. Found ({len(midi.instruments)}) instruments instead'

    for instrument in midi.instruments:
        instrument.pitch_bends = []
        instrument.control_changes = []
        instrument.notes = [note for note in instrument.notes if note.velocity > 0]

    midi.remove_invalid_notes()

    n_pulses = 16 * bars
    n_onsets = count_onsets(midi)

    assert n_onsets <= n_pulses, \
        f'Maximum number of onsets should be less than the number of pulses ({n_pulses}). Found ({n_onsets}) onsets.'

    return midi


def get_note_list(midi: pretty_midi.PrettyMIDI):
    notes = midi.instruments[0].notes
    assert len(notes) > 0, 'It appears that the input MIDI does not contain any note.'
    return notes


def get_pitch_list(midi: pretty_midi.PrettyMIDI):
    return [note.pitch for note in get_note_list(midi)]


def get_velocity_list(midi: pretty_midi.PrettyMIDI):
    return [note.velocity / 127 for note in get_note_list(midi)]


def count_onsets(midi: pretty_midi.PrettyMIDI):
    return len(get_note_list(midi))


def toussaint(midi: pretty_midi.PrettyMIDI, bars: int = 16, binary: bool = True):
    hierarchy = np.array([5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]).repeat(bars)
    max_sum = np.cumsum(np.sort(hierarchy)[::-1])

    n_pulses = len(hierarchy)
    n_onsets = count_onsets(midi)

    velocity = np.zeros(n_pulses)
    dur_16th = midi.get_end_time() / n_pulses
    midi_iter = iter(get_note_list(midi))
    note = next(midi_iter)

    for i in range(n_pulses):
        if i * dur_16th <= note.start < (i + 1) * dur_16th:
            velocity[i] = 1. if binary else note.velocity / 127
            note = next(midi_iter, None)
            if note is None:
                break

    metricity = np.sum(hierarchy * velocity)
    metric = max_sum[n_onsets] - metricity

    return metric


def note_density(midi: pretty_midi.PrettyMIDI, bars: int = 16, binary: bool = True):
    count = count_onsets(midi) if binary else np.sum(get_velocity_list(midi))
    metric = count / (16 * bars)
    return metric


def pitch_range(midi: pretty_midi.PrettyMIDI):
    pitch_list = get_pitch_list(midi)
    metric = (np.max(pitch_list) - np.min(pitch_list)) / 88
    return metric


def contour(midi: pretty_midi.PrettyMIDI):
    pitch_list = get_pitch_list(midi)
    metric = np.sum(np.diff(pitch_list)) / 88
    return metric


# Example
if __name__ == '__main__':

    midi_paths = [
        "D:\magenta_out\config_hierdec-mel_16bar_seed_99\grid_point_0\sample_0\sample_0_midi_out.mid",
        "D:\magenta_out\config_hierdec-mel_16bar_seed_99\grid_point_0\sample_1\sample_1_midi_out.mid",
    ]

    for midi_path in midi_paths:
        midi = pretty_midi.PrettyMIDI(midi_path)
        midi = sanitize(midi, bars=16)

        print('Toussaint:', toussaint(midi, bars=16))
        print('Note density:', note_density(midi, bars=16))
        print('Pitch range:', pitch_range(midi))
        print('Contour:', contour(midi))
        print()
