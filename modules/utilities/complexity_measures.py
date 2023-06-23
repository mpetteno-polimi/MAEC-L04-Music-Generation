import numpy as np
import pretty_midi


def sanitize(midi: pretty_midi.PrettyMIDI):
    assert len(midi.instruments) == 1, \
        f'MusicVAE output is expected to have a single MIDI instrument. Found ({len(midi.instruments)}) instruments instead'

    for instrument in midi.instruments:
        instrument.pitch_bends = []
        instrument.control_changes = []
    return midi


def get_notes_list(midi: pretty_midi.PrettyMIDI):
    return midi.instruments[0].notes


def get_pitch_list(midi: pretty_midi.PrettyMIDI):
    return [note.pitch for note in get_notes_list(midi)]


def count_onsets(midi: pretty_midi.PrettyMIDI):
    return len(get_notes_list(midi))


def toussaint(midi: pretty_midi.PrettyMIDI, binary: bool = False):
    midi = sanitize(midi)

    hierarchy = np.array(
        [6, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1])

    max_sum = np.array(
        [0, 6, 11, 15, 19, 22, 25, 28, 31, 33, 35, 37, 39, 41, 43, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
         58, 59, 60, 61, 62, 63])

    n_pulses = len(hierarchy)
    n_onsets = count_onsets(midi)

    assert n_onsets <= n_pulses, \
        f'Maximum number of onsets should be less than the number of pulses ({n_pulses}). Found ({n_onsets}).'

    velocity = np.zeros(n_pulses)
    dur_16th = midi.get_end_time() / n_pulses
    midi_iter = iter(get_notes_list(midi))
    note = next(midi_iter)

    for i in range(n_pulses):
        if i * dur_16th <= note.start < (i + 1) * dur_16th:
            velocity[i] = 1. if binary else note.velocity
            note = next(midi_iter)

    assert np.count_nonzero(velocity) == n_onsets, \
        f'Toussaint found ({np.count_nonzero(velocity)}) onsets. Expected ({n_onsets}).'

    metricity = np.sum(hierarchy * velocity)
    idx = count_onsets(midi)
    metric = max_sum[idx] - metricity

    return metric


def note_density(midi: pretty_midi.PrettyMIDI):
    midi = sanitize(midi)
    metric = count_onsets(midi) / midi.get_end_time()
    return metric


def pitch_range(midi: pretty_midi.PrettyMIDI):
    midi = sanitize(midi)
    pitches = [note.pitch for note in midi.instruments[0].notes]
    metric = (np.max(pitches) - np.min(pitches)) / 88
    return metric


def contour(midi: pretty_midi.PrettyMIDI):
    midi = sanitize(midi)
    pitch_seq = get_pitch_list(midi)
    metric = np.diff(pitch_seq) / 88
    return metric


def get_all_complexity_values(midi: pretty_midi.PrettyMIDI):
    return toussaint(midi), note_density(midi), pitch_range(midi), contour(midi)

if __name__ == '__main__':
    midi = pretty_midi.PrettyMIDI('D:/magenta_out/seed99_day23-06_16-15config-cat-mel_2bar_big/grid_point_0/sample_0/midi_file_0.mid')
    # midi = pretty_midi.PrettyMIDI('D:/magenta_out/seed99_day23-06_16-15config-cat-mel_2bar_big/grid_point_0/sample_1/midi_file_1.mid')
    # midi = pretty_midi.PrettyMIDI('D:/magenta_out/seed99_23-06_12-51/grid_point_0/sample_2/midi_file_2.mid')
    midi = pretty_midi.PrettyMIDI('D:/magenta_out/seed99_23-06_12-51/grid_point_0/sample_3/midi_file_3.mid')
    print('Toussaint:', toussaint(midi))
    print('Note density:', note_density(midi))
    print('Pitch range:', pitch_range(midi))
    print('Contour:', contour(midi))

