
from definitions import ConfigSections
from modules import utilities
from modules.data.augmentation.noteseq import NoteSequenceAugmenter
from modules.data.converters.pianoroll import PianoRollConverter
from modules.data.loaders.tfrecord import TFRecordLoader

if __name__ == "__main__":
    representation_config = utilities.load_configuration_section(ConfigSections.REPRESENTATION)
    training_config = utilities.load_configuration_section(ConfigSections.TRAINING)

    train_tfrecord = training_config.get("train_examples_path")
    validation_tfrecord = training_config.get("validation_examples_path")

    data_converter = PianoRollConverter(
        min_pitch=representation_config.get("piano_min_midi_pitch"),
        max_pitch=representation_config.get("piano_max_midi_pitch"),
        max_steps_discard=None,
        max_bars=None,
        slice_bars=representation_config.get("slice_bars"),
        steps_per_quarter=representation_config.get("steps_per_quarter"),
        quarters_per_bar=representation_config.get("quarters_per_bar"),
        pad_to_total_time=True,
        max_tensors_per_notesequence=None,
        presplit_on_time_changes=True
    )
    data_converter.set_mode('train')

    data_augmenter = NoteSequenceAugmenter(
        transpose_range=(representation_config.get("transposition_min"), representation_config.get("transposition_max"))
    )

    tfrecord_loader = TFRecordLoader(
        converter=data_converter,
        augmenter=data_augmenter
    )

    train_dataset = tfrecord_loader.load(train_tfrecord)
    validation_dataset = tfrecord_loader.load(validation_tfrecord)

    sequence = next(iter(train_dataset))
    pass

    # TODO - Launch model train
