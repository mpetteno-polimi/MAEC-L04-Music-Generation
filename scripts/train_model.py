
from definitions import ConfigSections
from modules import utilities
from modules.data.augmentation.noteseq import NoteSequenceAugmenter
from modules.data.converters.pianoroll import PianoRollConverter
from modules.data.loaders.tfrecord import TFRecordLoader
from modules.model.cnn import CNN
from modules.model.decoder import HierarchicalDecoder
from modules.model.encoder import BidirectionalLstmEncoder
from modules.model.maec_vae import MaecVAE
from modules.training.trainer import Trainer

if __name__ == "__main__":
    representation_config = utilities.config.load_configuration_section(ConfigSections.REPRESENTATION)
    training_config = utilities.config.load_configuration_section(ConfigSections.TRAINING)

    # Load data
    train_tfrecord = training_config.get("train_examples_path")
    validation_tfrecord = training_config.get("validation_examples_path")

    data_converter = PianoRollConverter(
        min_pitch=representation_config.get("piano_min_midi_pitch"),
        max_pitch=representation_config.get("piano_max_midi_pitch"),
        max_steps_discard=representation_config.get("max_steps_discard"),
        max_bars=representation_config.get("max_bars"),
        slice_bars=representation_config.get("slice_bars"),
        steps_per_quarter=representation_config.get("steps_per_quarter"),
        quarters_per_bar=representation_config.get("quarters_per_bar")
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
    # sequence = next(iter(train_dataset))

    # Create the model
    seq_length = data_converter.seq_length
    frame_length = data_converter.input_depth
    pianoroll_shape = (seq_length, frame_length)
    ssm_shape = (seq_length, seq_length)
    vae = MaecVAE(
        encoder=BidirectionalLstmEncoder(),
        decoder=HierarchicalDecoder(),
        cnn=CNN(ssm_shape)
    )
    # vae.summary()

    # Start training
    trainer = Trainer(model=vae)
    trainer.train(train_dataset, validation_dataset)
