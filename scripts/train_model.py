from definitions import ConfigSections
from modules.data.augmentation.noteseq import NoteSequenceAugmenter
from modules.data.converters.pianoroll import PianoRollConverter
from modules.data.loaders.tfrecord import TFRecordLoader
from modules.model.cnn import CNN
from modules.model.decoder import HierarchicalDecoder
from modules.model.encoder import BidirectionalLstmEncoder
from modules.model.maec_vae import MaecVAE
from modules.training.trainer import Trainer
from modules.utilities import config

if __name__ == "__main__":
    representation_config = config.load_configuration_section(ConfigSections.REPRESENTATION)
    training_config = config.load_configuration_section(ConfigSections.TRAINING)

    # Load data
    print('loading data... ', end='')
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
    print('Done')
    print('Creating model... ', end='')
    # Create the model
    vae = MaecVAE(
        encoder=BidirectionalLstmEncoder(),
        decoder=HierarchicalDecoder(output_depth=data_converter.output_depth),
        cnn=CNN(input_shape=(data_converter.seq_length, data_converter.seq_length, 1))
    )

    print('Done')
    # vae.summary()

    # Start training
    print('Started training')
    trainer = Trainer(model=vae)
    history = trainer.train(train_dataset, validation_dataset)
