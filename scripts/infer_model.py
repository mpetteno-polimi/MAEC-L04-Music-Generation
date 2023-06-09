from definitions import ConfigSections
from modules.data.augmentation.noteseq import NoteSequenceAugmenter
from modules.data.converters.pianoroll import PianoRollConverter
from modules.data.loaders.tfrecord import TFRecordLoader
from modules.model.cnn import CNN
from modules.model.decoder import HierarchicalDecoder
from modules.model.encoder import BidirectionalLstmEncoder
from modules.model.maec_vae import MaecVAE
from modules.utilities import config
from definitions import Paths


if __name__ == "__main__":
    representation_config = config.load_configuration_section(ConfigSections.REPRESENTATION)
    test_config = config.load_configuration_section(ConfigSections.TEST)

    # Load data
    print('loading data... ', end='')
    test_tfrecord = test_config.get("test_examples_path")

    data_converter = PianoRollConverter(
        min_pitch=representation_config.get("piano_min_midi_pitch"),
        max_pitch=representation_config.get("piano_max_midi_pitch"),
        max_steps_discard=representation_config.get("max_steps_discard"),
        max_bars=representation_config.get("max_bars"),
        slice_bars=representation_config.get("slice_bars"),
        steps_per_quarter=representation_config.get("steps_per_quarter"),
        quarters_per_bar=representation_config.get("quarters_per_bar")
    )
    data_converter.set_mode('infer')

    data_augmenter = NoteSequenceAugmenter(
        transpose_range=(representation_config.get("transposition_min"), representation_config.get("transposition_max"))
    )

    tfrecord_loader = TFRecordLoader(
        converter=data_converter,
        augmenter=data_augmenter
    )

    test_dataset = tfrecord_loader.load(test_tfrecord)
    # for elem in test_dataset:
    # print('elem', elem)
    print('Done')

    print('Loading model... ', end='')
    # Create the model
    vae = MaecVAE(
        encoder=BidirectionalLstmEncoder(),
        decoder=HierarchicalDecoder(output_depth=data_converter.output_depth),
        cnn=CNN(input_shape=(data_converter.seq_length, data_converter.seq_length, 1))
    )

    n_test_files = int(test_config.get('n_test_files'))
    test_batch_size = int(test_config.get('test_batch_size'))
    input_steps_len = int(representation_config.get('num_bars')) * int(representation_config.get('slice_bars'))
    input_feature_len = 2 * (int(representation_config.get('piano_max_midi_pitch')) - int(
        representation_config.get('piano_min_midi_pitch')) + 1)

    input_shape = (test_batch_size, input_steps_len, input_feature_len)
    print('shape:', input_shape, end='... ')
    vae.build(input_shape)

    # load checkpoint weights
    checkpoint_weights_file_name = test_config.get('infer_checkpoint_file_name')
    vae.load_weights(Paths.TRAIN_CHECK_DIR / checkpoint_weights_file_name)
    print('Done')
    # vae.summary()

    # data = keras.backend.random_normal(shape=(2, 256, 176), mean=0, stddev=1)
    results = []
    # Start inference
    print('Predicting... ', end='')
    for idx, test_batch in enumerate(test_dataset):
        # test_batch = keras.Input(tensor=test_batch)
        if idx >= n_test_files:
            break
        results.append(vae.sample(test_batch[0], pianoroll_format=True))

    print('Done')
    print(results)
