""" TODO - Module DOC """

from typing import Any

import tensorflow as tf
import magenta.models.music_vae.data as music_vae_data
import magenta.models.music_vae.configs as music_vae_configs
from magenta.contrib import training as contrib_training

from definitions import Paths
from modules.data.loaders.dataloader import DataLoader


class NoteSequenceLoader(DataLoader):
    """ TODO - Class DOC """

    def load_train(self, metadata: Any = None) -> tf.data.Dataset:
        return self._load("train")

    def load_validation(self, metadata: Any = None) -> tf.data.Dataset:
        return self._load("validation")

    def load_test(self, metadata: Any = None) -> tf.data.Dataset:
        return self._load("test")

    def _load(self, tfrecord_file_label: str) -> tf.data.Dataset:
        record_paths = list(self.input_path.glob('*-{}.tfrecord'.format(tfrecord_file_label)))
        record_paths = [str(record_path) for record_path in record_paths]

        converter = music_vae_data.OneHotMelodyConverter(steps_per_quarter=1, slice_bars=2,
                                                         max_tensors_per_notesequence=1)

        config = music_vae_configs.Config(
            hparams=contrib_training.HParams(batch_size=1),
            note_sequence_augmenter=None,
            data_converter=converter,
            train_examples_path=record_paths,
            eval_examples_path=record_paths
        )

        ds = music_vae_data.get_dataset(config, is_training="train" in tfrecord_file_label)
        return ds


if __name__ == '__main__':
    tf.get_logger().setLevel('INFO')

    loader = NoteSequenceLoader(Paths.DATA_RECORDS_DIR)
    loader.load_train()
