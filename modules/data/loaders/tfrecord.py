""" TODO - Module DOC """

from pathlib import Path
from typing import Any

import magenta
import tensorflow as tf
import magenta.models.music_vae.data as music_vae_data
import magenta.models.music_vae.configs as music_vae_configs
from magenta.contrib import training as contrib_training
from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_model

from modules import utilities
from modules.data.loaders.dataloader import DataLoader


class TFRecordLoader(DataLoader):
    """ TODO - Class DOC """

    def __init__(self, input_path: Path, collection_name: str):
        super().__init__(input_path)
        self.collection_name = collection_name

    def load_train(self, source_datasets: Any = None) -> tf.data.Dataset:
        return self._load("train", source_datasets)

    def load_validation(self, source_datasets: Any = None) -> tf.data.Dataset:
        return self._load("validation", source_datasets)

    def load_test(self, source_datasets: Any = None) -> tf.data.Dataset:
        return self._load("test", source_datasets)

    def _load(self, tfrecord_file_label: str, source_datasets: Any) -> tf.data.Dataset:
        tfrecords_path = utilities.get_tfrecords_path_for_source_datasets(source_datasets, self.input_path,
                                                                          tfrecord_file_label, self.collection_name)

        # TODO - Move
        config = pianoroll_rnn_nade_model.default_configs['rnn-nade']
        batch_size = 5
        input_size = config.encoder_decoder.input_size

        return magenta.common.get_padded_batch(
            file_list=tfrecords_path, batch_size=batch_size, input_size=input_size,
            shuffle="train" in tfrecord_file_label)
