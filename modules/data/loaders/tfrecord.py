""" TODO - Module DOC """

import functools
from typing import AnyStr

import tensorflow as tf

from modules.data.loaders.dataloader import DataLoader


class TFRecordLoader(DataLoader):

    def __init__(self, converter, augmenter=None):
        super().__init__()
        self._converter = converter
        self._augmenter = augmenter

    def load(self, filenames: AnyStr):
        return self._get_dataset(filenames)

    def _get_dataset(self, filenames):

        def load_dataset():
            ignore_order = tf.data.Options()
            ignore_order.experimental_deterministic = False
            tfrecord_data = tf.data.TFRecordDataset(filenames)
            tfrecord_data = tfrecord_data.with_options(ignore_order)
            return tfrecord_data

        def convert_to_tensors_op(item_scalar, converter):
            """TensorFlow's op that converts item into output tensors.
            Args:
              item_scalar: A scalar of type tf.String containing the raw item to be
                converted to tensors.
              converter: The DataConverter to be used.
            Returns:
              inputs: A Tensor, shaped [num encoded seqs, max(lengths), input_depth],
                  containing the padded input encodings.
              outputs: A Tensor, shaped [num encoded seqs, max(lengths), output_depth],
                  containing the padded output encodings resulting from the input.
            """

            def _convert(item_str):
                item = converter.str_to_item_fn(item_str.numpy())
                tensors = converter.to_tensors(item)
                return tensors.inputs, tensors.outputs

            inputs, outputs = tf.py_function(
                _convert,
                inp=[item_scalar],
                Tout=[converter.input_dtype, converter.output_dtype],
                name='convert')
            inputs.set_shape([None, converter.seq_length, converter.input_depth])
            outputs.set_shape([None, converter.seq_length, converter.output_depth])

            return inputs, outputs

        batch_size = self._training_config.get("batch_size")

        dataset = load_dataset()
        dataset = dataset.map(self._augmenter.tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            tf.autograph.experimental.do_not_convert(
                functools.partial(convert_to_tensors_op, converter=self._converter)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.unbatch()
        dataset = dataset.cache() if self._training_config.get("cache_dataset") else dataset
        dataset = dataset.shuffle(buffer_size=10 * batch_size, reshuffle_each_iteration=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size)
        dataset.padded_batch(batch_size=batch_size, drop_remainder=True)

        return dataset

