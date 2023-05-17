""" TODO - Module DOC """

import abc
import collections
from abc import ABC, abstractmethod
from pathlib import Path

import note_seq


class DataConverter(ABC):
    """ TODO - Class DOC """

    def __init__(self, input_path: Path, output_path: Path, collection_name: str):
        self.input_path = input_path
        self.output_path = output_path
        self.collection_name = collection_name

    @abstractmethod
    def convert_train(self, train_data: [str]) -> None:
        pass

    @abstractmethod
    def convert_validation(self, validation_data: [str]) -> None:
        pass

    @abstractmethod
    def convert_test(self, test_data: [str]) -> None:
        pass


class ConverterTensors(collections.namedtuple('ConverterTensors', ['inputs', 'outputs'])):
    """Tuple of tensors output by `to_tensors` method in converters.
    Attributes:
    inputs: Input tensors to feed to the encoder.
    outputs: Output tensors to feed to the decoder.
    """

    def __new__(cls, inputs=None, outputs=None):
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        return super(ConverterTensors, cls).__new__(cls, inputs, outputs)


class BaseNoteSequenceConverter(object):
    """Base class for data converters between items and tensors.
    Inheriting classes must implement the following abstract methods:
    -`to_tensors`
    -`from_tensors`
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 input_depth,
                 input_dtype,
                 output_depth,
                 output_dtype):
        """Initializes BaseNoteSequenceConverter.
        Args:
          input_depth: Depth of final dimension of input (encoder) tensors.
          input_dtype: DType of input (encoder) tensors.
          output_depth: Depth of final dimension of output (decoder) tensors.
          output_dtype: DType of output (decoder) tensors.
        """
        self._input_depth = input_depth
        self._input_dtype = input_dtype
        self._output_depth = output_depth
        self._output_dtype = output_dtype
        self._str_to_item_fn = note_seq.NoteSequence.FromString
        self._mode = None

    def set_mode(self, mode):
        if mode not in ['train', 'eval', 'infer']:
            raise ValueError('Invalid mode: %s' % mode)
        self._mode = mode

    @property
    def is_training(self):
        return self._mode == 'train'

    @property
    def is_inferring(self):
        return self._mode == 'infer'

    @property
    def str_to_item_fn(self):
        return self._str_to_item_fn

    @property
    def input_depth(self):
        """Dimension of inputs (to encoder) at each timestep of the sequence."""
        return self._input_depth

    @property
    def input_dtype(self):
        """DType of inputs (to encoder)."""
        return self._input_dtype

    @property
    def output_depth(self):
        """Dimension of outputs (from decoder) at each timestep of the sequence."""
        return self._output_depth

    @property
    def output_dtype(self):
        """DType of outputs (from decoder)."""
        return self._output_dtype

    @abc.abstractmethod
    def to_tensors(self, item):
        """Python method that converts `item` into list of `ConverterTensors`."""
        pass

    @abc.abstractmethod
    def from_tensors(self, samples):
        """Python method that decodes model samples into list of items."""
        pass
