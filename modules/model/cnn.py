# TODO - DOC

from keras import applications

from definitions import ConfigSections
from modules import utilities


class CNN(object):

    def __init__(self):
        self._model_config = utilities.load_configuration_section(ConfigSections.MODEL)
        self._model = None

    def build(self, seq_length):
        cnn_id = self._model_config.get("cnn_id")

        if cnn_id == 'mobilenet_v1':
            self._model = applications.MobileNet(
                input_shape=(seq_length, seq_length, 1),
                alpha=self._model_config.get("cnn_alpha"),
                depth_multiplier=self._model_config.get("cnn_depth_multiplier"),
                dropout=self._model_config.get("cnn_dropout"),
                include_top=False,
                weights=None,
                pooling="avg"
            )
        else:
            raise ValueError("CNN configuration {} not supported".format(cnn_id))

    def embed(self, input_):
        return self._model(input_)

    def summary(self):
        self._model.summary()
