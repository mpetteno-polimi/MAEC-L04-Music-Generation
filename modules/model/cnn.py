# TODO - DOC

from keras import applications, layers

from definitions import ConfigSections
from modules import utilities


class CNN(layers.Layer):

    def __init__(self, input_shape, name="cnn", **kwargs):
        super(CNN, self).__init__(name=name, **kwargs)
        self._model_config = utilities.config.load_configuration_section(ConfigSections.MODEL)

        cnn_id = self._model_config.get("cnn_id")
        if cnn_id == 'mobilenet_v1':
            self._model = applications.MobileNet(
                input_shape=input_shape,
                alpha=self._model_config.get("cnn_alpha"),
                depth_multiplier=self._model_config.get("cnn_depth_multiplier"),
                dropout=self._model_config.get("cnn_dropout"),
                include_top=False,
                weights=None,
                pooling="avg"
            )
        elif cnn_id == 'mobilenet_v2':
            self._model = applications.MobileNetV2(
                input_shape=input_shape,
                alpha=self._model_config.get("cnn_alpha"),
                include_top=False,
                weights=None,
                pooling="avg"
            )
        else:
            raise ValueError("CNN configuration {} not supported".format(cnn_id))

    def call(self, inputs, *args, **kwargs):
        return self._model(inputs)
