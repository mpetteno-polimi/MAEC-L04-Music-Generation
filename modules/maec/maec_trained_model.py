
import numpy as np
import tensorflow as tf

from magenta.models.music_vae import TrainedModel

from definitions import ConfigSections
from modules.utilities import sampling
from modules.utilities import config as config_file


class MAECTrainedModel(TrainedModel):

    def __init__(self, config, batch_size, checkpoint_dir_or_path=None,
                 var_name_substitutions=None, session_target='', **sample_kwargs):
        super(MAECTrainedModel, self).__init__(config, batch_size,
                                               checkpoint_dir_or_path=checkpoint_dir_or_path,
                                               var_name_substitutions=var_name_substitutions,
                                               session_target=session_target, **sample_kwargs)
        self._config_file = config_file.load_configuration_section(ConfigSections.LATENT_SPACE_SAMPLING)

    def grid_sample(self, grid_points, n_samples_per_grid_point, sigma, length=None, temperature=1.0):
        """ TODO

        Args:
          grid_points: .
          n_samples_per_grid_point: .
          sigma: .
          length: The maximum length of a sample in decoder iterations. Required
            if end tokens are not being used.
          temperature: The softmax temperature to use (if applicable).
        Returns:
          A list of samples as NoteSequence objects.
        Raises:
          ValueError: If `length` is not specified and an end token is not being
            used.
        """

        if length is None:
            length = self._config.hparams.max_seq_len

        feed_dict = {
            self._temperature: temperature,
            self._max_length: length
        }

        tf.compat.v1.logging.info("Performing gaussian sampling...")
        batched_gaussian_samples = sampling.batch_gaussian_sampling(
            d=self._config.hparams.z_size,
            grid_points=grid_points,
            samples_per_point=n_samples_per_grid_point,
            sigma=sigma
        )

        tf.compat.v1.logging.info("Decoding samples...")
        outputs = []
        for idx in range(grid_points.shape[0]):
            feed_dict[self._z_input] = batched_gaussian_samples[:, idx, :]
            outputs.append(self._sess.run(self._outputs, feed_dict))

        samples = np.vstack(outputs)
        return self._config.data_converter.from_tensors(samples), batched_gaussian_samples
