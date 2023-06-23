
import numpy as np

from magenta.models.music_vae import TrainedModel

from definitions import ConfigSections
from modules.utilities import sampling, math
from modules.utilities import config as config_file


class MAECTrainedModel(TrainedModel):

    def __init__(self, config, batch_size, checkpoint_dir_or_path=None,
                 var_name_substitutions=None, session_target='', **sample_kwargs):
        super(MAECTrainedModel, self).__init__(config, batch_size,
                                               checkpoint_dir_or_path=checkpoint_dir_or_path,
                                               var_name_substitutions=var_name_substitutions,
                                               session_target=session_target, **sample_kwargs)
        self._config_file = config_file.load_configuration_section(ConfigSections.LATENT_SPACE_SAMPLING)
        self._grid_width = self._config_file.get("grid_width")
        self._rand_seed = self._config_file.get("rand_seed")

    def grid_sample(self, n_grid_points, n_samples_per_grid_point, length=None, temperature=1.0):
        """ TODO

        Args:
          n_grid_points: .
          n_samples_per_grid_point: .
          length: The maximum length of a sample in decoder iterations. Required
            if end tokens are not being used.
          temperature: The softmax temperature to use (if applicable).
        Returns:
          A list of samples as NoteSequence objects.
        Raises:
          ValueError: If `length` is not specified and an end token is not being
            used.
        """

        feed_dict = {
            self._temperature: temperature,
            self._max_length: length
        }

        z_grid = sampling.latin_hypercube_sampling(
            d=self._config.hparams.z_size,
            grid_width=self._grid_width,
            n_grid_points=n_grid_points,
            rand_seed=self._rand_seed
        )

        batched_gaussian_samples = sampling.batch_gaussian_sampling(
            d=self._config.hparams.z_size,
            grid_points=z_grid,
            samples_per_point=n_samples_per_grid_point,
            sigma=math.mean_points_distance(z_grid) / 3,  # TODO: do distance make sense?
            rand_seed=self._rand_seed
        )

        # Decode samples
        outputs = []
        for idx in range(n_grid_points):
            feed_dict[self._z_input] = batched_gaussian_samples[:, idx, :]
            outputs.append(self._sess.run(self._outputs, feed_dict))

        samples = np.vstack(outputs)
        return self._config.data_converter.from_tensors(samples), z_grid, batched_gaussian_samples
