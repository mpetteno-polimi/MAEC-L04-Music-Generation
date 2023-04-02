import os
import magenta.models.music_vae.configs as configs
import tensorflow._api.v2.compat.v1 as tf
import tf_slim
from tester import Tester
from trainers.trainer import Trainer
from modules.utilities import trial_summary, get_input_tensors


class VAETester(Tester):
    # TODO: Class DOC

    def __init__(self, dataset, model):
        super(VAETester, self).__init__(dataset, model)
        """
        Trainer class constructor

        :param dataset:
        :param model:
        :param config_file:
        """

    def test(self):
        run_dir = os.path.expanduser(self._train_config.get('model_checkpoint_file_dir'))
        train_dir = os.path.join(run_dir, 'train')
        eval_dir = os.path.join(run_dir, 'eval')
        configuration = self.config_map[self._model_config.get('config_map_name')]
        num_batches = self._test_config.get('num_batches')

        self._test_loop(
            train_dir=train_dir,
            eval_dir=eval_dir,
            config=configuration,
            num_batches=num_batches,
        )

    def _test_loop(self,
                   train_dir,
                   eval_dir,
                   config,
                   num_batches,
                   master=''):
        """Evaluate the model repeatedly."""
        tf.gfile.MakeDirs(eval_dir)

        trial_summary(
            config.hparams, config.eval_examples_path or config.tfds_name, eval_dir)
        with tf.Graph().as_default():
            model = config.model
            model.build(config.hparams,
                        config.data_converter.output_depth,
                        is_training=False)

            eval_op = model.eval(
                **get_input_tensors(num_batches, config))

            hooks = [
                tf_slim.evaluation.StopAfterNEvalsHook(num_batches),
                tf_slim.evaluation.SummaryAtEndHook(eval_dir)
            ]
            tf_slim.evaluation.evaluate_repeatedly(
                train_dir,
                eval_ops=eval_op,
                hooks=hooks,
                eval_interval_secs=60,
                master=master)


