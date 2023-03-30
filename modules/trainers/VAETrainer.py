import os

import magenta.models.music_vae.configs as configs
import tensorflow._api.v2.compat.v1 as tf
import tf_slim

from trainer import Trainer


class VAETrainer(Trainer):
    # TODO: Class DOC

    def __init__(self, dataset, model, config_file):
        super(VAETrainer, self).__init__(dataset, model)
        """
        Trainer class constructor

        :param dataset:
        :param model:
        :param config_file:
        """

    def train(self):
        run_dir = os.path.expanduser(self._train_config.get('model_checkpoint_file_dir'))
        train_dir = os.path.join(run_dir, 'train')
        configuration = self.config_map[self._model_config.get('config_map_name')]
        checkpoints_max_num = self._train_config.get('checkpoints_max_num')
        h_between_checkpoints = self._train_config.get('h_between_checkpoints')
        num_steps = self._train_config.get('num_steps')

        self._train_loop(
            train_dir=train_dir,
            config=configuration,
            checkpoints_to_keep=checkpoints_max_num,
            keep_checkpoint_every_n_hours=h_between_checkpoints,
            num_steps=num_steps,
            # todo: master, num_sync_workers, num_ps_tasks, task
        )

    def _train_loop(self,
                    train_dir,
                    config,
                    checkpoints_to_keep=5,
                    keep_checkpoint_every_n_hours=1,
                    num_steps=None,
                    master='',
                    num_sync_workers=0,
                    num_ps_tasks=0,
                    task=0):
        """Train loop."""

        tf.gfile.MakeDirs(train_dir)
        is_chief = (task == 0)
        if is_chief:
            self._trial_summary(
                config.hparams, config.train_examples_path or config.tfds_name,
                train_dir)
        with tf.Graph().as_default():
            with tf.device(tf.train.replica_device_setter(
                num_ps_tasks, merge_devices=True)):

                model = config.model
                model.build(config.hparams,
                            config.data_converter.output_depth,
                            is_training=True)

                optimizer = model.train(self._get_input_tensors(self.dataset, config))

                hooks = []
                if num_sync_workers:
                    optimizer = tf.train.SyncReplicasOptimizer(
                        optimizer,
                        num_sync_workers)
                    hooks.append(optimizer.make_session_run_hook(is_chief))

                grads, var_list = list(zip(*optimizer.compute_gradients(model.loss)))
                global_norm = tf.global_norm(grads)
                tf.summary.scalar('global_norm', global_norm)

                if config.hparams.clip_mode == 'value':
                    g = config.hparams.grad_clip
                    clipped_grads = [tf.clip_by_value(grad, -g, g) for grad in grads]
                elif config.hparams.clip_mode == 'global_norm':
                    clipped_grads = tf.cond(
                        global_norm < config.hparams.grad_norm_clip_to_zero,
                        lambda: tf.clip_by_global_norm(  # pylint:disable=g-long-lambda
                            grads, config.hparams.grad_clip, use_norm=global_norm)[0],
                        lambda: [tf.zeros(tf.shape(g)) for g in grads])
                else:
                    raise ValueError(
                        'Unknown clip_mode: {}'.format(config.hparams.clip_mode))
                train_op = optimizer.apply_gradients(
                    list(zip(clipped_grads, var_list)),
                    global_step=model.global_step,
                    name='train_step')

                logging_dict = {'global_step': model.global_step,
                                'loss': model.loss}

                hooks.append(tf.train.LoggingTensorHook(logging_dict, every_n_iter=100))
                if num_steps:
                    hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

                scaffold = tf.train.Scaffold(
                    saver=tf.train.Saver(
                        max_to_keep=checkpoints_to_keep,
                        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))
                tf_slim.training.train(
                    train_op=train_op,
                    logdir=train_dir,
                    scaffold=scaffold,
                    hooks=hooks,
                    save_checkpoint_secs=60,
                    master=master,
                    is_chief=is_chief)
