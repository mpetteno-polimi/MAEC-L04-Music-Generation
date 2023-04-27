# TODO - DOC

from keras import optimizers, callbacks
from keras import backend as K

from definitions import ConfigSections, Paths
from modules import utilities


class Trainer(object):

    def __init__(self, model):
        self._training_config = utilities.config.load_configuration_section(ConfigSections.TRAINING)
        self._model = model
        self.optimizer = optimizers.Adam(
            learning_rate=self._training_config.get("learning_rate"),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            weight_decay=None,
            clipnorm=self._training_config.get("clip_norm"),
            clipvalue=self._training_config.get("clip_value"),
            global_clipnorm=self._training_config.get("global_clip_norm"),
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=self._training_config.get("jit_compile"),
            name="Adam"
        )

    def train(self, train_data, validation_data):
        self._compile_model()
        self._fit_model(train_data, validation_data)

    def _compile_model(self):
        self._model.compile(
            optimizer=self.optimizer,
            loss=self._model.loss_fn(),
            metrics=['accuracy'],
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=self._training_config.get("run_eagerly"),
            steps_per_execution=self._training_config.get("steps_per_execution"),
            jit_compile=self._training_config.get("jit_compile")
        )

    def _fit_model(self, train_data, validation_data):
        history = self._model.fit(
            x=train_data,
            y=train_data,
            batch_size=self._training_config.get("batch_size"),
            epochs=self._training_config.get("epochs"),
            verbose=self._training_config.get("verbose"),
            callbacks=self._get_callbacks(),
            validation_split=0.0,
            validation_data=(validation_data, validation_data),
            shuffle=self._training_config.get("shuffle"),
            class_weight=None,
            sample_weight=None,
            initial_epoch=self._training_config.get("initial_epoch"),
            steps_per_epoch=self._training_config.get("steps_per_epoch"),
            validation_steps=self._training_config.get("validation_steps"),
            validation_batch_size=self._training_config.get("validation_batch_size"),
            validation_freq=self._training_config.get("validation_freq"),
            max_queue_size=self._training_config.get("max_queue_size"),
            workers=self._training_config.get("workers"),
            use_multiprocessing=self._training_config.get("use_multiprocessing")
        )

        return history

    def _get_callbacks(self):

        def learning_rate_scheduler(epoch, lr):
            decay_rate = self._training_config.get("decay_rate")
            min_learning_rate = self._training_config.get("min_learning_rate")
            return (lr - min_learning_rate) * K.pow(decay_rate, epoch) + min_learning_rate

        training_callbacks = []

        checkpoint_config = utilities.config.load_configuration_section(ConfigSections.CHECKPOINTS)
        if checkpoint_config.get("enabled"):
            checkpoint_cb = callbacks.ModelCheckpoint(
                filepath=Paths.TRAIN_CHECK_DIR / checkpoint_config.get("checkpoint_filename"),
                monitor=checkpoint_config.get("monitor_metric"),
                verbose=checkpoint_config.get("verbose"),
                save_best_only=checkpoint_config.get("save_best_only"),
                save_weights_only=checkpoint_config.get("save_weights_only"),
                mode=checkpoint_config.get("mode"),
                save_freq=checkpoint_config.get("save_freq"),
                options=None,
                initial_value_threshold=checkpoint_config.get("initial_value_threshold")
            )
            training_callbacks.append(checkpoint_cb)

        backup_config = utilities.config.load_configuration_section(ConfigSections.BACKUP)
        if backup_config.get("enabled"):
            backup_cb = callbacks.BackupAndRestore(
                backup_dir=Paths.TRAIN_BACKUP_DIR,
                save_freq=backup_config.get("save_freq"),
                delete_checkpoint=backup_config.get("delete_checkpoint"),
                save_before_preemption=backup_config.get("save_before_preemption")
            )
            training_callbacks.append(backup_cb)

        tensorboard_config = utilities.config.load_configuration_section(ConfigSections.TENSORBOARD)
        if tensorboard_config.get("enabled"):
            tensorboard_cb = callbacks.TensorBoard(
                log_dir=Paths.TRAIN_LOG_DIR,
                histogram_freq=tensorboard_config.get("histogram_freq"),
                write_graph=tensorboard_config.get("write_graph"),
                write_images=tensorboard_config.get("write_images"),
                write_steps_per_second=tensorboard_config.get("write_steps_per_second"),
                update_freq=tensorboard_config.get("update_freq"),
                profile_batch=tensorboard_config.get("profile_batch"),
                embeddings_freq=tensorboard_config.get("embeddings_freq"),
                embeddings_metadata=None
            )
            training_callbacks.append(tensorboard_cb)

        early_stopping_config = utilities.config.load_configuration_section(ConfigSections.EARLY_STOPPING)
        if early_stopping_config.get("enabled"):
            early_stopping_cb = callbacks.EarlyStopping(
                monitor=early_stopping_config.get("monitor"),
                min_delta=early_stopping_config.get("min_delta"),
                patience=early_stopping_config.get("patience"),
                verbose=early_stopping_config.get("verbose"),
                mode=early_stopping_config.get("mode"),
                baseline=early_stopping_config.get("baseline"),
                restore_best_weights=early_stopping_config.get("restore_best_weights"),
                start_from_epoch=early_stopping_config.get("start_from_epoch")
            )
            training_callbacks.append(early_stopping_cb)

        lr_schedule_cb = callbacks.LearningRateScheduler(
            schedule=learning_rate_scheduler,
            verbose=1
        )
        training_callbacks.append(lr_schedule_cb)

        return training_callbacks
