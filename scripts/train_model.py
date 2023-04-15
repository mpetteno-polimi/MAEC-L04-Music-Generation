import tensorflow as tf
from magenta.models.music_vae.configs import CONFIG_MAP
from magenta.models.music_vae.music_vae_train import run, FLAGS

from config import configs
from modules import utilities
from definitions import ConfigSections

config_file = utilities.load_configuration_section(ConfigSections.TRAINER)


def update_flags_from_config_file():
    FLAGS.master = config_file.get("master")
    FLAGS.examples_path = config_file.get("examples_path")
    FLAGS.tfds_name = config_file.get("tfds_name")
    FLAGS.run_dir = config_file.get("run_dir")
    FLAGS.num_steps = config_file.get("num_steps")
    FLAGS.eval_num_batches = config_file.get("eval_num_batches")
    FLAGS.checkpoints_to_keep = config_file.get("checkpoints_to_keep")
    FLAGS.keep_checkpoint_every_n_hours = config_file.get("keep_checkpoint_every_n_hours")
    FLAGS.mode = config_file.get("mode")
    FLAGS.config = config_file.get("config")
    FLAGS.cache_dataset = config_file.get("cache_dataset")
    FLAGS.task = config_file.get("task")
    FLAGS.num_ps_tasks = config_file.get("num_ps_tasks")
    FLAGS.num_sync_workers = config_file.get("num_sync_workers")
    FLAGS.eval_dir_suffix = config_file.get("eval_dir_suffix")
    FLAGS.log = config_file.get("log")


def main(_):
    run(CONFIG_MAP)


if __name__ == "__main__":
    update_flags_from_config_file()
    configs.update_magenta_config_map()
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.logging.set_verbosity(FLAGS.log)
    tf.compat.v1.app.run(main)
