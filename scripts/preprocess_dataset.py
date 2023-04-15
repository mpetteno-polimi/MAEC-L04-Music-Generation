from absl import app
from magenta.models.music_vae import preprocess_tfrecord
from magenta.models.music_vae.preprocess_tfrecord import FLAGS

from config import configs
from definitions import ConfigSections
from modules import utilities

config_file = utilities.load_configuration_section(ConfigSections.DATA_PRE_PROCESS)


def update_flags_from_config_file():
    FLAGS.input_tfrecord = config_file.get("input_tfrecord")
    FLAGS.output_tfrecord = config_file.get("output_tfrecord")
    FLAGS.output_shards = config_file.get("output_shards")
    FLAGS.config = config_file.get("config")
    FLAGS.enable_filtering = config_file.get("enable_filtering")
    FLAGS.max_total_time = config_file.get("max_total_time")
    FLAGS.max_num_notes = config_file.get("max_num_notes")
    FLAGS.min_velocities = config_file.get("min_velocities")
    FLAGS.min_metric_positions = config_file.get("min_metric_positions")
    FLAGS.is_drum = config_file.get("is_drum")
    FLAGS.drums_only = config_file.get("drums_only")
    FLAGS.pipeline_options = config_file.get("pipeline_options")


if __name__ == '__main__':
    update_flags_from_config_file()
    configs.update_magenta_config_map()
    app.run(preprocess_tfrecord.main)
