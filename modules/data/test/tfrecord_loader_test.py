
import tensorflow as tf

from definitions import Paths
from modules.data.datasets.sources.maestro import MaestroDataset
from modules.data.loaders.tfrecord import TFRecordLoader

if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    tf.get_logger().setLevel('INFO')

    # SOURCE DATASETS
    source_datasets = [MaestroDataset()]

    loader = TFRecordLoader(Paths.DATA_PIANOROLL_RECORDS_DIR, "pianoroll")
    loaded_data = loader.load_train(source_datasets)
    pass
