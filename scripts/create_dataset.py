import tensorflow as tf

from modules.data.datasets.sources.maestro import MaestroDataset

if __name__ == '__main__':
    tf.get_logger().setLevel('INFO')

    # SOURCE DATASETS
    source_datasets = [MaestroDataset()]
    for dataset in source_datasets:
        dataset.plot()
        dataset.convert()
