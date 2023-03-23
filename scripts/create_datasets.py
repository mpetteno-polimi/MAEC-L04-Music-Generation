import tensorflow as tf

from maestro import MaestroDataset

if __name__ == '__main__':
    tf.get_logger().setLevel('INFO')

    # MAESTRO DATASET
    maestro_dataset = MaestroDataset()
    maestro_dataset.download()
    maestro_dataset.convert()

