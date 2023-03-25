import tensorflow as tf

from modules.data.datasets.records.pianoroll import PianoRollDataset
from modules.data.datasets.sources.maestro import MaestroDataset

if __name__ == '__main__':
    tf.get_logger().setLevel('INFO')

    # SOURCE DATASETS
    source_datasets = [MaestroDataset()]

    # PIANO ROLL DATASET
    piano_roll_dataset = PianoRollDataset(source_datasets)
    piano_roll_dataset.convert()
