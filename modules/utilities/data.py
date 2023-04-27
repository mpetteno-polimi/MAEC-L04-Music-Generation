# TODO - DOC

import hashlib
from pathlib import Path


def get_tfrecords_path_for_source_datasets(source_datasets, input_path: Path, mode_label: str, collection_name: str) \
        -> [str]:
    """ TODO - Function DOC """

    tfrecord_file_patterns = map(lambda source_dataset: source_dataset.name + "-{}_{}.tfrecord", source_datasets)
    tfrecord_paths = [list(input_path.glob(e.format(mode_label, collection_name))) for e in tfrecord_file_patterns]
    tfrecord_paths = [j for i in tfrecord_paths for j in i]
    tfrecord_paths = [str(path) for path in tfrecord_paths]
    return tfrecord_paths


def generate_note_sequence_id(filename, collection_name, source_type):
    """Generates a unique ID for a sequence.
    The format is:'/id/<type>/<collection name>/<hash>'.
    Args:
      filename: The string path to the source file relative to the root of the
          collection.
      collection_name: The collection from which the file comes.
      source_type: The source type as a string (e.g. "midi" or "abc").
    Returns:
      The generated sequence ID as a string.
    """
    filename_fingerprint = hashlib.sha1(filename.encode('utf-8'))
    return '/id/%s/%s/%s' % (source_type.lower(), collection_name, filename_fingerprint.hexdigest())
