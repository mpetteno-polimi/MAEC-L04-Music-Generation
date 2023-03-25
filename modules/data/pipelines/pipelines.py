""" TODO - Module DOC """

from magenta.models.shared.events_rnn_model import EventSequenceRnnConfig
from magenta.pipelines import note_sequence_pipelines, dag_pipeline, pipeline, event_sequence_pipeline
from magenta.pipelines.pianoroll_pipeline import PianorollSequenceExtractor
from note_seq import PianorollSequence
from note_seq.protobuf import music_pb2


def pianoroll_pipeline(config: EventSequenceRnnConfig, min_steps: int, max_steps: int, out_filename: str) \
        -> pipeline.Pipeline:
    """Returns the Pipeline instance which creates the RNN dataset.

    Args:
      config: An EventSequenceRnnConfig.
      min_steps: Minimum number of steps for an extracted sequence.
      max_steps: Maximum number of steps for an extracted sequence.
      out_filename: Name for the DagOutput pipeline

    Returns:
      A pipeline.Pipeline instance.
    """
    # Transpose up to a major third in either direction.
    transposition_range = list(range(-4, 5))

    time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(name='TimeChangeSplitter')

    dag = {time_change_splitter: dag_pipeline.DagInput(music_pb2.NoteSequence)}

    quantizer = note_sequence_pipelines.Quantizer(steps_per_quarter=config.steps_per_quarter, name='Quantizer')
    transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(transposition_range,
                                                                           name='TranspositionPipeline')
    pianoroll_extractor = PianorollSequenceExtractor(min_steps=min_steps, max_steps=max_steps,
                                                     name='PianorollExtractor')
    encoder_pipeline = event_sequence_pipeline.EncoderPipeline(PianorollSequence, config.encoder_decoder,
                                                               name='EncoderPipeline')

    dag[quantizer] = time_change_splitter
    dag[transposition_pipeline] = quantizer
    dag[pianoroll_extractor] = transposition_pipeline
    dag[encoder_pipeline] = pianoroll_extractor
    dag[dag_pipeline.DagOutput(out_filename)] = encoder_pipeline

    return dag_pipeline.DAGPipeline(dag)
