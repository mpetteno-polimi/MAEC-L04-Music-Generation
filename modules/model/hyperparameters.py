from enum import Enum


class Hyperparameters(Enum):
    BATCH_SIZE: None

    ENCODER: None
    DECODER: None

    ENCODER_IN_DEPTH: None
    ENCODER_OUT_DEPTH: None
    ENCODER_ACTIVATION: None
    ENCODER_LOSS: None
    ENCODER_DISTRIBUTION: None

    DECODER_IN_DEPTH: None
    DECODER_OUT_DEPTH: None
    DECODER_ACTIVATION: None
