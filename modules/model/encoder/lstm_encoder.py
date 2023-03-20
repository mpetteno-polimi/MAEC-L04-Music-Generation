from model.encoder.abstract_encoder import Encoder
import logging as log


class LstmEncoder(Encoder):
    def build(self, hparams):
        log.debug("Building LstmEncoder... ")

        log.debug("Done")

    def encode(self, sequence, sequence_length):
        log.debug("Encoding...")

        log.debug("Done")
