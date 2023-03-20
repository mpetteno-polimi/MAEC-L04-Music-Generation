from model.decoder.abstract_decoder import VariationalDecoder
import logging as log


class LstmDecoder(VariationalDecoder):
    def build(self, hparams):
        log.debug("Building LstmDecoder... ")

        log.debug("Done")

    def decode(self, sample, hparams):
        log.debug("Computing LstmDecoder Reconstruction Loss... ")

        log.debug("Done")

    def reconstruction_loss(self, x_input, x_target):
        log.debug("Computing LstmDecoder Reconstruction Loss... ")

        log.debug("Done")
