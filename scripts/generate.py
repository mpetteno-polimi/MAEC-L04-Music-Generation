from magenta.models.music_vae import MusicVAE, lstm_models, configs
from modules.model.maecModel import MaecModelAPI
from definitions import ConfigSections
from modules.model.vae import MyVAE
from modules import utilities


if __name__ == '__main__':
    """Load model params, load and generates midi sequences."""
    # create and add new configuration
    utilities.add_magenta_config(
        config_name='aaab',
        model=MyVAE(lstm_models.BidirectionalLstmEncoder(), lstm_models.CategoricalLstmDecoder()),
        hparams=configs.HParams(
                batch_size=512,
                max_seq_len=32,  # 2 bars w/ 16 steps per bar
                z_size=512,
                enc_rnn_size=[2048],
                dec_rnn_size=[2048, 2048, 2048],
                free_bits=0,
                max_beta=0.5,
                beta_rate=0.99999,
                sampling_schedule='inverse_sigmoid',
                sampling_rate=1000,
            )
    )

    # check configuration validity
    utilities.check_configuration_section(ConfigSections.MODEL)

    # creates Maec model and runs one generation according to config file
    model_api = MaecModelAPI()
    model_api.load_model()
    model_api.generate()

