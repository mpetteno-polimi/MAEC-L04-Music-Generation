from modules.model.modelWrapper import MaecModelAPI
from magenta.models.music_vae import lstm_models
import magenta.models.music_vae.configs as configs
from modules.model.encoders.bidirectional import MyBidirectionalLstmEncoder as BidirectionalEncoder
from modules.model.decoders.hierarchical import MyHierarchicalLstmDecoder as HierarchicalDecoder
from modules.model.vae import MyVAE
from modules import utilities
from records.pianoroll import PianoRollDataset
from sources.maestro import MaestroDataset

if __name__ == '__main__':
    # tf_train_rec_path = os.path.join(Paths.DATA_RECORDS_DIR, "maestro-v3.0.0-train.tfrecord")
    #
    # dataset = tf.data.TFRecordDataset(tf_train_rec_path)
    # for raw_record in dataset.take(1):
    #     print(repr(raw_record))

    """Load model params, load and generates midi sequences."""
    # create and add new configuration
    utilities.add_magenta_config(
        config_name='aaab',
        # model=MyVAE(lstm_models.BidirectionalLstmEncoder(), lstm_models.CategoricalLstmDecoder()),
        model=MyVAE(BidirectionalEncoder(), HierarchicalDecoder(
            lstm_models.CategoricalLstmDecoder(),
            level_lengths=[16, 16],
            disable_autoregression=True)),
        hparams=configs.HParams(
            batch_size=512,
            max_seq_len=256,  # 2 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
            free_bits=0,
            max_beta=0.5,
            beta_rate=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )
    )

    # check configuration validity
    # utilities.check_configuration_section(ConfigSections.MODEL)

    # SOURCE DATASETS
    source_datasets = [MaestroDataset()]

    # PIANO ROLL DATASET
    piano_roll_dataset = PianoRollDataset(source_datasets)
    # piano_roll_dataset.convert()

    # creates Maec model and runs one generation according to config file
    model_api = MaecModelAPI(dataset=piano_roll_dataset,
                             model=MyVAE(BidirectionalEncoder(), HierarchicalDecoder(
                                 lstm_models.CategoricalLstmDecoder(),
                                 level_lengths=[16, 16],
                                 disable_autoregression=True))
                             )
    print("created model")

    model_api.train()
    print("trained")
