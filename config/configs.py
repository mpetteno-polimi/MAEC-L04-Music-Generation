from magenta.models.music_vae import MusicVAE, lstm_models
from magenta.models.music_vae.configs import CONFIG_MAP, Config, HParams
from magenta.models.music_vae.data import NoteSequenceAugmenter, PIANO_MIN_MIDI_PITCH, PIANO_MAX_MIDI_PITCH

from modules.data.converters.pianoroll import PianoRollConverter
from modules.model.maec_decoder import MaecDecoder

QUARTER_PER_BARS = 4
STEPS_PER_QUARTER = 4
SLICE_BARS = 16
STEPS_PER_BAR = QUARTER_PER_BARS * STEPS_PER_QUARTER

pr_16bar_converter = PianoRollConverter(
    min_pitch=PIANO_MIN_MIDI_PITCH,
    max_pitch=PIANO_MAX_MIDI_PITCH,
    min_steps_discard=STEPS_PER_BAR,
    max_steps_discard=None,
    max_bars=None,
    slice_bars=SLICE_BARS,
    steps_per_quarter=STEPS_PER_QUARTER,
    quarters_per_bar=QUARTER_PER_BARS,
    pad_to_total_time=True,
    max_tensors_per_notesequence=None,
    presplit_on_time_changes=True
)


def update_magenta_config_map():
    CONFIG_MAP['hierdec-pr_16bar'] = Config(
        model=MusicVAE(
            encoder=lstm_models.BidirectionalLstmEncoder(),
            decoder=MaecDecoder(
                core_decoder=lstm_models.CategoricalLstmDecoder(),
                level_lengths=[16, 16],
                disable_autoregression=True,
                cnn='inceptionv3'
            )
        ),
        hparams=HParams(
            max_seq_len=SLICE_BARS * STEPS_PER_BAR,  # Maximum sequence length. Others will be truncated.
            z_size=512,  # Size of latent vector z.
            free_bits=256,  # Bits to exclude from KL loss per dimension.
            max_beta=0.2,  # Maximum KL cost weight, or cost if not annealing.
            beta_rate=0.0,  # Exponential rate at which to anneal KL cost.
            batch_size=5,  # Minibatch size.
            grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
            clip_mode='global_norm',  # value or global_norm.
            # If clip_mode=global_norm and global_norm is greater than this value,
            # the gradient will be clipped to 0, effectively ignoring the step.
            grad_norm_clip_to_zero=10000,
            learning_rate=0.001,  # Learning rate.
            decay_rate=0.9999,  # Learning rate decay per minibatch.
            min_learning_rate=0.00001,  # Minimum learning rate.
            conditional=True,
            dec_rnn_size=[2048, 2048],  # Decoder RNN: number of units per layer.
            enc_rnn_size=[1024, 1024],  # Encoder RNN: number of units per layer per dir.
            dropout_keep_prob=1.0,  # Probability all dropout keep.
            sampling_schedule='constant',  # constant, exponential, inverse_sigmoid
            sampling_rate=0.0,  # Interpretation is based on `sampling_schedule`.
            use_cudnn=False,  # DEPRECATED
            residual_encoder=False,  # Use residual connections in encoder.
            residual_decoder=False,  # Use residual connections in decoder.
            control_preprocessing_rnn_size=[256],  # Decoder control preprocessing.
        ),
        note_sequence_augmenter=NoteSequenceAugmenter(transpose_range=(-5, 5)),
        data_converter=pr_16bar_converter,
        train_examples_path=None,
        eval_examples_path=None,
    )
