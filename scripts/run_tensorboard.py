
from tensorboard import program

from definitions import ConfigSections, Paths
from modules.utilities import config

tensorboard_config = config.load_configuration_section(ConfigSections.TENSORBOARD)
tracking_address = Paths.LOG_DIR

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print("Tensorflow listening on {}".format(url))
