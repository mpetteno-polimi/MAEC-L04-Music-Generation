from definitions import ConfigSections
from modules import utilities
from modules.model.maecModel import MaecModel

if __name__ == '__main__':
    """Load model params, load TrainedModel and generates midi sequences."""

    utilities.check_configuration_section(ConfigSections.MODEL)
    model = MaecModel()
    model.load_model()
    model.generate()

