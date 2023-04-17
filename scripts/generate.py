from modules.model.modelWrapper import MaecModelAPI
from definitions import ConfigSections
from modules import utilities

if __name__ == '__main__':
    """Load model params, load and generates midi sequences."""
    # create and add new configuration

    # check configuration validity
    utilities.check_configuration_section(ConfigSections.MODEL)
    utilities.check_configuration_section(ConfigSections.TRAINING)
    utilities.check_configuration_section(ConfigSections.TESTING)
    utilities.check_configuration_section(ConfigSections.GENERATION)

    # creates Maec model and runs one generation according to config file
    model_api = MaecModelAPI()
    model_api.load_trained_model()
    model_api.generate_midi()
