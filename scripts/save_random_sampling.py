from definitions import ConfigSections
from modules.utilities import config
import numpy as np
from datetime import datetime

model_config = config.load_configuration_section(ConfigSections.MODEL)
test_config = config.load_configuration_section(ConfigSections.TEST)

num_random_samples = test_config.get('n_test_files')
z_size = model_config.get('z_size')
z = []
for i in range(num_random_samples):
    z.append(np.random.randn(z_size).astype(np.float32))

save_dir_path = test_config.get("z_samples_file_path")
date_time = datetime.now().strftime("%m-%d-%Y_%H-%M")

save_file_path = save_dir_path + date_time + '_Z' + str(z_size) + "_S" + str(num_random_samples) + ".npy"
np.save(save_file_path, z)

z_load = np.load(save_file_path)
print(save_file_path)
