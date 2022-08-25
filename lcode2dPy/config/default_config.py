from .config import Config
from .default_config_values import default_config_values


default_config = Config()

default_config.update(default_config_values)