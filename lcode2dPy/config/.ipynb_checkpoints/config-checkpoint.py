from copy import copy
from typing import Dict, Any, Optional


class Config:
    config_values: Dict[str, str]

    def __init__(self):
        self.config_values = {}

    def get(self, option_name: str, fallback: str = '') -> str:
        if option_name in self.config_values:
            return self.config_values.get(option_name)
        return fallback

    def getbool(self, option_name: str, fallback: bool = 0) -> bool:
        str_value = self.config_values.get(option_name).lower()
        if str_value == 'true':
            return True
        elif str_value == 'false':
            return False
        else:
            return fallback

    def getint(self, option_name: str, fallback: int = 0) -> int:
        try:
            return int(self.config_values.get(option_name))
        except ValueError:
            return fallback

    def getfloat(self, option_name: str, fallback: float = 0.0) -> float:
        try:
            return float(self.get(option_name))
        except ValueError:
            return fallback

    def set(self, option_name: str, option_value: Any):
        self.config_values[option_name] = str(option_value)

    def __copy__(self) -> 'Config':
        ret = Config()
        ret.config_values = copy(self.config_values)
        return ret


default_config = Config()

default_config.set('geometry', 'c')
default_config.set('window-width', 5.0)
default_config.set('window-length', 15.0)
default_config.set('r-step', 0.05)
default_config.set('xi-step', 0.05)
default_config.set('time-limit', 200.5)
default_config.set('time-step', 25)
default_config.set('continuation', 'n')

default_config.set('rigid-beam', 'y')
default_config.set('beam-substepping-energy', 2)
default_config.set('focusing', 'n')
default_config.set('foc-period', 100)
default_config.set('foc-strength', 0.1)

default_config.set('plasma-model', 'P')
default_config.set('magnetic-field', 0)
default_config.set('magnetic-field-type', 'c')
default_config.set('magnetic-field-period', 200)
default_config.set('plasma-zshape', '')

default_config.set('plasma-particles-per-cell', 10)
default_config.set('plasma-profile', '1')
default_config.set('plasma-width', 2)
default_config.set('plasma-width-2', 1)
default_config.set('plasma-density-2', 0.5)
default_config.set('plasma-temperature', 0)
default_config.set('ion-model', 'y')
default_config.set('ion-mass', 1836)
default_config.set('substepping-depth', 3)
default_config.set('substepping-sensitivity', 0.2)
default_config.set('trapped-path-limit', 0)
default_config.set('noise-reductor-enabled', False)
default_config.set('corrector-steps', 2)