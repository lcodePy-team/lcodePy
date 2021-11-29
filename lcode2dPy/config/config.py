from copy import copy
import numpy as np

from typing import Dict, Any, Optional
from lcode2dPy.config.template import lcode_template

class Config:
    config_values: Dict[str, str]

    def __init__(self):
        self.config_values = {}

    def get(self, option_name: str, fallback: str = '') -> str:
        self.adjust_config_values(option_name)

        if option_name in self.config_values:
            return self.config_values.get(option_name)
        else:
            return fallback

    def getbool(self, option_name: str, fallback: bool = 0) -> bool:
        str_value = self.get(option_name).lower()
        if str_value == 'true':
            return True
        elif str_value == 'false':
            return False
        else:
            return fallback

    def getint(self, option_name: str, fallback: int = 0) -> int:
        try:
            return int(self.get(option_name))
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

    def c_config(self, path:str = None) -> str:
        cfg = lcode_template.format(**self.config_values)
        if path:
            with open(path, 'w') as cfg_f:
                cfg_f.write(cfg)

        return cfg

    def adjust_config_values(self, option_name: str):
        """
        Adjusts config values in 3d.
        Required for compatibility of 2d and 3d codes.
        """
        if (option_name == 'window-width-steps' and
            self.get('geometry') == '3d'):
            # Goes here every time Config.get('window-width-steps') is
            # called in 3d to adjust window-width and window-width-steps.
            self.adjust_window_width_and_steps_3d()

        if (option_name == 'plasma-fineness' and
            self.get('geometry') == '3d'):
            # Goes here every time Config.get('plasma-fineness') is called
            # in 3d to adjust it according to 'plasma-particles-per-cell'.
            self.adjust_plasma_fineness()

    def adjust_window_width_and_steps_3d(self):
        """
        Calculates the optimal number for window-width-steps and uses
        it to adjust window-width and window-width-steps in case of 3d.
        """
        # 0. Calculates an estimation of 'window-width-steps' value.
        estim = int(self.getfloat('window-width') /
                    self.getfloat('window-width-step-size'))

        # 1. Calculates good numbers around the estimation.
        #    Hopefully, the difference is less than 100.
        # TODO: Rewrite so that there are no exceptions.
        lower_bound = (estim - 100) if (estim - 100) > 0 else 0
        good_numbers = np.array([a for a in range(lower_bound, estim + 100)
                                 if good_size(a)])

        # 2. Finds the closest good number to the estimation and
        #    uses it to adjust window-width and window-width-steps.
        optimal_steps = good_numbers[np.abs(good_numbers - estim).argmin()]
        self.set('window-width', (optimal_steps *
                                  self.getfloat('window-width-step-size')))
        self.set('window-width-steps', optimal_steps)
        # TODO: Add a message for the user.

    def adjust_plasma_fineness(self):
        """
        Calculates and adjusts 'plasma-fineness' using
        'plasma-particles-per-cell'.
        """
        sqrt_per_cell = np.sqrt(self.getfloat('plasma-particles-per-cell'))
        self.set('plasma-fineness', round(sqrt_per_cell))
        # TODO: Add a message for the user.


def factorize(number, factors = []):
    """
    Finds all factors of a number.
    """
    if number <= 1:
        return factors
    for i in range(2, number + 1):
        if number % i == 0:
            return factorize(number // i, factors + [i])


def good_size(number):
    """
    Checks if a number is a good number. For more information
    on what good numbers are:
    http://www.fftw.org/doc/Real_002dto_002dReal-Transforms.html
    """
    factors = factorize(number - 1)
    return (all([f in [2, 3, 4, 5, 7, 11, 13] for f in factors]) and
            factors.count(11) + factors.count(13) < 2 and number % 2)
