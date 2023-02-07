from copy import copy
import numpy as np
import re

from typing import Any
from .template import lcode_template

from .default_config_values import default_config_values


class Config:
    # We don't really need config_values, can work with _arr. TODO: discuss this.
    # config_values: dict[str, str] # And we don't really need to declare this.

    def __init__(self, new_config_values: dict=None, default_config=True):
        # Initialize an empty dictionaty for config values:
        self.config_values = {}

        # This value stores information about whether we use numpy or cupy in
        # our code. Bu now, it is used only by 3d parts of lcodePy.
        self.xp = np

        # If a user wants to set config_values over default ones, then he/she
        # just writes: config = Config() /If a user wants to create a new empty
        # config, then he/she writes: config = Config(default_config=False)
        if default_config:
            self.update(default_config_values)

        # If a user sets new config values when initializing a new config:
        if new_config_values is not None:
            self.update(new_config_values)

    def get(self, option_name: str, fallback: str = '') -> str:
        # TODO: Should we move adjusting from get() to set()?
        self.adjust_config_values_on_get(option_name)

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
            return int(float(self.get(option_name)))
        except ValueError:
            return fallback

    def getfloat(self, option_name: str, fallback: float = 0.0) -> float:
        try:
            return float(self.get(option_name))
        except ValueError:
            return fallback

    def set(self, option_name: str, option_value: Any):
        self.config_values[option_name] = str(option_value)

        # We intercept the setting of a new configuration value
        # in order to adjust other values:
        self.adjust_config_values_on_set(option_name)

    def update(self, new_config_values: dict=None): #, **kconfig_values):
        """
        Works similarly to the update() method of a Python dictionary:
        this method inserts the specified items to the config_values.
        """
        if new_config_values is not None:
            for key in new_config_values:
                self.set(key, new_config_values[key])

        # We can add this part if we want to support **karg
        # for key in kconfig_values:
        #     self.set(key, kconfig_values[key])

    def __copy__(self) -> 'Config':
        """
        Works similarly to the copy() method of a Python dictionary.
        """
        ret = Config()
        ret.config_values = copy(self.config_values)
        return ret

    def c_config(self, path:str = None) -> str:
        cfg = lcode_template.format(**self.config_values)
        if path:
            with open(path, 'w') as cfg_f:
                cfg_f.write(cfg)
        return cfg

    def adjust_config_values_on_get(self, option_name: str):
        """
        Adjusts config values in 3d when we use config.get(...).
        Required for compatibility of 2d and 3d codes.
        """
        if (option_name == 'window-width-steps' and
            self.get('geometry').lower() == '3d'):
            # Goes here every time Config.get('window-width-steps') is
            # called in 3d to adjust window-width and window-width-steps.
            self.adjust_window_width_and_steps_3d()

        if (option_name == 'plasma-fineness' and
            self.get('geometry').lower() == '3d'):
            # Goes here every time Config.get('plasma-fineness') is called
            # in 3d to adjust it according to 'plasma-particles-per-cell'.
            self.adjust_plasma_fineness()

    def adjust_config_values_on_set(self, option_name: str):
        """
        Adjusts config values when we use config.set(...).
        """
        if option_name == 'processing-unit-type':
            # Goes here every time Config['processing-unit-type'] = 'cpu'/'gpu'
            # or something similar is called to change the type of the main
            # array manipulating library.
            pu_type = self.get('processing-unit-type').lower()
            if pu_type == 'cpu':
                self.xp = np
            elif pu_type == 'gpu':
                import cupy as cp
                self.xp = cp

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
        self.set('plasma-particles-per-cell', round(sqrt_per_cell) ** 2)
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

def find(cfg, par):
    ans = re.search('\s' + par + '\s?=\s?[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', cfg)
    return float(ans.group(0).replace(par,'').replace('=', ''))

def find_char(cfg, par):
    ans = re.search(par + '\s?=\s?[a-zA-Z][a-zA-Z]*', cfg)
    return ans.group(0).replace(par,'').replace('=', '').replace(' ','')

def find_beam_profile(cfg):
    ans = re.search('beam-profile\s?=\s?"""([^\>]*)"""', cfg)
    return ans.group(1)