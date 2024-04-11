from math import remainder
from ..config.config import Config
class OUTPUT_TYPE:

    __slots__ = ()
    
    NUMBERS = 0x1
    PICTURES = 0x2
    ALL = NUMBERS | PICTURES

def get(array):
    if 'numpy' in str(type(array)):
        return array # numpy
    return array.get() # cupy

class Diagnostic:

    @staticmethod
    def _absremainder(x, y):
        return abs(remainder(x, y))
    
    @staticmethod
    def condition_check(current_time, output_period, time_step_size):
        return Diagnostic._absremainder(current_time, output_period) <= time_step_size / 2

    def pull_config(self, config: Config):
        raise NotImplementedError
    
    def after_step_dxi(self, current_time, xi_plasma_layer,
                       plasma_particles, plasma_fields, plasma_currents,
                       beam_particles, beam_fields, beam_currents):
        raise NotImplementedError
    
    def dump(self, current_time, xi_plasma_layer,
             plasma_particles, plasma_fields, plasma_currents, 
             beam_drain, clean_data=True):
        raise NotImplementedError
    