from math import remainder
from ..config.config import Config
class OutputType:

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
    
def absremainder(x, y):
    """
    This is a universal modulo operator that behaves predictably not only for
    integers, but also for floating point numbers.
    """
    # Due to the nature of floating-number arithmetic, the regular % operator
    # doesn't work as expected for floats. Thus, we use math.remainder that 
    # behaves a little more predictably. Still, while 0.15 % 0.05 = 0.04999...,
    # math.remainder(0.15, 0.05) = -1.3877787807814457e-17. Also, math.remainder
    # isn't exactly the modulo operator like math.fmod and may produce results
    # such as math.remainder(14, 5) = -1.0, but it solves the puzzle of
    # floating-number arithmetic.
    return abs(remainder(x, y))