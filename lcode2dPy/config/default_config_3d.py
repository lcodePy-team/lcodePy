from lcode2dPy.config.config import Config


default_config = Config()

default_config.set('window-xy-steps', 769)
default_config.set('window-xy-step-size', 0.02)

default_config.set('window-length', 3000.0)
default_config.set('xi-step', 0.02)

default_config.set('time_limit', 100.5)
default_config.set('time_step', 25.)
# time_steps = int(time_limit / time_step)

default_config.set('rigid-beam', 'y')
default_config.set('max-radius', 7.)

default_config.set('field-solver-subtraction-trick', 1)
default_config.set('field-solver-variant-A', True)

default_config.set('reflect-padding-steps', 5)
default_config.set('plasma-padding-steps', 10)
default_config.set('plasma-fineness', 2)