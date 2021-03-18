from lcode2dPy.config.config import Config


default_config = Config()

# 3d extra pars
default_config.set('window-xy-steps', 769)
default_config.set('window-xy-step-size', 0.02)
default_config.set('max-radius', 7.)

default_config.set('field-solver-subtraction-trick', 1)
default_config.set('field-solver-variant-A', True)

default_config.set('reflect-padding-steps', 5)
default_config.set('plasma-padding-steps', 10)
default_config.set('plasma-fineness', 2)

# 2d pars
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
