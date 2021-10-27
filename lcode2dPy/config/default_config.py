from lcode2dPy.config.config import Config


default_config = Config()

default_config.set('geometry', 'circ') # or 3d or 2d_plane

# Parameters of simulation window:
default_config.set('window-width', 5.0) # Has different meanings in 2d and 3d!
default_config.set('r-step', 5.0) 
default_config.set('window-width-step-size', 0.05)
default_config.set('window-width-steps', 101) # Only needed in 3d
# TODO: get rid of this parameter in 3d

default_config.set('window-length', 15.0)
default_config.set('xi-step', 0.05)
# time_steps = int(time_limit / time_step)

default_config.set('time-limit', 200.5)
default_config.set('time-step', 25)
default_config.set('continuation', 'n') # for 3d - only 'n' is available
# TODO: implement other models for 3d simulation

# Parameters of beam model:
default_config.set('rigid-beam', 'y') # Only this parameter from this group is used in 3d
default_config.set('beam-substepping-energy', 2)
default_config.set('focusing', 'n')
default_config.set('foc-period', 100)
default_config.set('foc-strength', 0.1)

# Useless parameters:
default_config.set('plasma-model', 'P')
default_config.set('magnetic-field', 0)
default_config.set('magnetic-field-type', 'c')
default_config.set('magnetic-field-period', 200)
default_config.set('plasma-zshape', '')

default_config.set('plasma-temperature', 0)
default_config.set('ion-model', 'y')
default_config.set('ion-mass', 1836)

# Parameters of plasma model in 2d simulations:
default_config.set('trapped-path-limit', 0)
default_config.set('noise-reductor-enabled', False)

default_config.set('plasma-profile', '1')
default_config.set('plasma-width', 2)
default_config.set('plasma-width-2', 1)
default_config.set('plasma-density-2', 0.5)

default_config.set('substepping-depth', 3)
default_config.set('substepping-sensitivity', 0.2)

# Parameters of plasma model in 3d simulations:
# TODO: add other parameters to 3d simulations
default_config.set('field-solver-subtraction-trick', 1)
default_config.set('field-solver-variant-A', True)

default_config.set('reflect-padding-steps', 5)
default_config.set('plasma-padding-steps', 10)

# Conflicts:
default_config.set('plasma-particles-per-cell', 10)
default_config.set('plasma-fineness', 2) # Similar to 'plasma-particles-per-cell'

default_config.set('corrector-steps', 2) # Can we even change this???
