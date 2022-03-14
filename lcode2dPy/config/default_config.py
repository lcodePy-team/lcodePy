from lcode2dPy.config.config import Config


default_config = Config()

default_config.set('geometry', 'circ') # or 3d or 2d_plane

# Here we set the type of processing unit: CPU or GPU.
# For now, GPU can be used only for 3d simulations.
default_config.set('processing-unit-type', 'cpu')

# Parameters of simulation window:

# window-width has different meanings in 2d and 3d! window-width is the length
# of the square window in 3d, border to border. And window-width is the length
# from the main axis to a border in 2d.
default_config.set('window-width', 5.0)
default_config.set('window-width-step-size', 0.05)

# Here we set a window length along xi axis.
default_config.set('window-length', 15.0)
default_config.set('xi-step', 0.05)

# Set time-limit a bit bigger than the last time moment you want to calculate.
default_config.set('time-limit', 200.5)
default_config.set('time-step', 25)
default_config.set('continuation', 'n') # for 3d - only 'n' is available
# TODO: implement other models for 3d simulation

# Parameters of plasma model:
# The number of plasma particles per one cell must be the square of a number
# in 3d. This parameter will be adjusted if 3d geometry is chosen by finding
# the closest square number to plasma-particles-per-cell parameter.
default_config.set('plasma-particles-per-cell', 10)

# Parameters of beam model:
default_config.set('rigid-beam', 'y') # Only this parameter from this group is used in 3d
default_config.set('beam-substepping-energy', 2)
default_config.set('focusing', 'n')
default_config.set('foc-period', 100)
default_config.set('foc-strength', 0.1)

# Useless parameters (for now):
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
# For more information about these parameters, look up the documentation manual
# of lcode3d.
default_config.set('field-solver-subtraction-trick', 1)
default_config.set('field-solver-variant-A', True)

default_config.set('reflect-padding-steps', 5)
default_config.set('plasma-padding-steps', 10)

# Conflicts:
default_config.set('corrector-steps', 2) # Can we even change this???
