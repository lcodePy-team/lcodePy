"""Default values for a lcodePy config."""

default_config_values = {
    'geometry': 'circ', # or 3d or 2d_plane

    # Here we set the type of processing unit: CPU or GPU.
    # For now, GPU can be used only for 3d simulations.
    'processing-unit-type': 'cpu',

    # Parameters of simulation window:

    # window-width has different meanings in 2d and 3d! window-width is the length
    # of the square window in 3d, border to border. And window-width is the length
    # from the main axis to a border in 2d.
    'window-width': 5.0,
    'window-width-step-size': 0.05,

    # Here we set a window length along xi axis.
    'window-length': 15.0,
    'xi-step': 0.05,

    # Set time-limit a bit bigger than the last time moment you want to calculate.
    'time-limit': 200.5,
    'time-step': 25,
    'continuation': 'n', # for 3d - only 'n' is available
    # TODO: implement other models for 3d simulation

    # Parameters of plasma model:
    # The number of plasma particles per one cell must be the square of a number
    # in 3d. This parameter will be adjusted if 3d geometry is chosen by finding
    # the closest square number to plasma-particles-per-cell parameter.
    'plasma-particles-per-cell': 10,

    # Parameters of beam model:
    'rigid-beam': 'n', # Only this parameter from this group is used in 3d
    'beam-substepping-energy': 2,
    'focusing': 'n',
    'foc-period': 100,
    'foc-strength': 0.1,

    # Useless parameters (for now):
    'plasma-model': 'P',
    'magnetic-field': 0,
    'magnetic-field-type': 'c',
    'magnetic-field-period': 200,

    'plasma-temperature': 0,
    'ion-model': 'y',
    'ion-mass': 1836,

    # Parameters of plasma model in 2d simulations:
    'trapped-path-limit': 0,
    'noise-reductor-enabled': False,

    'plasma-profile': 1,
    'plasma-width': 2,
    'plasma-width-2': 1,
    'plasma-density-2': 0.5,

    'substepping-depth': 3,
    'substepping-sensitivity': 0.2,

    # Parameters of plasma model in 3d simulations:
    # TODO: add other parameters to 3d simulations
    # For more information about these parameters, look up the documentation manual
    # of lcode3d.
    'plasma-zshape': '',
    
    'field-solver-subtraction-trick': 1,
    'field-solver-variant-A': True,

    'reflect-padding-steps': 10,
    'plasma-padding-steps': 15,

    'dual-plasma-approach': False,
    'plasma-coarseness': 5,

    # Conflicts:
    'corrector-steps': 2, # Can we even change this???
}
