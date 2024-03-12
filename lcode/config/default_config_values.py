"""Default values for a lcodePy config."""

default_config_values = {
    'geometry': 'circ', # circ or 3d now available.

    # Here we set the type of processing unit: CPU or GPU.
    # For now, GPU can be used only for 3d simulations.
    'processing-unit-type': 'cpu',

    # Parameters of simulation window:

    ## Window-width from the main axis to a boundary.
    'window-width': 5.0,
    'window-width-step-size': 0.05,

    ## Here we set a window length along xi axis.
    'window-length': 15.0,
    'xi-step': 0.05,

    ## Set time-limit a bit bigger than the last time moment you want to calculate.
    'time-limit': 200.5,
    'time-step': 25,

    # Parameters of plasma model:

    ## The number of plasma particles per one cell must be the square of a number
    ## in 3d. This parameter will be adjusted if 3d geometry is chosen by finding
    ## the closest square number to plasma-particles-per-cell parameter.
    'plasma-particles-per-cell': 10,
    'ion-model': 'mobile',
    'ion-mass': 1836,

    ## Numerical noise reduction.
    'noise-reductor-enabled': False,
    ### The following parameters are used for 3d noise reduction, 
    ### for 2d they are ignored. 
    'filter-window-length': 3,
    'filter-coefficient': 0.3,
    'damping-coefficient': 0.1,
    'dx-max': 1e-3,
    
    
    ## Longitudinal profile for plasma density. For 3d only. 
    'plasma-zshape': '',


    # Parameters of beam model:
    'beam-substepping-energy': 2,

#This part of the configuration contains experimental and unfinished options. 
#There is no guarantee that it will work or develop in the future. 

    # Partially supported options: 

    # Plasma:
    ## Bz amplitude, supported for 2D only.
    'magnetic-field': 0,
    
    ## Plasma transverse profile settings, 2d only.
    'plasma-profile': 1,
    'plasma-width': 2,
    'plasma-width-2': 1,
    'plasma-density-2': 0.5,
    
    ## Longitudinal substepping, 2d only
    'substepping-depth': 3,
    'substepping-sensitivity': 0.2,

    ## Parameters of the area available for motion of plasma particles, 3d only.
    'reflect-padding-steps': 10,
    'plasma-padding-steps': 15,

    ## Dual plasma approache for 3d, only GPU  
    'dual-plasma-approach': False,
    'plasma-coarseness': 5,
    # Beam:

    ## It works for 3d with an unusual beam configuration.
    'rigid-beam': False, 
    ####
    
    # Options not currently supported:

    ## Plasma:
    'plasma-temperature': 0,
    'trapped-path-limit': 0,

    ## Beam:
    'focusing': 'n',
    'foc-period': 100,
    'foc-strength': 0.1,
    ####

# Developer settings
    'field-solver-subtraction-coefficient': 1,
    'field-solver-variant-A': True,
    'corrector-steps': 2, # Can we even change this???


# Useless legacy parameters.
    'filter-polyorder': 3,
    'plasma-model': 'P',
    'continuation': 'n', 
}