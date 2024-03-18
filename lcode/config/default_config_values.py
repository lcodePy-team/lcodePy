"""Default values for a lcodePy config"""

default_config_values = {
    'geometry': 'circ', # 'circ' or '3d' 

    # Here we set the type of processing unit: 'cpu' or 'gpu'.
    # For now, GPU can be used only for 3d simulations.
    'processing-unit-type': 'cpu',

    # Parameters of simulation window:

    ## Transverse size of the window from axis to boundary and transverse grid step (dr in 2d or 
    ## dx(=dy) in 3d). The window width is adjusted to the closest "good" value
    ## to ensure fft performance. 
    'window-width': 5.0,
    'transverse-step': 0.05,

    ## Window length along xi-axis and grid step in dxi.
    'window-length': 15.0,
    'xi-step': 0.05,

    ## Time limit for beam evolution and time step.
    'time-limit': 200.5,
    'time-step': 25,

    # Parameters of plasma model:

    ## The number of plasma particles per one cell must be the square of 
    ## an integer in 3d. This parameter will be adjusted in 3d geometry
    ## to the nearest integer squared (to 9 by default).
    'plasma-particles-per-cell': 10,
    'ion-model': 'mobile', # 'mobile' or 'background'

	## Mass of plasma ions in unita of electron mass
    'ion-mass': 1836,

    ## Declustering of plasma electrons
    'declustering-enabled': False,
    ### The following parameters are used for 3d declustering, 
    ### In 2d they are ignored. 
    'declustering-averaging': 3,
    'declustering-force': 0.3,
    'declustering-damping': 0.1,
    'declustering-limit': 1e-3,
    
    
    ## Longitudinal profile of the plasma density. For 3d only. 
    ## example https://lcode.info/site-files/manual.pdf p. 17
    'plasma-zshape': '''
        ''',

    # Parameters of beam model:
    'beam-substepping-energy': 2,

#This part of the configuration contains experimental and unfinished options. 
#There is no guarantee that it will work or develop in the future. 

    # Partially supported options: 

    # Plasma:
    ## External Bz amplitude, 2D only.
    'magnetic-field': 0,
    
    ## Plasma transverse profile settings, 2d only.
    'plasma-profile': 1,
    'plasma-width': 2,
    'plasma-width-2': 1,
    'plasma-density-2': 0.5,
    
    ## Plasma substepping for xi, 2d only
    'substepping-depth': 3,
    'substepping-sensitivity': 0.2,

    ## Parameters of the area available for motion of plasma particles, 3d only.
    'reflect-padding-steps': 10,
    'plasma-padding-steps': 15,

    ## Dual plasma approache for 3d, only GPU  
    'dual-plasma-approach': False,
    'plasma-coarseness': 5,
    
    # Beam:
    ## Works for 3d with an unusual beam configuration.
    'rigid-beam': False, 
    ####

# Developer settings
    
    ## Only 3d, default value is 1+dxi
    'field-solver-subtraction-coefficient': 1, 
    
    ## Only 2d
    ## Plasma:
    'trapped-path-limit': 0,
    'correctotransverse-steps': 2, # Can we even change this???
}