import warnings
warnings.filterwarnings('ignore', '.*Grid size.*', )

# Import required modules
from lcode2dPy.simulation import Simulation
from lcode2dPy import DiagnosticsFXi, SaveRunState, DiagnosticsColormaps

# Set some parameters of the config:
config = {
    'geometry': '3d',
    'processing-unit-type': 'cpu',
    'window-width-step-size': 0.05,
    'window-width': 10,

    'window-length': 15,
    'xi-step': 0.05,

    'time-limit': 200,
    'time-step': 25,
    
    'plasma-particles-per-cell': 9,

    'enable-noise-filter': False,

    'beam-substepping-energy': 2
}

# Set beam parameters
from math import pi

beam_parameters = {
    'current': 0.1, 'particles_in_layer': 200,
    'beam': {'xishape':'c', 'ampl': 1., 'length': 2*pi, 'rshape':'g', 'radius': 1,
             'angshape':'l', 'angspread':1e-5, 'energy':1000, 'eshape':'m',
             'espread':0, 'mass_charge_ratio':1}
}

# Set diagnostics
diag = [DiagnosticsFXi(
            output_period=0,
            f_xi='Ez',
            f_xi_type='numbers'),
        DiagnosticsColormaps(output_period=100, colormaps='rho'),
        SaveRunState(output_period=0, save_beam=True)]

sim = Simulation(config=config, diagnostics=diag,
                 beam_parameters=beam_parameters)

sim.step()