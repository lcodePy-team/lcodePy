import warnings
warnings.filterwarnings('ignore', '.*Grid size.*', )

# Import required modules
from lcode.simulation import Simulation
from lcode.diagnostics.diagnostics_3d import DiagnosticsFXi, SaveRunState, \
                                             DiagnosticsColormaps

# Set some parameters of the config:
config = {
    'geometry': '3d',
    'processing-unit-type': 'cpu',
    'transverse-step': 0.02,
    'window-width': 2.5,

    'window-length': 160,
    'xi-step': 0.02,

    'time-limit': 10000,
    'time-step': 50,
    
    'plasma-zshape': 
        """
        200 1.0 L 1.0
        500 1.0 L 1.085
        1000000 1.085 L 1.085
        """,

    'plasma-particles-per-cell': 1,
    'ino-model' : 'background',

    'declustering-averaging': 11,
    'declustering-force': 0.003,
    'damping-declustering': 0.001,
    'declustering-limit': 1e-3,

    'beam-substepping-energy': 1200
}


beam_parameters = {
    'current': 0.0005, 'particles_in_layer': 2000,
    'beam': {'xishape':'l', 'ampl': 1., 'length':160, 'rshape':'g', 'radius':0.5,
             'angshape':'l', 'angspread':2e-4, 'energy':1000, 'eshape':'m',
             'espread':0, 'mass_charge_ratio':1}
}

# Set diagnostics
diag = [DiagnosticsFXi(
            output_period=0, saving_xi_period=10,
            f_xi='Ez,Phi',
            f_xi_type='numbers',
            x_probe_lines=[0, 2.5], y_probe_lines=[0, 0]),
        DiagnosticsColormaps(output_period=100, colormaps='ne,ni'),
        SaveRunState(output_period=100, save_beam=True)]

sim = Simulation(config=config, diagnostics=diag,
                 beam_parameters=beam_parameters)

sim.step()
