from lcode.simulation import Simulation
from lcode.diagnostics import FXiDiag, FXiType, OutputType, ParticlesDiag
from lcode.diagnostics import SliceDiag, SliceType, SliceValue

# Set some parameters of the config:
config = {
    'geometry': '3d',
    'processing-unit-type': 'cpu',

    'window-width': 5,
    'transverse-step': 0.05,
    'window-length': 15,
    'xi-step': 0.05,
    'time-limit': 200,
    'time-step': 25,
    
    'plasma-particles-per-cell': 9,
    'ion-model': 'background'
}

# Set beam parametrs
beam = {
    'current': 0.1, 
    'particles_in_layer': 200,
    'beam': {'xishape':'c', 'ampl': 1., 'length': 1, 'rshape':'g', 'radius': 1,
             'angshape':'l', 'angspread':1e-5, 'energy':1000, 'eshape':'m',
             'espread':0, 'mass_charge_ratio':1}
}

# Set diagnostics
diag = [FXiDiag(output_type=OutputType.NUMBERS,
            output_period=100, 
            f_xi=FXiType.Ez),
        SliceDiag(
            SliceType.XI_X,
            output_period=100, 
            output_type=OutputType.NUMBERS,
            slice_value = SliceValue.ne),
        ParticlesDiag(
            save_beam=True,
            output_period=25)
       ]

sim = Simulation(config=config, diagnostics=diag,
                 beam_parameters=beam)

sim.step()