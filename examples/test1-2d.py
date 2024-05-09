"""
This script run lcode to reproduce Fig. 2 and 3 from
Nuclear Inst. and Methods in Physics Research, A 909 (2018) 446–449
https://doi.org/10.1016/j.nima.2017.12.051
"""

import matplotlib.pyplot as plt
from lcode import Simulation, Config
from lcode.diagnostics.targets import FieldDiagnostics

config = {
    'geometry': '2d',
    'processing-unit-type': 'cpu',
    
    'window-width-step-size': 0.0025,
    'window-width': 16,

    'window-length': 3002.506628274,
    'xi-step': 0.0025,

    'time-limit': 1,
    'time-step': 1,

    'plasma-particles-per-cell': 10,
}

#TODO rigid beam and better beam parametrs
default = {'length' : 5.013256548}
beam_parameters = {'current': 0.05, 'particles_in_layer': 5000, 
                   'default' : default} 


if __name__ == "__main__":
    conf = Config(config)
    # TODO diagnostics and plots
    diags = [] 
    sim = Simulation(config=config, diagnostics=diags,
                             beam_parameters=beam_parameters)
    sim.step()


