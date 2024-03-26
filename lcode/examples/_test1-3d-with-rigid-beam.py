import numpy as np
import numba

from lcode import Simulation 
from lcode.diagnostics.diagnostics_3d import  DiagnosticsFXi, \
        DiagnosticsColormaps, DiagnosticsTransverse, SaveRunState

# Checks the number of available threads.
print(numba.config.NUMBA_NUM_THREADS) 

# Here you can set the number of threads you want to use.
# Your number must be less than numba.config.NUMBA NUM_THREADS.
numba.set_num_threads(1)

# Sets parameters of test 1:
config = {
    'geometry': '3d',
    'processing-unit-type': 'cpu',
    'window-width-step-size': 0.05,
    'window-width': 16,

    'window-length': 5,
    'xi-step': 0.05,

    'plasma-particles-per-cell': 4,
    
    'rigid-beam': True
}

# Sets the beam charge distribution for test 1.
COMPRESS, BOOST, SIGMA, SHIFT = 1, 1, 1, 0
def rho_b_test1(xp: np, xi, x_grid, y_grid):
    if xi < -2 * xp.sqrt(2 * xp.pi) / COMPRESS or xi > 0:
        return xp.zeros_like(x_grid)
    r = xp.sqrt(x_grid**2 + (y_grid - SHIFT)**2)
    return (.05 * BOOST * xp.exp(-.5 * (r / SIGMA)**2) *
            (1 - xp.cos(xi * COMPRESS * xp.sqrt(xp.pi / 2))))

print('The length of a beam is', 2 * np.sqrt(2 * np.pi) / COMPRESS)

# Sets diagnostics and their parameters:
diag = [DiagnosticsFXi(
            output_period=0, saving_xi_period=10,
            f_xi='Ex,Ey,Ez,Bx,By,Bz,rho,rho_beam,Phi,Sf',
            f_xi_type='both',
            x_probe_lines=np.array([0, 1, 2]), y_probe_lines=[-5, 0, 5]),
        DiagnosticsTransverse(
            output_period=0, saving_xi_period=0.2,
            colormaps='Ex,Ey,Ez,Bx,By,Bz,rho,rho_beam,Phi,px,py,pz',
            colormaps_type='both'),
        DiagnosticsColormaps(
            output_period=0, 
            colormaps='Ex,Ey,Ez,Bx,By,Bz,rho,rho_beam,Phi'),
        SaveRunState(
            output_period=0, saving_xi_period=10, save_plasma=True)]

# and beam parameters:
beam_parameters = rho_b_test1

sim = Simulation(config=config, diagnostics=diag,
                            beam_parameters=beam_parameters)

sim.step()



