"""Routine to compute charge density and currents of particles."""
import numpy as np
import numba as nb

from lcode2dPy.plasma3d.data import Currents
from lcode2dPy.plasma3d.weights import deposit

class RhoJComputer(object):
    def __init__(self, config):
        self.grid_step_size = config.getfloat('window-xy-step-size')
        self.grid_steps = config.getint('window-xy-steps')

    def compute_rhoj(self, particles, const_arrays):
        rho, j_x, j_y, j_z = deposit(self.grid_steps, self.grid_step_size,
                                 particles.x_init, particles.x_init,
                                 particles.x_offt, particles.x_offt,
                                 particles.px, particles.py,
                                 particles.pz,
                                 particles.q, particles.m)
        
        # Also add the background ion charge density.
        # ro += ro_initial  # Do it last to preserve more float precision
        rho += const_arrays.ro_initial
        return Currents(rho, j_x, j_y, j_z)
    

