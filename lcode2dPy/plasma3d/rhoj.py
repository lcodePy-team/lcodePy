"""Routine to compute charge density and currents of particles."""
from ..config.config import Config
from .data import Currents, Particles, Const_Arrays
from .weights import deposit

class RhoJComputer(object):
    def __init__(self, config: Config):
        self.grid_step_size = config.getfloat('window-width-step-size')
        self.grid_steps = config.getint('window-width-steps')

    def compute_rhoj(self, particles: Particles, const_arrays: Const_Arrays):
        ro, jx, jy, jz = deposit(self.grid_steps, self.grid_step_size,
                                 particles.x_init, particles.y_init,
                                 particles.x_offt, particles.y_offt,
                                 particles.px, particles.py,
                                 particles.pz,
                                 particles.q, particles.m)
        
        # Also add the background ion charge density.
        # Do it last to preserve more float precision
        ro += const_arrays.ro_initial

        return Currents(ro, jx, jy, jz)
    

