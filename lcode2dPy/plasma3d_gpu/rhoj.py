"""Routine to compute charge density and currents of particles."""
from ..config.config import Config
from .data import GPUArrays
from .weights import get_deposit_func

class RhoJComputer(object):
    def __init__(self, config: Config):
        self.grid_step_size = config.getfloat('window-width-step-size')
        self.grid_steps = config.getint('window-width-steps')
        self.plasma_fineness = config.getint('plasma-fineness')
        
        dual_plasma_approach = config.getbool('dual-plasma-approach')
        if dual_plasma_approach:
            self.plasma_coarseness = config.getint('plasma-coarseness')
        else:
            self.plasma_coarseness = None
        
        self.deposit_func = get_deposit_func(
            dual_plasma_approach, self.grid_steps, self.grid_step_size,
            self.plasma_coarseness, self.plasma_fineness)

    def compute_rhoj(self, particles: GPUArrays, const_arrays: GPUArrays):
        ro, jx, jy, jz = self.deposit_func(
            particles.x_init, particles.y_init,
            particles.x_offt, particles.y_offt,
            particles.px, particles.py, particles.pz,
            particles.q, particles.m, const_arrays
        )

        # Also add the background ion charge density.
        # Do it last to preserve more float precision
        ro += const_arrays.ro_initial

        return GPUArrays(ro=ro, jx=jx, jy=jy, jz=jz)
