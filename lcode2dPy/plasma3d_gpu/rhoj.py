"""Routine to compute charge density and currents of particles."""
from lcode2dPy.config.config import Config
from lcode2dPy.plasma3d_gpu.data import GPUArrays
from lcode2dPy.plasma3d_gpu.weights import deposit

class RhoJComputer(object):
    def __init__(self, config: Config):
        self.grid_step_size = config.getfloat('window-width-step-size')
        self.grid_steps = config.getint('window-width-steps')

    def compute_rhoj(self, particles: GPUArrays, const_arrays: GPUArrays):
        ro, jx, jy, jz = deposit(self.grid_steps, self.grid_step_size,
                                 particles.x_init, particles.y_init,
                                 particles.x_offt, particles.y_offt,
                                 particles.px, particles.py,
                                 particles.pz,
                                 particles.q, particles.m)

        # Also add the background ion charge density.
        # Do it last to preserve more float precision
        ro += const_arrays.ro_initial

        return GPUArrays(ro=ro, jx=jx, jy=jy, jz=jz)
