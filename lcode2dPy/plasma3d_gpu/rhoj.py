"""Routine to compute charge density and currents of particles."""
from ..config.config import Config
from .data import GPUArrays
from .weights import get_deposit_plasma

def getRhoJComputer(config: Config):
    grid_step_size = config.getfloat('window-width-step-size')
    grid_steps = config.getint('window-width-steps')
    plasma_fineness = config.getint('plasma-fineness')
    
    dual_plasma_approach = config.getbool('dual-plasma-approach')
    if dual_plasma_approach:
        plasma_coarseness = config.getint('plasma-coarseness')
    else:
        plasma_coarseness = None
    
    deposit = get_deposit_plasma(
        dual_plasma_approach, grid_steps, grid_step_size,
        plasma_coarseness, plasma_fineness)

    def compute_rhoj(particles: GPUArrays, const_arrays: GPUArrays):
        ro, jx, jy, jz = deposit(
            particles.x_init, particles.y_init,
            particles.x_offt, particles.y_offt,
            particles.px, particles.py, particles.pz,
            particles.q, particles.m, const_arrays
        )

        # Also add the background ion charge density.
        # Do it last to preserve more float precision
        ro += const_arrays.ro_initial

        return GPUArrays(ro=ro, jx=jx, jy=jy, jz=jz)
    
    return compute_rhoj
