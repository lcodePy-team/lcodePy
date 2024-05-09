"""Routine to compute charge density and currents of particles."""
from ..config.config import Config
from .weights import get_deposit_plasma
from .data import Arrays


def get_rhoj_computer(config: Config):
    deposit = get_deposit_plasma(config)

    def compute_rhoj(particles: Arrays, const_arrays: Arrays):
        ro, jx, jy, jz = deposit(particles, const_arrays)

        # Also add the background ion charge density.
        # Do it last to preserve more float precision
        # TODO: Add background ions that can move.
        ro += const_arrays.ro_initial

        return Arrays(particles.xp, ro=ro, jx=jx, jy=jy, jz=jz)

    return compute_rhoj
