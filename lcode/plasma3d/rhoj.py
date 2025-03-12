"""Routine to compute charge density and currents of particles."""
from ..config.config import Config
from .weights import get_deposit_plasma
from .data import Arrays


def get_rhoj_computer(config: Config):
    deposit = get_deposit_plasma(config)
    ion_model = config.get("ion-model")

    def compute_rhoj(particles: dict, const_arrays: Arrays):
        ro, jx, jy, jz = deposit(particles, const_arrays)

        # Also add the background ion charge density.
        # Do it last to preserve more float precision
        # TODO: Add background ions that can move.
        if ion_model == "background":
            ro[1,:,:] = const_arrays.ro_initial

        return Arrays(const_arrays.xp, ro=ro, jx=jx, jy=jy, jz=jz)

    return compute_rhoj
