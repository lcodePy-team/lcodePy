"""Routine to compute currents of particles."""
import numpy as np
from numba import njit

from lcode2dPy.plasma.data import Currents
from lcode2dPy.plasma.gamma import gamma_mass_nb
from lcode2dPy.plasma.weights import (
    deposit_antisymmetric,
    deposit_symmetric,
    particles_weights,
)


@njit
def _deposit_currents(particles, r_step, n_cells):
    grid_particle_params = particles_weights(particles.r, r_step, n_cells)
    gamma_mass = gamma_mass_nb(
        particles.m, particles.p_r, particles.p_f, particles.p_z,
    )
    # Particle charge depends on velocity to save continuity equation in QSA
    deposited_charge = particles.q / (gamma_mass - particles.p_z)

    drho = deposited_charge * gamma_mass
    dj_r = deposited_charge * particles.p_r
    dj_f = deposited_charge * particles.p_f
    dj_z = deposited_charge * particles.p_z

    rho = deposit_symmetric(drho, grid_particle_params)
    j_z = deposit_symmetric(dj_z, grid_particle_params)
    j_r = deposit_antisymmetric(dj_r, grid_particle_params)
    j_f = deposit_antisymmetric(dj_f, grid_particle_params)

    # Due to symmetry
    j_r[0] = 0
    j_f[0] = 0
    # Boundary condition
    j_r[n_cells - 1] = 0
    j_f[n_cells - 1] = 0
    j_z[n_cells - 1] = 0

    return Currents(rho, j_r, j_f, j_z)


def _cell_volume(r_step, particles_per_cell, n_cells):
    cells_per_particle = 1 / particles_per_cell
    cell_volume = 2 * np.pi * r_step ** 2 * np.arange(n_cells)
    # Volume of boundary cells is corrected to have rho = 0 at the beginning
    cell_volume[0] = np.pi * r_step ** 2
    cell_volume[0] *= (13 + 2 * cells_per_particle ** 2) / 32
    cell_volume[1] *= (193 + 2 * cells_per_particle ** 2) / 192
    cell_volume[-1] = np.pi * r_step ** 2 * (n_cells - 45/32)
    return cell_volume


class RhoJComputer(object):
    def __init__(self, config):
        self.r_step = config.getfloat('r-step')
        max_radius = config.getfloat('window-width')
        self.n_cells = int(max_radius / self.r_step) + 1
        particles_per_cell = config.getint('plasma-particles-per-cell')
        self.cell_volume = _cell_volume(
            self.r_step, particles_per_cell, self.n_cells,
        )

    def compute_rhoj(self, particles):
        currents = _deposit_currents(particles, self.r_step, self.n_cells)
        currents = currents.normalize(self.cell_volume)
        # Add charge density of background ions
        currents.rho += np.full_like(currents.rho, 1.0)
        return currents
