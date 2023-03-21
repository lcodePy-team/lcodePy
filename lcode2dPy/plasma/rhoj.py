"""Routine to compute currents of particles."""
import numpy as np
from numba import njit

from ..config.config import Config
from .data import Arrays
from .gamma import gamma_mass_nb
from .weights import (
    deposit_antisymmetric,
    deposit_symmetric,
    particles_weights,
)


@njit
def _deposit_currents(r_step, n_cells, r, p_r, p_f, p_z, q, m):
    grid_particle_params = particles_weights(r, r_step, n_cells)
    gamma_mass = gamma_mass_nb(m, p_r, p_f, p_z)

    # Particle charge depends on velocity to save continuity equation in QSA
    deposited_charge = q / (gamma_mass - p_z)

    drho = deposited_charge * gamma_mass
    dj_r = deposited_charge * p_r
    dj_f = deposited_charge * p_f
    dj_z = deposited_charge * p_z

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

    return rho, j_r, j_f, j_z


def _cell_volume(r_step, particles_per_cell, n_cells):
    cells_per_particle = 1 / particles_per_cell
    cell_volume = 2 * np.pi * r_step ** 2 * np.arange(n_cells)
    # Volume of boundary cells is corrected to have rho = 0 at the beginning
    cell_volume[0] = np.pi * r_step ** 2
    cell_volume[0] *= (13 + 2 * cells_per_particle ** 2) / 32
    cell_volume[1] *= (193 + 2 * cells_per_particle ** 2) / 192
    cell_volume[-1] = np.pi * r_step ** 2 * (n_cells - 45/32)
    return cell_volume


def get_rhoj_computer(config: Config):
    grid_step_size = config.getfloat('window-width-step-size')
    max_radius = config.getfloat('window-width')
    n_cells = int(max_radius / grid_step_size) + 1
    particles_per_cell = config.getint('plasma-particles-per-cell')
    cell_volume = _cell_volume(grid_step_size, particles_per_cell, n_cells)

    def compute_rhoj(particles):
        rho, j_r, j_f, j_z = _deposit_currents(
            grid_step_size, n_cells, particles.r,
            particles.p_r, particles.p_f, particles.p_z,
            particles.q, particles.m)
        
        rho = rho / cell_volume
        j_r = j_r / cell_volume
        j_f = j_f / cell_volume
        j_z = j_z / cell_volume
        
        # Add charge density of background ions
        rho += 1

        return Arrays(xp=np, rho=rho, j_r=j_r, j_f=j_f, j_z=j_z)
    
    return compute_rhoj
