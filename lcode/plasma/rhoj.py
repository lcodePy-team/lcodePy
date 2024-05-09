"""Routine to compute currents of particles."""
import numpy as np
import numba as nb

from math import sqrt, floor

from ..config.config import Config
from .data import Arrays


@nb.njit
def weight_quadratic(local_coordinate, place):
    """
    Corresponds to the area of the triangular particle in the corresponding cell.
    """
    # TODO: Change to switch statement (match and case) when Python 3.10 is
    #       supported by Anaconda.
    if place == -1:
        return (local_coordinate - 1 / 2) ** 2 / 2
    if place == 0:
        return 3 / 4 - local_coordinate ** 2
    if place == 1:
        return (local_coordinate + 1 / 2) ** 2 / 2

#TODO do deposition in parallel
@nb.njit
def deposit_plasma(grid_step_size, grid_steps, r, p_r, p_f, p_z, q, m):
    """
    Deposit plasma particles onto the charge density and current grids.
    """
    out_rho = np.zeros(grid_steps, dtype=np.float64)
    out_j_r = np.zeros(grid_steps, dtype=np.float64)
    out_j_f = np.zeros(grid_steps, dtype=np.float64)
    out_j_z = np.zeros(grid_steps, dtype=np.float64)

    for k in np.arange(m.size):
        gamma_mass = sqrt(m[k] ** 2 + p_r[k] ** 2 + p_f[k] ** 2 + p_z[k] ** 2)

        # Particle charge depends on velocity to save continuity equation in QSA
        deposited_charge = q[k] / (gamma_mass - p_z[k])
        drho = deposited_charge * gamma_mass
        dj_r = deposited_charge * p_r[k]
        dj_f = deposited_charge * p_f[k]
        dj_z = deposited_charge * p_z[k]

        r_normalized = r[k] / grid_step_size
        particle_cell_index = int(floor(r_normalized + 0.5))
        r_local = r_normalized - particle_cell_index

        first_place = -1
        # A unique case for particles in the central cell
        # that are deposited antisymmetrically
        if particle_cell_index == 0:
            unique_place = -1; cell_index = 1
            weight = weight_quadratic(r_local, unique_place)

            out_rho[cell_index] += drho * weight
            out_j_z[cell_index] += dj_z * weight
            out_j_r[cell_index] -= dj_r * weight
            out_j_f[cell_index] -= dj_f * weight

            first_place = 0

        for place in range(first_place, 2):
            weight = weight_quadratic(r_local, place)
            cell_index = particle_cell_index + place
            
            if particle_cell_index == grid_steps - 1 and place == 1:
                cell_index = grid_steps - 2 # Why?

            out_rho[cell_index] += drho * weight
            out_j_z[cell_index] += dj_z * weight
            out_j_r[cell_index] += dj_r * weight
            out_j_f[cell_index] += dj_f * weight

    # Due to symmetry
    out_j_r[0] = 0
    out_j_f[0] = 0
    # Boundary condition
    out_j_r[grid_steps - 1] = 0
    out_j_f[grid_steps - 1] = 0
    out_j_z[grid_steps - 1] = 0

    return out_rho, out_j_r, out_j_f, out_j_z


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

    def compute_rhoj(particles: Arrays):
        xp = particles.xp

        rho, j_r, j_f, j_z = deposit_plasma(
            grid_step_size, n_cells, particles.r,
            particles.p_r, particles.p_f, particles.p_z,
            particles.q, particles.m)
        
        rho /= cell_volume
        j_r /= cell_volume
        j_f /= cell_volume
        j_z /= cell_volume
        
        # Add charge density of background ions
        rho += 1

        return Arrays(xp=np, rho=rho, j_r=j_r, j_f=j_f, j_z=j_z)
    
    return compute_rhoj
