import numba as nb
import numpy as np


@nb.njit(parallel=True)
def particles_weights(transverse_coord, transverse_step, n_cells):
    transverse_coord_normalized = transverse_coord / transverse_step
    # j_1, j_2, j_3 - indexes of three cells to deposit into/interpolate from
    j_2 = np.floor(transverse_coord_normalized + 0.5).astype(np.int64)
    j_1 = j_2 - 1
    j_1[j_2 == 0] = 1
    j_3 = j_2 + 1
    j_3[j_2 == n_cells - 1] = n_cells - 2

    transverse_coord_local = transverse_coord_normalized - j_2

    # a_1, a_2, a_3 - deposition/interpolation coefficients
    # (area of the triangular particle in the corresponding cell)
    a_1 = 0.5 * (transverse_coord_local - 0.5) ** 2
    a_2 = 0.75 - transverse_coord_local ** 2
    a_3 = 0.5 * (transverse_coord_local + 0.5) ** 2

    # This coefficient marks particles
    # in central cell for antisymmetric deposition
    sgn = np.full_like(j_2, 1.0)
    sgn[j_2 == 0] = -1.0
    return n_cells, j_1, j_2, j_3, sgn, a_1, a_2, a_3


@nb.njit(cache=True)
def _deposit(deposited_value, grid_particle_parameters, symmetric):
    n_cells, j_1, j_2, j_3, sgn, a_1, a_2, a_3 = grid_particle_parameters
    out = np.zeros(n_cells, dtype=np.float64)
    if symmetric:
        _add_at_numba(out, j_1, a_1 * deposited_value)
    else:
        _add_at_numba(out, j_1, a_1 * deposited_value * sgn)
    _add_at_numba(out, j_2, a_2 * deposited_value)
    _add_at_numba(out, j_3, a_3 * deposited_value)
    return out


@nb.njit(cache=True)
def deposit_antisymmetric(deposited_value, grid_particle_parameters):
    return _deposit(deposited_value, grid_particle_parameters, symmetric=False)


@nb.njit(cache=True)
def deposit_symmetric(deposited_value, grid_particle_parameters):
    return _deposit(deposited_value, grid_particle_parameters, symmetric=True)


@nb.njit(cache=True)
def _add_at_numba(out, idx, input_value):
    for i in np.arange(len(idx)):
        out[idx[i]] += input_value[i]


@nb.njit(parallel=True)
def interpolate_antisymmetric(value, grid_particle_parameters):
    _, j_1, j_2, j_3, sgn, a_1, a_2, a_3 = grid_particle_parameters
    return sgn * a_1 * value[j_1] + a_2 * value[j_2] + a_3 * value[j_3]


@nb.njit(parallel=True)
def interpolate_symmetric(value, grid_particle_parameters):
    _, j_1, j_2, j_3, _, a_1, a_2, a_3 = grid_particle_parameters
    return a_1 * value[j_1] + a_2 * value[j_2] + a_3 * value[j_3]


@nb.njit
def interpolate_noisereductor(
    noise_amplitude, transverse_coord, transverse_step,
):
    transverse_coord_normalized = transverse_coord / transverse_step
    cell_idx = np.floor(transverse_coord_normalized + 0.5).astype(np.int64)
    local_coord = transverse_coord_normalized - cell_idx
    local_coord_z = local_coord[cell_idx != 0]
    cell_idx_z = cell_idx[cell_idx != 0]

    noise_func = np.pi * np.sin(np.pi * local_coord_z) + np.cos(np.pi * local_coord_z) / cell_idx_z
    noise_func /= (np.pi ** 2 + 1 / cell_idx_z ** 2)
    result = np.zeros_like(transverse_coord)
    result[cell_idx != 0] = noise_amplitude[cell_idx_z] * transverse_step * noise_func
    return result
